# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a BSD-style license.

import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import quantize_model


def build_rope_half_pt(seq_len: int, head_dim: int, base: float, device="cpu", dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos()[None].to(dtype)
    sin = freqs.sin()[None].to(dtype)
    return cos, sin


class LayaBackboneClean(nn.Module):
    def __init__(self, m):
        super().__init__()
        enc = m.encoder
        cfg = enc.config
        self.embeddings = enc.embeddings.tok_embeddings
        self.emb_norm_w = enc.embeddings.norm.weight
        self.final_norm_w = enc.final_norm.weight
        self.eps = float(cfg.layer_norm_eps)
        self.layers = enc.layers
        self.layer_types = cfg.layer_types
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.type_emb = m.type_emb
        self.head_layers = m.head.layers
        self.register_buffer("zero_bias", torch.zeros(self.hidden_size))

    @staticmethod
    def _rope(x, cos, sin):
        x1, x2 = torch.tensor_split(x, (32,), dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def forward(self, input_ids, qtype, sliding_mask,
                cos_half_full, sin_half_full, cos_half_slide, sin_half_slide):
        cos_full = torch.cat((cos_half_full, cos_half_full), dim=-1).unsqueeze(1)
        sin_full = torch.cat((sin_half_full, sin_half_full), dim=-1).unsqueeze(1)
        cos_slide = torch.cat((cos_half_slide, cos_half_slide), dim=-1).unsqueeze(1)
        sin_slide = torch.cat((sin_half_slide, sin_half_slide), dim=-1).unsqueeze(1)

        # 1. Embeddings
        h = self.embeddings(input_ids)
        h = F.layer_norm(h, (self.hidden_size,), weight=self.emb_norm_w, bias=self.zero_bias, eps=self.eps)

        # 2. ModernBERT layers (28 layers)
        for i, layer in enumerate(self.layers):
            is_slide = (self.layer_types[i] == "sliding_attention")
            cos = cos_slide if is_slide else cos_full
            sin = sin_slide if is_slide else sin_full
            mask = sliding_mask if is_slide else None

            h_attn = h if i == 0 else F.layer_norm(h, (self.hidden_size,), weight=layer.attn_norm.weight, bias=self.zero_bias, eps=self.eps)
            qkv = layer.attn.Wqkv(h_attn).reshape(1, -1, 3, self.num_heads, self.head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)

            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)

            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            o = o.transpose(1, 2).reshape(1, -1, self.hidden_size)
            o = layer.attn.Wo(o)
            h = h + o

            h_mlp = F.layer_norm(h, (self.hidden_size,), weight=layer.mlp_norm.weight, bias=self.zero_bias, eps=self.eps)
            wi = layer.mlp.Wi(h_mlp)
            i1, i2 = torch.chunk(wi, 2, dim=-1)
            mlp_out = layer.mlp.Wo(F.gelu(i1) * i2)
            h = h + mlp_out

        h = F.layer_norm(h, (self.hidden_size,), weight=self.final_norm_w, bias=self.zero_bias, eps=self.eps)
        # Add question type embedding
        h = h + self.type_emb(qtype)

        # 3. Head layers (2 TransformerEncoder layers with ReLU)
        for hl in self.head_layers:
            hn1 = F.layer_norm(h, (self.hidden_size,), weight=hl.norm1.weight, bias=hl.norm1.bias, eps=self.eps)
            qkv = F.linear(hn1, hl.self_attn.in_proj_weight, hl.self_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).reshape(1, -1, self.hidden_size)
            attn_out = F.linear(attn_out, hl.self_attn.out_proj.weight, hl.self_attn.out_proj.bias)
            h = h + attn_out

            hn2 = F.layer_norm(h, (self.hidden_size,), weight=hl.norm2.weight, bias=hl.norm2.bias, eps=self.eps)
            ffn = hl.linear2(F.relu(hl.linear1(hn2)))
            h = h + ffn

        return h


class ScorerExport(nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, m):
        # m: [1, K, 1024]
        return self.scorer(m).squeeze(-1)


class ActHeadExport(nn.Module):
    def __init__(self, act_head):
        super().__init__()
        self.act_head = act_head

    def forward(self, feats):
        # feats: [1, 1028]
        return self.act_head(feats)


def extract_laya_tokenizer(tokenizer_dir: str, output_dir: str):
    tok_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tok_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pre_tok = data.get("pre_tokenizer", {})
    is_metaspace = (pre_tok.get("type") == "Metaspace")
    use_byte_encoder = not is_metaspace

    vocab = data.get("model", {}).get("vocab", {})
    added = data.get("added_tokens", [])
    all_vocab = dict(vocab)
    for item in added:
        all_vocab[item["content"]] = item["id"]

    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8", newline="\n") as f:
        for token, _ in sorted(all_vocab.items(), key=lambda x: x[1]):
            esc_token = token.replace("\n", "\\n").replace("\r", "\\r")
            f.write(f"{esc_token}\n")

    merges = data.get("model", {}).get("merges", [])
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8", newline="\n") as f:
        for m in merges:
            if isinstance(m, list) and len(m) == 2:
                f.write(f"{m[0]} {m[1]}\n")
            elif isinstance(m, str):
                f.write(f"{m}\n")

    print(f"[Tokenizer] Extracted vocab ({len(all_vocab)} tokens) to {vocab_path}")
    print(f"[Tokenizer] Extracted merges ({len(merges)} pairs) to {merges_path}")
    print(f"[Tokenizer] use_byte_encoder = {use_byte_encoder}")
    return use_byte_encoder


def export_laya_model(model_dir: str, output_dir: str, int8_dir: str, bits: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp_ts")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"=== 1. Loading model from {model_dir} ===")
    sys.path.insert(0, model_dir)
    from rl_agent_api import RLAgent
    agent = RLAgent(model_dir, device="cpu")
    model = agent.model.float().eval()
    cfg = agent.cfg

    print("=== 2. Extracting Tokenizer ===")
    tok_dir = os.path.join(model_dir, "tokenizer")
    use_byte_encoder = extract_laya_tokenizer(tok_dir, output_dir)

    print("=== 3. Wrapping & Numerically Checking Modules ===")
    clean_backbone = LayaBackboneClean(model).eval()
    scorer_mod = ScorerExport(model.scorer).eval()
    act_mod = ActHeadExport(model.act_head).eval()

    hidden_size = clean_backbone.hidden_size
    head_dim = clean_backbone.head_dim
    num_heads = clean_backbone.num_heads
    enc_cfg = model.encoder.config
    rope_cfg = getattr(enc_cfg, 'rope_parameters', {})
    rope_theta_full = float(rope_cfg.get('full_attention', {}).get('rope_theta', 160000.0))
    rope_theta_slide = float(rope_cfg.get('sliding_attention', {}).get('rope_theta', 160000.0 if hidden_size == 768 else 10000.0))
    sliding_win = int(getattr(enc_cfg, 'local_attention', 128))

    L0 = 32
    sample_ids = torch.randint(0, 1000, (1, L0))
    sample_qtype = torch.tensor([0], dtype=torch.long)
    pos = torch.arange(L0)
    dist = (pos[:, None] - pos[None, :]).abs()
    sample_mask = torch.where(dist <= (sliding_win // 2), 0.0, -1e4).view(1, 1, L0, L0)
    c_f, s_f = build_rope_half_pt(L0, head_dim, rope_theta_full)
    c_s, s_s = build_rope_half_pt(L0, head_dim, rope_theta_slide)

    with torch.no_grad():
        ref_h = model.encoder(input_ids=sample_ids).last_hidden_state + model.type_emb(sample_qtype)
        for hl in model.head.layers:
            ref_h = hl(ref_h)
        my_h = clean_backbone(sample_ids, sample_qtype, sample_mask, c_f, s_f, c_s, s_s)
        err_backbone = (my_h - ref_h).abs().max().item()
        print(f"[Check] Backbone max absolute error vs PyTorch: {err_backbone:.6e}")
        assert err_backbone < 1e-3, "Backbone numerical mismatch!"

        sample_m = torch.randn(1, 3, hidden_size)
        ref_logits = model.scorer(sample_m).squeeze(-1)
        my_logits = scorer_mod(sample_m)
        err_scorer = (my_logits - ref_logits).abs().max().item()
        print(f"[Check] Scorer max absolute error vs PyTorch: {err_scorer:.6e}")
        assert err_scorer < 1e-5, "Scorer numerical mismatch!"

        sample_act = torch.randn(1, hidden_size + 4)
        ref_act = model.act_head(sample_act)
        my_act = act_mod(sample_act)
        err_act = (my_act - ref_act).abs().max().item()
        print(f"[Check] ActHead max absolute error vs PyTorch: {err_act:.6e}")
        assert err_act < 1e-5, "ActHead numerical mismatch!"

    print("=== 4. Tracing TorchScript Modules ===")
    ts_backbone_path = os.path.join(temp_dir, "backbone.ts")
    ts_scorer_path = os.path.join(temp_dir, "scorer.ts")
    ts_act_path = os.path.join(temp_dir, "act_head.ts")

    with torch.no_grad():
        ts_backbone = torch.jit.trace(
            clean_backbone,
            (sample_ids, sample_qtype, sample_mask, c_f, s_f, c_s, s_s),
            check_trace=False
        )
        torch.jit.save(ts_backbone, ts_backbone_path)

        ts_scorer = torch.jit.trace(scorer_mod, (sample_m,), check_trace=False)
        torch.jit.save(ts_scorer, ts_scorer_path)

        ts_act = torch.jit.trace(act_mod, (sample_act,), check_trace=False)
        torch.jit.save(ts_act, ts_act_path)

    print("TorchScript modules saved.")

    print("=== 5. Converting to NCNN with PNNX ===")
    bb_param = os.path.join(output_dir, "laya_backbone.ncnn.param")
    bb_bin = os.path.join(output_dir, "laya_backbone.ncnn.bin")
    cmd_bb = (
        f'pnnx "{ts_backbone_path}" '
        f'"inputshape=[1,32]i64,[1]i64,[1,1,32,32]f32,[1,32,32]f32,[1,32,32]f32,[1,32,32]f32,[1,32,32]f32" '
        f'"inputshape2=[1,64]i64,[1]i64,[1,1,64,64]f32,[1,64,32]f32,[1,64,32]f32,[1,64,32]f32,[1,64,32]f32" '
        f'fp16=0 device=cpu ncnnparam="{bb_param}" ncnnbin="{bb_bin}"'
    )
    print("Running:", cmd_bb)
    subprocess.run(cmd_bb, shell=True, check=True)

    sc_param = os.path.join(output_dir, "laya_scorer.ncnn.param")
    sc_bin = os.path.join(output_dir, "laya_scorer.ncnn.bin")
    cmd_sc = (
        f'pnnx "{ts_scorer_path}" "inputshape=[1,3,{hidden_size}]f32" "inputshape2=[1,5,{hidden_size}]f32" '
        f'fp16=0 device=cpu ncnnparam="{sc_param}" ncnnbin="{sc_bin}"'
    )
    print("Running:", cmd_sc)
    subprocess.run(cmd_sc, shell=True, check=True)

    act_param = os.path.join(output_dir, "laya_act_head.ncnn.param")
    act_bin = os.path.join(output_dir, "laya_act_head.ncnn.bin")
    cmd_act = (
        f'pnnx "{ts_act_path}" "inputshape=[1,{hidden_size + 4}]f32" '
        f'fp16=0 device=cpu ncnnparam="{act_param}" ncnnbin="{act_bin}"'
    )
    print("Running:", cmd_act)
    subprocess.run(cmd_act, shell=True, check=True)

    # Clean up temp TorchScript and generated pnnx artifacts
    shutil.rmtree(temp_dir, ignore_errors=True)
    for junk in os.listdir("."):
        if junk.startswith("backbone.") or junk.startswith("backbone_") or \
           junk.startswith("scorer.") or junk.startswith("scorer_") or \
           junk.startswith("act_head.") or junk.startswith("act_head_"):
            try:
                os.remove(junk)
            except OSError:
                pass

    print("=== 6. Generating model.json ===")
    tok_cfg_file = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    tok_cfg = {}
    if os.path.exists(tok_cfg_file):
        with open(tok_cfg_file, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

    cls_tok = tok_cfg.get("cls_token") or tok_cfg.get("bos_token") or "[CLS]"
    sep_tok = tok_cfg.get("sep_token") or tok_cfg.get("eos_token") or "[SEP]"
    pad_tok = tok_cfg.get("pad_token") or "[PAD]"
    mask_tok = tok_cfg.get("mask_token") or "[MASK]"
    unk_tok = tok_cfg.get("unk_token") or "[UNK]"

    model_json = {
        "model_type": "laya",
        "params": {
            "backbone_param": "laya_backbone.ncnn.param",
            "backbone_bin": "laya_backbone.ncnn.bin",
            "scorer_param": "laya_scorer.ncnn.param",
            "scorer_bin": "laya_scorer.ncnn.bin",
            "act_head_param": "laya_act_head.ncnn.param",
            "act_head_bin": "laya_act_head.ncnn.bin"
        },
        "tokenizer": {
            "type": "bpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "cls": cls_tok,
            "sep": sep_tok,
            "pad": pad_tok,
            "mask": mask_tok,
            "unk": unk_tok,
            "use_byte_encoder": use_byte_encoder
        },
        "setting": {
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "max_len": cfg.get("max_len", 512),
            "head_max_len": cfg.get("head_max_len", 192),
            "rope_theta_full": rope_theta_full,
            "rope_theta_slide": rope_theta_slide,
            "sliding_window": sliding_win,
            "temperature": cfg.get("temperature", [1.6369, 1.2514, 1.9834]),
            "temperature_by_options": cfg.get("temperature_by_options", {})
        }
    }

    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=4)
    print(f"Generated {model_json_path}")

    print("=== 7. Quantizing to Int8 ===")
    tool_path = quantize_model.find_ncnnllm2int()
    if not tool_path:
        print("Warning: ncnnllm2int not found! Skipping int8 quantization.")
    else:
        print(f"Found ncnnllm2int at {tool_path}")
        quantize_model.quantize_model_dir(output_dir, int8_dir, tool_path, bits=bits, block=64, method="minmax")
        print(f"Int8 model successfully generated in {int8_dir}")

    print("\n[SUCCESS] LAYA export complete!")
    print(f"  BF16/FP32 model: {output_dir}")
    print(f"  INT8 model:      {int8_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export LAYA model to NCNN (BF16 & INT8)")
    parser.add_argument("--model-dir", "-m", type=str, default="laya_hf", help="Source Hugging Face LAYA directory")
    parser.add_argument("--output-dir", "-o", type=str, default="assets/laya", help="Destination NCNN baseline directory")
    parser.add_argument("--int8-dir", type=str, default="assets/laya_int8", help="Destination NCNN INT8 directory")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8], help="Quantization bits (default: 8)")
    args = parser.parse_args()

    export_laya_model(args.model_dir, args.output_dir, args.int8_dir, args.bits)


if __name__ == "__main__":
    main()
