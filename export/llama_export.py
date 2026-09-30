# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Export a plain LLaMA-architecture causal LM (e.g. MiniCPM5-1B / 2B) to ncnn.

Pipeline (matches the project convention):

  1. load the HF model in **fp32**
  2. wrap it into three traceable modules (embed_token / decoder / proj_out)
  3. assert the wrappers agree with the reference PyTorch implementation
  4. torch.jit.trace -> pnnx with ``fp16=0``  => **fp32** ncnn param/bin
  5. cross-check the fp32 ncnn graph against the torch wrapper
  6. rewrite SDPA into its KV-cache form (add_kvcache.py)
  7. ncnnoptimize ``flag=2`` -> **bf16** weights
  8. ncnnllm2int -> separate **int8** directory

Usage::

    python export/llama_export.py -m models/minicpm5_1b \
        -o assets/minicpm5_1b --int8-dir assets/minicpm5_1b_int8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import add_kvcache           # noqa: E402
import bf16_convert          # noqa: E402
import quantize_model        # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def build_rope_half(seq_len, head_dim, base, device="cpu", dtype=torch.float32):
    """cos/sin at HALF width [1, L, head_dim//2] for interleaved-half RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos()[None].to(dtype), freqs.sin()[None].to(dtype)


def build_causal_mask(seq_len, device="cpu", dtype=torch.float32):
    m = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None]


def maxerr(a, b):
    return (a.float() - b.float()).abs().max().item()


def get_rope_theta(cfg):
    """ rope_theta lives in cfg.rope_theta for most models, but some configs
    (MiniCPM5 among them) stash it under cfg.rope_scaling instead."""
    theta = getattr(cfg, "rope_theta", None)
    if theta is None:
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict):
            theta = rs.get("rope_theta", None)
    if theta is None:
        theta = 10000.0
    return float(theta)


def get_eps(cfg, layer):
    for cand in (getattr(cfg, "rms_norm_eps", None),
                 getattr(cfg, "layer_norm_eps", None),
                 getattr(layer.input_layernorm, "variance_epsilon", None),
                 getattr(layer.input_layernorm, "eps", None)):
        if cand is not None:
            return float(cand)
    return 1e-6


# ---------------------------------------------------------------------------
# export wrappers
# ---------------------------------------------------------------------------
class EmbedExport(nn.Module):
    def __init__(self, model, scale=None):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.scale = float(scale) if scale else None

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        if self.scale:
            x = x * self.scale
        return x


class LMHeadExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden):
        return self.lm_head(hidden)


class DecoderExport(nn.Module):
    """Full-sequence decoder. Input mirrors what ncnn_llm_gpt feeds in."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.cfg = model.config
        # NOTE: do **not** cache submodules here (`self.layers = self.model.layers`).
        # Doing so registers the same ModuleList under a second attribute name and
        # pnnx then fails to resolve the parameters as constants: it emits a second
        # Gemm input carrying the weight (MemoryData) while ALSO baking the weight
        # into the layer. ncnn's Gemm reads `A0 = bottom_blobs[0]` when constantA=0,
        # so the weight would be consumed as the A matrix and the hidden-state shape
        # collapses from (L, hidden) to something like (hidden, hidden).
        # See export/probe2_wrapper.py and export/probe3_llama.py.
        self.num_layers = len(self.model.layers)
        self.hidden_size = int(self.cfg.hidden_size)
        self.num_q = int(self.cfg.num_attention_heads)
        self.num_kv = int(getattr(self.cfg, "num_key_value_heads", self.num_q))
        self.head_dim = int(getattr(self.cfg, "head_dim", None) or
                            self.hidden_size // self.num_q)
        self.eps = get_eps(self.cfg, self.model.layers[0])

    @staticmethod
    def _rms(x, weight, eps):
        # maps 1:1 onto ncnn's RMSNorm layer
        return F.rms_norm(x, (weight.shape[0],), weight, eps)

    @staticmethod
    def _rope(x, cos, sin, half):
        """Non-interleaved (half-split) RoPE.

        Written in exactly the same form as ncnn's RotaryEmbed non-interleaved
        branch and HF's ``rotate_half`` so that

            out[:half]  = x1 * cos - x2 * sin
            out[half:]  = x2 * cos + x1 * sin

        with cos/sin at HALF width [1, L, half] - which is what ncnn_llm_gpt
        hands in via in2/in3 (``generate_rope_embed_cache`` builds a
        ``Mat(head_dim/2, seqlen)``).

        Deliberately does **not** duplicate cos/sin up to the full head width
        inside the graph: doing so leaves stray Concat/ExpandDims nodes behind
        and pnnx then produces a decoder whose hidden-state layout no longer
        matches the runtime's.
        """
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    def forward(self, hidden, attn_mask, cos_half, sin_half):
        half = self.head_dim // 2
        for layer in self.model.layers:
            attn = layer.self_attn
            res = hidden
            h = self._rms(hidden, layer.input_layernorm.weight, self.eps)

            q = attn.q_proj(h).reshape(1, -1, self.num_q, self.head_dim).transpose(1, 2)
            k = attn.k_proj(h).reshape(1, -1, self.num_kv, self.head_dim).transpose(1, 2)
            v = attn.v_proj(h).reshape(1, -1, self.num_kv, self.head_dim).transpose(1, 2)

            q = self._rope(q, cos_half, sin_half, half)
            k = self._rope(k, cos_half, sin_half, half)

            # Do NOT tile k/v up to the query head count: ncnn's SDPA layer maps
            # groups itself (num_heads_per_group = num_q / num_kv), so keeping k/v
            # grouped keeps the KV cache num_kv times smaller. torch needs to be
            # told explicitly via enable_gqa.
            if self.num_kv != self.num_q:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                   enable_gqa=True)
            else:
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            o = o.transpose(1, 2).reshape(1, -1, self.num_q * self.head_dim)
            hidden = res + attn.o_proj(o)

            res = hidden
            h = self._rms(hidden, layer.post_attention_layernorm.weight, self.eps)
            hidden = res + layer.mlp(h)

        return self._rms(hidden, self.model.norm.weight, self.eps)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def write_tokenizer(model_dir, output_dir):
    tok_json = os.path.join(model_dir, "tokenizer.json")
    with open(tok_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = {token: int(token_id) for token, token_id in data["model"]["vocab"].items()}
    by_id = {token_id: token for token, token_id in vocab.items()}
    added = []
    next_id = max(by_id, default=-1) + 1
    for entry in data.get("added_tokens", []):
        token = entry.get("content", "")
        if not token:
            continue
        token_id = entry.get("id")
        if token_id is None:
            token_id = vocab.get(token, next_id)
        token_id = int(token_id)
        if token in vocab and vocab[token] != token_id:
            raise ValueError(
                f"tokenizer ID conflict for {token!r}: "
                f"vocab={vocab[token]}, added_tokens={token_id}")
        if token_id in by_id and by_id[token_id] != token:
            raise ValueError(
                f"tokenizer ID {token_id} is assigned to both "
                f"{by_id[token_id]!r} and {token!r}")
        vocab[token] = token_id
        by_id[token_id] = token
        added.append(token)
        next_id = max(next_id, token_id + 1)

    max_id = max(by_id, default=-1)
    missing = [i for i in range(max_id + 1) if i not in by_id]
    if missing:
        print(f"[tokenizer] WARNING: vocab ids are not contiguous ({len(missing)} gaps)")

    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8", newline="\n") as f:
        for token_id in range(max_id + 1):
            token = by_id.get(token_id, f"<|ncnn_unused_{token_id}|>")
            f.write(token.replace("\n", "\\n").replace("\r", "\\r") + "\n")

    merges = data["model"].get("merges", [])
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8", newline="\n") as f:
        for m in merges:
            if isinstance(m, list) and len(m) == 2:
                f.write(f"{m[0]} {m[1]}\n")
            elif isinstance(m, str):
                f.write(m + "\n")

    # byte-level BPE iff the vocabulary uses the unicode byte mapping (e.g. 'Ġ')
    use_byte_encoder = any("Ġ" in t for t in vocab)
    print(f"[tokenizer] vocab={len(by_id)} merges={len(merges)} use_byte_encoder={use_byte_encoder}")

    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    tok_cfg = {}
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

    # Some checkpoints (MiniCPM5, for example) use both </s> and a chat
    # boundary token as valid generation terminators. Keep the numeric list
    # from the model config so the native runtime can stop on either one.
    eos_ids = []
    model_cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        raw_eos_ids = model_cfg.get("eos_token_id")
        if raw_eos_ids is None and isinstance(model_cfg.get("text_config"), dict):
            raw_eos_ids = model_cfg["text_config"].get("eos_token_id")
        if isinstance(raw_eos_ids, int):
            raw_eos_ids = [raw_eos_ids]
        if isinstance(raw_eos_ids, list):
            eos_ids = [int(x) for x in raw_eos_ids]

    if not eos_ids:
        eos_token = tok_cfg.get("eos_token")
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content")
        if eos_token and eos_token in vocab:
            eos_ids = [int(vocab[eos_token])]

    def tok(key):
        v = tok_cfg.get(key)
        if isinstance(v, dict):
            v = v.get("content")
        return v or ""

    return {
        "use_byte_encoder": use_byte_encoder,
        "bos": tok("bos_token"),
        "eos": tok("eos_token") or "</s>",
        "eos_ids": eos_ids,
        "pad": tok("pad_token"),
        "unk": tok("unk_token") or "<unk>",
        "added": added,
    }


# ---------------------------------------------------------------------------
# ncnn cross-check
# ---------------------------------------------------------------------------
def ncnn_infer(param, rng_bin, inputs, out_name="out0"):
    """Run an ncnn param/bin pair, in a **child process**.

    torch and ncnn each bring their own OpenMP runtime; importing both in the
    same interpreter hangs on Windows as soon as ncnn starts computing.  See
    export/ncnn_run.py for the worker.
    """
    import tempfile

    worker = os.path.join(HERE, "ncnn_run.py")
    tmpdir = tempfile.mkdtemp(prefix="ncnn_run_")
    in_npz = os.path.join(tmpdir, "in.npz")
    out_npz = os.path.join(tmpdir, "out.npz")
    np.savez(in_npz, **inputs)
    cmd = [sys.executable, "-u", worker,
           "--param", os.path.abspath(param),
           "--bin", os.path.abspath(rng_bin),
           "--inputs", in_npz, "--outputs", out_npz,
           "--out-name", out_name]
    subprocess.run(cmd, check=True)
    res = np.load(out_npz)[out_name]
    shutil.rmtree(tmpdir, ignore_errors=True)
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def export(model_dir, output_dir, int8_dir, prefix, model_type, bits=8,
           do_bf16=True, do_int8=True, cross_check=True, reuse_fp32=False,
           keep_fp32=False):
    os.makedirs(output_dir, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    hidden_size = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden_size // cfg.num_attention_heads)
    num_layers = int(cfg.num_hidden_layers)
    num_kv = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    rope_theta = get_rope_theta(cfg)
    embed_scale = getattr(cfg, "scale_emb", None)

    embed_param = os.path.join(output_dir, f"{prefix}_embed_token.ncnn.param")
    embed_bin = os.path.join(output_dir, f"{prefix}_embed_token.ncnn.bin")
    dec_param = os.path.join(output_dir, f"{prefix}_decoder.ncnn.param")
    dec_bin = os.path.join(output_dir, f"{prefix}_decoder.ncnn.bin")
    head_param = os.path.join(output_dir, f"{prefix}_proj_out.ncnn.param")
    head_bin = os.path.join(output_dir, f"{prefix}_proj_out.ncnn.bin")
    have_fp32 = all(os.path.isfile(p) and os.path.getsize(p) > 0
                    for p in (embed_param, embed_bin, dec_param, dec_bin,
                              head_param, head_bin))

    print(f"[meta] hidden={hidden_size} layers={num_layers} head_dim={head_dim} "
          f"q={cfg.num_attention_heads} kv={num_kv} rope_theta={rope_theta}")
    if embed_scale:
        print(f"[meta] embedding scale (scale_emb) = {embed_scale}")

    # reference tensors used to cross-check the ncnn graph; cached so the later
    # steps can be re-run without reloading the torch model
    ref_cache = os.path.join(output_dir, "_check_ref.npz")

    print("=== 2. Tokenizer ===")
    tok_info = write_tokenizer(model_dir, output_dir)

    check_inputs = None
    check_target = None

    if reuse_fp32 and have_fp32 and os.path.isfile(ref_cache):
        print("=== 3-5. Reusing existing fp32 ncnn export, skipping torch steps ===")
        z = np.load(ref_cache)
        check_inputs = {"in0": z["in0"], "in1": z["in1"], "in2": z["in2"], "in3": z["in3"]}
        check_target = z["target"]
    else:
        print(f"=== 3. Loading {model_dir} (fp32, cpu) ===")
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        model.eval()

        print("=== 4. Wrapping & Numerically Checking ===")
        w_embed = EmbedExport(model, embed_scale).eval()
        w_dec = DecoderExport(model).eval()
        w_head = LMHeadExport(model).eval()

        L0 = 32
        torch.manual_seed(0)
        ids = torch.randint(0, min(getattr(cfg, "vocab_size", 100000), 50000), (1, L0))
        embeds = w_embed(ids)
        mask = build_causal_mask(L0)
        cos_h, sin_h = build_rope_half(L0, head_dim, rope_theta)

        with torch.no_grad():
            ref = model.model(input_ids=ids).last_hidden_state
            out = w_dec(embeds, mask, cos_h, sin_h)
            err = maxerr(out, ref)
            print(f"  [check] decoder vs PyTorch reference: max|d| = {err:.3e}")
            assert err < 2e-3, f"decoder numerical mismatch too large: {err}"

            ref_logits = model(input_ids=ids).logits[:, -1]
            head_out = w_head(out[:, -1:])
            err_head = maxerr(head_out, ref_logits)
            print(f"  [check] lm_head vs PyTorch reference: max|d| = {err_head:.3e}")
            assert err_head < 5e-3, f"lm_head numerical mismatch too large: {err_head}"

        print("=== 5. Tracing TorchScript ===")
        # keep every intermediate outside the repository tree
        temp_dir = os.path.join(tempfile.gettempdir(),
                                "ncnn_llm_export_" + os.path.basename(output_dir))
        os.makedirs(temp_dir, exist_ok=True)
        ts_embed = os.path.join(temp_dir, "embed_token.ts")
        ts_dec = os.path.join(temp_dir, "decoder.ts")
        ts_head = os.path.join(temp_dir, "proj_out.ts")
        with torch.no_grad():
            torch.jit.trace(w_embed, (ids,), check_trace=False).save(ts_embed)
            torch.jit.trace(w_dec, (embeds, mask, cos_h, sin_h), check_trace=False).save(ts_dec)
            torch.jit.trace(w_head, (out[:, -1:]), check_trace=False).save(ts_head)
        print("  traced 3 modules into " + temp_dir)

        print("=== 6. pnnx (fp32) ===")
        half = head_dim // 2
        L1, L2 = 32, 64
        cmds = [
            (f'pnnx "{ts_embed}" "inputshape=[1,{L1}]i64" "inputshape2=[1,{L2}]i64" '
             f'fp16=0 device=cpu ncnnparam="{embed_param}" ncnnbin="{embed_bin}"'),
            (f'pnnx "{ts_dec}" '
             f'"inputshape=[1,{L1},{hidden_size}]f32,[1,1,{L1},{L1}]f32,[1,{L1},{half}]f32,[1,{L1},{half}]f32" '
             f'"inputshape2=[1,{L2},{hidden_size}]f32,[1,1,{L2},{L2}]f32,[1,{L2},{half}]f32,[1,{L2},{half}]f32" '
             f'fp16=0 device=cpu ncnnparam="{dec_param}" ncnnbin="{dec_bin}"'),
            (f'pnnx "{ts_head}" "inputshape=[1,1,{hidden_size}]f32" '
             f'fp16=0 device=cpu ncnnparam="{head_param}" ncnnbin="{head_bin}"'),
        ]
        for cmd in cmds:
            print("Running:", cmd, flush=True)
            subprocess.run(cmd, shell=True, check=True)

        # pnnx intermediates are huge (tens of GB for the bigger models)
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.isdir(temp_dir):
            print(f"  WARNING: could not remove temp dir {temp_dir}")
        else:
            print(f"  cleaned temp dir {temp_dir}")

        # every tensor below may still be attached to the autograd graph
        # (they were built outside torch.no_grad), so detach before numpy()
        with torch.no_grad():
            check_inputs = {
                "in0": embeds.detach().squeeze(0).numpy(),
                "in1": mask.detach().squeeze(0).squeeze(0).numpy(),
                "in2": cos_h.detach().squeeze(0).numpy(),
                "in3": sin_h.detach().squeeze(0).numpy(),
            }
            check_target = out.detach().squeeze(0).numpy()
        np.savez(ref_cache, target=check_target, **check_inputs)
        del model

    # 6b. sanity check the produced graph has the operators we rely on.
    # The Gemm checks below are load-bearing: pnnx happily emits a *second* Gemm
    # input carrying the weight (plus a MemoryData node) while still baking the
    # weight into the layer, and ncnn's Gemm then uses that blob as the A matrix.
    def _graph_stats(path):
        from collections import Counter
        types = Counter()
        nin = Counter()
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh.read().splitlines()[1:]:
                f = line.split()
                if len(f) < 4:
                    continue
                types[f[0]] += 1
                if f[0] == "Gemm":
                    nin[int(f[2])] += 1
        return types, nin

    def _assert_baked(path, what):
        types, nin = _graph_stats(path)
        print(f"  [graph] {what}: Gemm nin={dict(nin)} MemoryData={types.get('MemoryData', 0)}")
        if types.get("MemoryData", 0) != 0 or set(nin) - {1}:
            raise RuntimeError(
                f"{what}: linear weights were not folded into the Gemm layers "
                f"(Gemm nin={dict(nin)}, MemoryData={types.get('MemoryData', 0)}). "
                f"Almost always caused by re-binding a submodule to a new attribute "
                f"in the export wrapper - see the note in DecoderExport.__init__.")

    _assert_baked(dec_param, "decoder")
    _assert_baked(embed_param, "embed_token")
    _assert_baked(head_param, "proj_out")

    graph = open(dec_param, "r", encoding="utf-8").read()
    for op in ("SDPA", "RotaryEmbed", "RMSNorm"):
        n = graph.count("\n" + op) + int(graph.startswith(op))
        print(f"  [graph] {op}: {n}")
        if op == "SDPA" and n != num_layers:
            raise RuntimeError(f"expected {num_layers} SDPA layers, found {n}")
    print(f"  [graph] Input layers: {graph.count(chr(10) + 'Input')} (expect 4)")

    if cross_check and check_target is not None:
        print("=== 6c. Cross-checking fp32 ncnn graph vs torch ===")
        got = ncnn_infer(dec_param, dec_bin, check_inputs)
        err_ncnn = float(np.abs(got - check_target).max())
        print(f"  [check] ncnn(f32) decoder vs torch: max|d| = {err_ncnn:.3e}")
        assert err_ncnn < 2e-3, f"ncnn graph deviates from torch too much: {err_ncnn}"

    print("=== 7. Adding KV cache ===")
    info = add_kvcache.add_kvcache_to_param(dec_param)
    print(f"  SDPA rewritten: {info['sdpa']}, header {info['old']} -> {info['new']}")

    # after rewriting we re-verify: an empty cache must reproduce the
    # full-sequence result exactly
    if cross_check and check_target is not None:
        got2 = ncnn_infer(dec_param, dec_bin, check_inputs)
        err_kv = float(np.abs(got2 - check_target).max())
        print(f"  [check] kvcache decoder (empty cache) vs torch: max|d| = {err_kv:.3e}")
        assert err_kv < 2e-3, f"kvcache graph deviates too much: {err_kv}"

    print("=== 8. model.json ===")
    params = {
        "embed_token_param": f"{prefix}_embed_token.ncnn.param",
        "embed_token_bin": f"{prefix}_embed_token.ncnn.bin",
        "decoder_param": f"{prefix}_decoder.ncnn.param",
        "decoder_bin": f"{prefix}_decoder.ncnn.bin",
        "proj_out_param": f"{prefix}_proj_out.ncnn.param",
        "proj_out_bin": f"{prefix}_proj_out.ncnn.bin",
    }
    function_setting = {"type": "close"}
    if model_type == "minicpm5":
        function_setting = {
            "type": "tool_call",
            "tool_call_id": "<function",
            "tool_call_end_id": "</function>",
        }
    model_json = {
        "type": model_type,
        "params": params,
        "tokenizer": {
            "type": "bbpe" if tok_info["use_byte_encoder"] else "bpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "bos": tok_info["bos"],
            "eos": tok_info["eos"],
            "eos_ids": tok_info["eos_ids"],
            "pad": tok_info["pad"],
            "unk": tok_info["unk"],
            "additional_special_tokens": tok_info["added"],
        },
        "setting": {
            "attn_cnt": num_layers,
            "rope": {
                "type": "RoPE",
                "rope_head_dim": head_dim,
                "rope_theta": rope_theta,
            },
            "vision": {"type": "close"},
            "functions": function_setting,
        },
    }
    with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)
    print(f"  wrote {os.path.join(output_dir, 'model.json')}")

    if keep_fp32:
        fp32_dir = output_dir + "_fp32"
        os.makedirs(fp32_dir, exist_ok=True)
        for name in list(params.values()) + ["vocab.txt", "merges.txt", "model.json"]:
            src = os.path.join(output_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(fp32_dir, name))
        print(f"  fp32 snapshot kept at {fp32_dir}")

    if do_bf16:
        print("=== 9. ncnnoptimize -> bf16 (flag=2) ===")
        tool = bf16_convert.find_ncnnoptimize()
        if not tool:
            raise RuntimeError("ncnnoptimize not found; build it with "
                               "-DNCNN_LLM_ENABLE_TOOLS=ON and copy it into tools/")
        print(f"  using {tool}")
        bf16_convert.convert_model_dir(output_dir, tool, flag=bf16_convert.BF16)

        # the size-ratio check above only proves the file shrank; run the graph
        # to prove it is still *correct*
        if cross_check and check_target is not None:
            print("=== 9b. Cross-checking bf16 ncnn graph vs torch ===")
            import check_graph
            m = check_graph.compare(dec_param, dec_bin, ref_cache)
            print(f"  [check] bf16 decoder: max|d|={m['max_abs']:.3e} "
                  f"rel_l2={m['rel_l2']:.3e} cos={m['cos']:.6f}")
            # when the upstream checkpoint is already bf16 the conversion is
            # exactly lossless and rel_l2 will be ~0
            if m["rel_l2"] > 0.02 or m["cos"] < 0.999:
                raise RuntimeError(
                    f"bf16 graph deviates too much from the fp32 reference: "
                    f"rel_l2={m['rel_l2']:.3e} cos={m['cos']:.6f}")

    if do_int8:
        print("=== 10. ncnnllm2int -> int8 ===")
        tool = quantize_model.find_ncnnllm2int()
        if not tool:
            print("  WARNING: ncnnllm2int not found, skipping int8")
        else:
            quantize_model.quantize_model_dir(output_dir, int8_dir, tool,
                                              bits=bits, block=64, method="minmax")
            print(f"  int8 model written to {int8_dir}")

            if cross_check and check_target is not None:
                print("=== 10b. Cross-checking int8 ncnn graph vs torch ===")
                import check_graph
                mi = check_graph.compare(
                    os.path.join(int8_dir, os.path.basename(dec_param)),
                    os.path.join(int8_dir, os.path.basename(dec_bin)),
                    ref_cache)
                print(f"  [check] int{dict(bits=bits)['bits']} decoder: "
                      f"max|d|={mi['max_abs']:.3e} rel_l2={mi['rel_l2']:.3e} "
                      f"cos={mi['cos']:.6f}")
                # 8-bit block quantised weights typically land around 3% rel_l2;
                # anything past ~10% means the quantisation went wrong
                if mi["rel_l2"] > 0.10 or mi["cos"] < 0.995:
                    raise RuntimeError(
                        f"int8 graph deviates too much: rel_l2={mi['rel_l2']:.3e} "
                        f"cos={mi['cos']:.6f}")

    print("\n[SUCCESS] export complete")
    print(f"  bf16 model: {output_dir}")
    if do_int8:
        print(f"  int8 model: {int8_dir}")


def main():
    ap = argparse.ArgumentParser(description="Export a LLaMA-arch model to ncnn (bf16 + int8).")
    ap.add_argument("--model-dir", "-m", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    ap.add_argument("--int8-dir", default="")
    ap.add_argument("--prefix", default="", help="file name prefix inside the output dir")
    ap.add_argument("--type", dest="model_type", default="llama")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--keep-fp32", action="store_true", help="keep a fp32 snapshot next to the bf16 one")
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--reuse-fp32", action="store_true",
                    help="skip torch/pnnx and reuse the fp32 param/bin already in the output dir")
    args = ap.parse_args()

    prefix = args.prefix or os.path.basename(args.output_dir)
    int8_dir = args.int8_dir or (args.output_dir.rstrip("/\\") + f"_int{args.bits}")

    export(args.model_dir, args.output_dir, int8_dir, prefix, args.model_type,
           bits=args.bits, do_bf16=not args.no_bf16, do_int8=not args.no_int8,
           keep_fp32=args.keep_fp32, cross_check=not args.no_cross_check,
           reuse_fp32=args.reuse_fp32)


if __name__ == "__main__":
    main()
