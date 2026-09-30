# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Export Qwen3.5 vision tower to ncnn."""

import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import torch
from safetensors import safe_open


def load_visual_state(model_dir):
    st = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if st_files:
        for f in st_files:
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    if "visual" in k:
                        st[k] = sf.get_tensor(k).float()
    else:
        pt_files = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
        for f in pt_files:
            sd = torch.load(f, map_location="cpu")
            for k, v in sd.items():
                if "visual" in k:
                    st[k] = v.float()
    return st


def build_vision_encoder_param(hidden_size, depth, intermediate_size, num_heads, head_dim, out_hidden_size):
    scale = 1.0 / math.sqrt(head_dim)
    layers = []
    layers.append("Input                    in0                      0 1 in0")
    layers.append("Input                    in1                      0 1 in1")
    layers.append("Input                    in2                      0 1 in2")

    cos_blobs = [f"cos_{i}" for i in range(depth * 2)]
    sin_blobs = [f"sin_{i}" for i in range(depth * 2)]
    layers.append(f"Split                    split_cos                1 {depth * 2} in2 {' '.join(cos_blobs)}")
    layers.append("Input                    in3                      0 1 in3")
    layers.append(f"Split                    split_sin                1 {depth * 2} in3 {' '.join(sin_blobs)}")

    layers.append("BinaryOp                 add_0                    2 1 in0 in1 blob_0")
    cur_blob = "blob_0"

    for b in range(depth):
        res1 = f"res1_{b}"
        branch1 = f"branch1_{b}"
        layers.append(f"Split                    split_attn_{b}           1 2 {cur_blob} {res1} {branch1}")

        ln1_out = f"ln1_{b}"
        layers.append(f"LayerNorm                ln_{b}_0                 1 1 {branch1} {ln1_out} 0={hidden_size} 1=1.000000e-06")

        qkv_out = f"qkv_{b}"
        layers.append(f"Gemm                     gemm_{b}_0               1 1 {ln1_out} {qkv_out} 3=1 5=1 6=1 8={3*hidden_size} 9={hidden_size} 10=4")

        reshape_qkv = f"rqkv_{b}"
        layers.append(f"Reshape                  reshape_{b}_0            1 1 {qkv_out} {reshape_qkv} 0={head_dim} 1={num_heads} 11=3 2=-1")

        permute_qkv = f"pqkv_{b}"
        layers.append(f"Permute                  permute_{b}_0            1 1 {reshape_qkv} {permute_qkv} 0=8")

        q_blob = f"q_{b}"
        k_blob = f"k_{b}"
        v_blob = f"v_{b}"
        layers.append(f"Slice                    unbind_{b}               1 3 {permute_qkv} {q_blob} {k_blob} {v_blob} -23300=3,-233,-233,-233")

        q_3d = f"q3d_{b}"
        k_3d = f"k3d_{b}"
        v_3d = f"v3d_{b}"
        layers.append(f"Reshape                  reshape_{b}_1            1 1 {v_blob} {v_3d} 0={head_dim} 1=-1 2={num_heads}")
        layers.append(f"Reshape                  reshape_{b}_2            1 1 {k_blob} {k_3d} 0={head_dim} 1=-1 2={num_heads}")
        layers.append(f"Reshape                  reshape_{b}_3            1 1 {q_blob} {q_3d} 0={head_dim} 1=-1 2={num_heads}")

        cos_q = cos_blobs[2 * b + 1]
        cos_k = cos_blobs[2 * b]
        sin_q = sin_blobs[2 * b + 1]
        sin_k = sin_blobs[2 * b]

        q_roped = f"qrope_{b}"
        k_roped = f"krope_{b}"
        layers.append(f"RotaryEmbed              rope_{b}_0               3 1 {q_3d} {cos_q} {sin_q} {q_roped}")
        layers.append(f"RotaryEmbed              rope_{b}_1               3 1 {k_3d} {cos_k} {sin_k} {k_roped}")

        sdpa_out = f"sdpa_{b}"
        layers.append(f"SDPA                     sdpa_{b}                 3 1 {q_roped} {k_roped} {v_3d} {sdpa_out} 6={scale:.6e}")

        permute_sdpa = f"psdpa_{b}"
        layers.append(f"Permute                  permute_{b}_1            1 1 {sdpa_out} {permute_sdpa} 0=2")

        reshape_sdpa = f"rsdpa_{b}"
        layers.append(f"Reshape                  reshape_{b}_4            1 1 {permute_sdpa} {reshape_sdpa} 0={hidden_size} 1=-1")

        proj_out = f"proj_{b}"
        layers.append(f"Gemm                     gemm_{b}_1               1 1 {reshape_sdpa} {proj_out} 3=1 5=1 6=1 8={hidden_size} 9={hidden_size} 10=4")

        attn_out = f"attn_out_{b}"
        layers.append(f"BinaryOp                 add_attn_{b}             2 1 {res1} {proj_out} {attn_out}")

        res2 = f"res2_{b}"
        branch2 = f"branch2_{b}"
        layers.append(f"Split                    split_mlp_{b}            1 2 {attn_out} {res2} {branch2}")

        ln2_out = f"ln2_{b}"
        layers.append(f"LayerNorm                ln_{b}_1                 1 1 {branch2} {ln2_out} 0={hidden_size} 1=1.000000e-06")

        fc1_out = f"fc1_{b}"
        layers.append(f"Gemm                     gemm_{b}_2               1 1 {ln2_out} {fc1_out} 3=1 5=1 6=1 8={intermediate_size} 9={hidden_size} 10=4")

        gelu_out = f"gelu_{b}"
        layers.append(f"GELU                     gelu_{b}_0               1 1 {fc1_out} {gelu_out} 0=1")

        fc2_out = f"fc2_{b}"
        layers.append(f"Gemm                     gemm_{b}_3               1 1 {gelu_out} {fc2_out} 3=1 5=1 6=1 8={hidden_size} 9={intermediate_size} 10=4")

        mlp_out = f"mlp_out_{b}"
        layers.append(f"BinaryOp                 add_mlp_{b}              2 1 {res2} {fc2_out} {mlp_out}")
        cur_blob = mlp_out

    # Merger
    ln_merger = "ln_merger"
    layers.append(f"LayerNorm                ln_merger                1 1 {cur_blob} {ln_merger} 0={hidden_size} 1=1.000000e-06")

    reshape_merger = "reshape_merger"
    layers.append(f"Reshape                  reshape_merger           1 1 {ln_merger} {reshape_merger} 0={hidden_size * 4} 1=-1")

    merger_fc1 = "merger_fc1"
    layers.append(f"Gemm                     gemm_merger_0            1 1 {reshape_merger} {merger_fc1} 3=1 5=1 6=1 8={hidden_size * 4} 9={hidden_size * 4} 10=4")

    gelu_merger = "gelu_merger"
    layers.append(f"GELU                     gelu_merger              1 1 {merger_fc1} {gelu_merger}")

    layers.append(f"Gemm                     gemm_merger_1            1 1 {gelu_merger} out0 3=1 5=1 6=1 8={out_hidden_size} 9={hidden_size * 4} 10=4")

    all_blobs = set()
    for l in layers:
        parts = l.split()
        in_cnt = int(parts[2])
        out_cnt = int(parts[3])
        all_blobs.update(parts[4:4+in_cnt+out_cnt])

    header = f"7767517\n{len(layers)} {len(all_blobs)}"
    return header + "\n" + "\n".join(layers) + "\n"


def build_vision_encoder_bin(st, depth):
    bin_parts = []
    def write_gemm(w, b):
        bin_parts.append(b"\x00\x00\x00\x00")
        bin_parts.append(w.contiguous().numpy().tobytes())
        bin_parts.append(b"\x00\x00\x00\x00")
        bin_parts.append(b.contiguous().numpy().tobytes())

    def write_ln(w, b):
        bin_parts.append(w.contiguous().numpy().tobytes())
        bin_parts.append(b.contiguous().numpy().tobytes())

    for b in range(depth):
        write_ln(st[f"model.visual.blocks.{b}.norm1.weight"], st[f"model.visual.blocks.{b}.norm1.bias"])
        write_gemm(st[f"model.visual.blocks.{b}.attn.qkv.weight"], st[f"model.visual.blocks.{b}.attn.qkv.bias"])
        write_gemm(st[f"model.visual.blocks.{b}.attn.proj.weight"], st[f"model.visual.blocks.{b}.attn.proj.bias"])
        write_ln(st[f"model.visual.blocks.{b}.norm2.weight"], st[f"model.visual.blocks.{b}.norm2.bias"])
        write_gemm(st[f"model.visual.blocks.{b}.mlp.linear_fc1.weight"], st[f"model.visual.blocks.{b}.mlp.linear_fc1.bias"])
        write_gemm(st[f"model.visual.blocks.{b}.mlp.linear_fc2.weight"], st[f"model.visual.blocks.{b}.mlp.linear_fc2.bias"])

    write_ln(st["model.visual.merger.norm.weight"], st["model.visual.merger.norm.bias"])
    write_gemm(st["model.visual.merger.linear_fc1.weight"], st["model.visual.merger.linear_fc1.bias"])
    write_gemm(st["model.visual.merger.linear_fc2.weight"], st["model.visual.merger.linear_fc2.bias"])
    return b"".join(bin_parts)


def export_qwen35_vision(model_dir, output_dir, prefix="", do_bf16=True, tool_path=None):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    vc = config.get("vision_config")
    if not vc:
        print("[vision] No vision_config found, skipping vision export.")
        return None

    hidden_size = vc["hidden_size"]
    depth = vc["depth"]
    intermediate_size = vc["intermediate_size"]
    num_heads = vc["num_heads"]
    head_dim = hidden_size // num_heads
    out_hidden_size = vc["out_hidden_size"]
    patch_size = vc.get("patch_size", 16)
    spatial_merge_size = vc.get("spatial_merge_size", 2)
    temporal_patch_size = vc.get("temporal_patch_size", 2)
    num_pos = vc.get("num_position_embeddings", 2304)
    num_grid = int(num_pos ** 0.5)

    print(f"[vision] Exporting vision tower: hidden={hidden_size}, depth={depth}, num_heads={num_heads}, head_dim={head_dim}, out_hidden={out_hidden_size}")
    st = load_visual_state(model_dir)

    os.makedirs(output_dir, exist_ok=True)
    patch_param_file = f"{prefix}vision_embed_patch.ncnn.param"
    patch_bin_file = f"{prefix}vision_embed_patch.ncnn.bin"
    pos_param_file = f"{prefix}vision_embed_pos.ncnn.param"
    pos_bin_file = f"{prefix}vision_embed_pos.ncnn.bin"
    enc_param_file = f"{prefix}vision_encoder.ncnn.param"
    enc_bin_file = f"{prefix}vision_encoder.ncnn.bin"

    patch_param_path = os.path.join(output_dir, patch_param_file)
    patch_bin_path = os.path.join(output_dir, patch_bin_file)
    pos_param_path = os.path.join(output_dir, pos_param_file)
    pos_bin_path = os.path.join(output_dir, pos_bin_file)
    enc_param_path = os.path.join(output_dir, enc_param_file)
    enc_bin_path = os.path.join(output_dir, enc_bin_file)

    # 1. Vision Patch Embed
    w_size = hidden_size * 3 * temporal_patch_size * patch_size * patch_size
    patch_param = f"""7767517
2 2
Input                    in0                      0 1 in0
Convolution3D            conv3d_0                 1 1 in0 out0 0={hidden_size} 1={patch_size} 21={temporal_patch_size} 3={patch_size} 23={temporal_patch_size} 5=1 6={w_size}
"""
    with open(patch_param_path, "w", encoding="utf-8") as f:
        f.write(patch_param)
    with open(patch_bin_path, "wb") as f:
        f.write(b"\x00\x00\x00\x00")
        f.write(st["model.visual.patch_embed.proj.weight"].contiguous().numpy().tobytes())
        f.write(st["model.visual.patch_embed.proj.bias"].contiguous().numpy().tobytes())

    # 2. Vision Pos Embed
    pos_param = f"""7767517
5 5
Input                    in0                      0 1 in0
MemoryData               pnnx_fold_input.1        0 1 1 0={num_grid} 1={num_grid} 2={hidden_size}
Interp                   F.upsample_6             2 1 1 in0 2 0=2 5=1 6=1 9="1w,1h"
Reshape                  reshape_0                1 1 2 3 0=-1 1={hidden_size}
Permute                  transpose_1              1 1 3 out0 0=1
"""
    with open(pos_param_path, "w", encoding="utf-8") as f:
        f.write(pos_param)
    pos_weight = st["model.visual.pos_embed.weight"].view(num_grid, num_grid, hidden_size).permute(2, 0, 1).contiguous()
    with open(pos_bin_path, "wb") as f:
        f.write(pos_weight.numpy().tobytes())

    # 3. Vision Encoder
    enc_param = build_vision_encoder_param(hidden_size, depth, intermediate_size, num_heads, head_dim, out_hidden_size)
    with open(enc_param_path, "w", encoding="utf-8") as f:
        f.write(enc_param)
    enc_bin = build_vision_encoder_bin(st, depth)
    with open(enc_bin_path, "wb") as f:
        f.write(enc_bin)

    # 4. BF16 convert if requested
    if do_bf16:
        if not tool_path:
            import bf16_convert
            tool_path = bf16_convert.find_ncnnoptimize()
        if tool_path and os.path.isfile(tool_path):
            print(f"[vision] Converting vision weights to BF16 using {tool_path}")
            pairs = [
                (patch_param_path, patch_bin_path),
                (enc_param_path, enc_bin_path),
            ]
            import bf16_convert
            bf16_convert.convert_pairs(pairs, tool_path, flag=bf16_convert.BF16)

    vision_setting = {
        "type": "qwen3.5_vl",
        "vision_embed_patch_param": patch_param_file,
        "vision_embed_patch_bin": patch_bin_file,
        "vision_embed_pos_param": pos_param_file,
        "vision_embed_pos_bin": pos_bin_file,
        "vision_encoder_param": enc_param_file,
        "vision_encoder_bin": enc_bin_file,
        "patch_size": patch_size,
        "patch_dim": hidden_size,
        "head_dim": head_dim,
        "rope_section": [head_dim // 4, head_dim // 4],
        "max_num_patches": 49152,
        "spatial_merge_size": spatial_merge_size,
    }
    return vision_setting
