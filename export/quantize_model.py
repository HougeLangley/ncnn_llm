# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a BSD-style license.

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def find_ncnnllm2int(custom_path=None):
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    env_path = os.environ.get("NCNNLLM2INT")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    # 1. Check tools/ directory
    for name in ["tools/ncnnllm2int.exe", "tools/ncnnllm2int"]:
        if os.path.isfile(name):
            return os.path.abspath(name)

    # 2. Check system PATH
    which_path = shutil.which("ncnnllm2int")
    if which_path:
        return os.path.abspath(which_path)

    # 3. Search in xmake cache and typical build paths
    user_home = os.path.expanduser("~")
    patterns = [
        os.path.join(user_home, "AppData", "Local", ".xmake", "cache", "packages", "**", "ncnnllm2int.exe"),
        os.path.join(user_home, ".xmake", "cache", "packages", "**", "ncnnllm2int"),
        os.path.join("..", "ncnn", "build**", "tools", "quantize", "ncnnllm2int*"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            if os.path.isfile(m) and not m.endswith(".dir") and not m.endswith(".obj"):
                return os.path.abspath(m)

    return None


def run_ncnnllm2int(tool_path, in_param, in_bin, out_param, out_bin, bits=8, block=64, method="minmax"):
    args = [
        tool_path,
        os.path.abspath(in_param),
        os.path.abspath(in_bin),
        os.path.abspath(out_param),
        os.path.abspath(out_bin),
        f"bits={bits}",
        f"block={block}",
        f"method={method}",
    ]
    print("Running: " + " ".join(args))
    res = subprocess.run(args)
    if res.returncode != 0:
        raise RuntimeError(f"ncnnllm2int exited with error code {res.returncode}")


def get_quantizable_targets(config):
    """
    Returns a list of (role_name, param_key, bin_key) to quantize.
    Explicitly keeps LM Head (proj_out, lm_head), token embeddings, and
    preprocessing untouched.
    """
    params = config.get("params", {})
    targets = []

    # 1. LLM / VLM decoder
    if "decoder_param" in params and "decoder_bin" in params:
        targets.append(("decoder", "decoder_param", "decoder_bin"))
    # 2. OCR / ASR text decoder
    elif "text_decoder_param" in params and "text_decoder_bin" in params:
        targets.append(("text_decoder", "text_decoder_param", "text_decoder_bin"))
    # 3. Embedding encoder
    elif "encoder_param" in params and "encoder_bin" in params:
        targets.append(("encoder", "encoder_param", "encoder_bin"))

    # 4. CLIP text & vision encoders
    model_type = config.get("model_type", "")
    if "text_encoder_param" in params and "text_encoder_bin" in params:
        targets.append(("text_encoder", "text_encoder_param", "text_encoder_bin"))
    if "vision_encoder_param" in params and "vision_encoder_bin" in params and model_type == "clip":
        targets.append(("vision_encoder", "vision_encoder_param", "vision_encoder_bin"))

    return targets


def quantize_model_dir(model_dir, output_dir, tool_path, bits=8, block=64, method="minmax"):
    model_json_path = os.path.join(model_dir, "model.json")
    if not os.path.isfile(model_json_path):
        raise FileNotFoundError(f"model.json not found in {model_dir}")

    with open(model_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    targets = get_quantizable_targets(config)
    if not targets:
        raise ValueError(f"No quantizable targets found in {model_dir}")

    os.makedirs(output_dir, exist_ok=True)
    params = config.get("params", {})

    quantized_bin_files = set()
    quantized_param_files = set()
    total_orig_bytes = 0
    total_quant_bytes = 0
    quantized_parts = []

    for role_name, param_key, bin_key in targets:
        param_file = params[param_key]
        bin_file = params[bin_key]

        in_param = os.path.join(model_dir, param_file)
        in_bin = os.path.join(model_dir, bin_file)

        if not os.path.isfile(in_param):
            raise FileNotFoundError(f"{role_name} param file not found: {in_param}")
        if not os.path.isfile(in_bin):
            raise FileNotFoundError(f"{role_name} bin file not found: {in_bin}")

        out_param = os.path.join(output_dir, param_file)
        out_bin = os.path.join(output_dir, bin_file)

        print(f"\n=== Quantizing {role_name} ({bits}-bit, block={block}, method={method}) ===")
        print(f"Input:  {in_param} / {in_bin}")
        print(f"Output: {out_param} / {out_bin}")

        run_ncnnllm2int(tool_path, in_param, in_bin, out_param, out_bin, bits, block, method)

        quantized_param_files.add(param_file)
        quantized_bin_files.add(bin_file)
        quantized_parts.append(role_name)

        orig_sz = os.path.getsize(in_bin)
        quant_sz = os.path.getsize(out_bin)
        total_orig_bytes += orig_sz
        total_quant_bytes += quant_sz
        print(f"Compressed {role_name} from {orig_sz / (1024*1024):.2f} MB to {quant_sz / (1024*1024):.2f} MB")

    # Copy all other assets (LM head, token embeddings, tokenizers, config, images, etc.)
    print("\n=== Copying unquantized assets (LM Head, Embedding, Tokenizer, Config) ===")
    for item in os.listdir(model_dir):
        if item == "model.json":
            continue
        if item in quantized_bin_files or item in quantized_param_files:
            continue
        src = os.path.join(model_dir, item)
        dst = os.path.join(output_dir, item)
        if os.path.isfile(src) and not os.path.exists(dst):
            print(f"Copying {item}...")
            shutil.copy2(src, dst)
        elif os.path.isdir(src) and not os.path.exists(dst):
            print(f"Copying directory {item}...")
            shutil.copytree(src, dst)

    # Save updated model.json with quantization metadata
    config["quantization"] = {
        "type": "weight_block_quant",
        "bits": bits,
        "block_size": block,
        "method": method,
        "quantized_parts": quantized_parts,
    }

    out_json = os.path.join(output_dir, "model.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n=== Quantization Summary ===")
    print(f"Model output directory: {output_dir}")
    print(f"Quantized components: {', '.join(quantized_parts)}")
    print(f"Total original quantized weight size: {total_orig_bytes / (1024*1024):.2f} MB")
    print(f"Total compressed weight size: {total_quant_bytes / (1024*1024):.2f} MB")
    if total_quant_bytes > 0:
        ratio = total_orig_bytes / total_quant_bytes
        saved = (total_orig_bytes - total_quant_bytes) / (1024*1024)
        print(f"Compression ratio: {ratio:.2f}x (saved {saved:.2f} MB)")
    print("Note: LM Head (proj_out/lm_head) and embeddings kept unquantized for precision preservation.")


def quantize_all_models(assets_dir, tool_path, bits=8, block=64, method="minmax"):
    assets_dir = os.path.abspath(assets_dir)
    print(f"Scanning for quantizable models in {assets_dir}...")

    candidates = []
    for name in sorted(os.listdir(assets_dir)):
        p = os.path.join(assets_dir, name)
        if not os.path.isdir(p):
            continue
        if name.endswith("_int4") or name.endswith("_int6") or name.endswith("_int8"):
            continue
        mjson = os.path.join(p, "model.json")
        if not os.path.isfile(mjson):
            continue
        try:
            with open(mjson, "r", encoding="utf-8") as f:
                c = json.load(f)
            targets = get_quantizable_targets(c)
            if targets:
                candidates.append((name, p, targets))
        except Exception as e:
            print(f"Skipping {name}: {e}")

    print(f"Found {len(candidates)} quantizable model(s):")
    for name, _, targets in candidates:
        target_str = ", ".join(f"{t[0]} ({t[1]})" for t in targets)
        print(f"  - {name}: {target_str}")

    for idx, (name, src_dir, _) in enumerate(candidates, 1):
        out_name = f"{name}_int{bits}"
        out_dir = os.path.join(assets_dir, out_name)
        print(f"\n=================================================================")
        print(f"[{idx}/{len(candidates)}] Processing {name} -> {out_name}")
        print(f"=================================================================")
        quantize_model_dir(src_dir, out_dir, tool_path, bits, block, method)

    print("\nAll models quantized successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize LLM model for ncnn_llm using ncnn built-in ncnnllm2int tool."
    )
    parser.add_argument("--model", "-m", type=str, default="", help="Input model directory (containing model.json)")
    parser.add_argument("--output", "-o", type=str, default="", help="Output model directory (defaults to <model>_int8)")
    parser.add_argument("--all", action="store_true", help="Quantize all models found in assets directory")
    parser.add_argument("--assets-dir", type=str, default="assets", help="Assets directory path (default: assets)")
    parser.add_argument("--param", type=str, default="", help="Single param file path")
    parser.add_argument("--bin", type=str, default="", help="Single bin file path")
    parser.add_argument("--out-param", type=str, default="", help="Output param file path")
    parser.add_argument("--out-bin", type=str, default="", help="Output bin file path")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8], help="Weight quantization bits (default: 8)")
    parser.add_argument("--block", type=int, default=64, choices=[32, 64, 128], help="Quantization block size (default: 64)")
    parser.add_argument("--method", type=str, default="minmax", choices=["minmax", "mseclip"], help="Quantization method (default: minmax)")
    parser.add_argument("--ncnnllm2int", type=str, default="", help="Path to ncnnllm2int executable")

    args = parser.parse_args()

    tool_path = find_ncnnllm2int(args.ncnnllm2int)
    if not tool_path:
        print("Error: ncnnllm2int tool not found!", file=sys.stderr)
        print("Please build ncnnllm2int or specify --ncnnllm2int /path/to/ncnnllm2int", file=sys.stderr)
        sys.exit(1)
    print(f"Using ncnnllm2int tool: {tool_path}")

    if args.all:
        quantize_all_models(args.assets_dir, tool_path, args.bits, args.block, args.method)
    elif args.param and args.bin:
        out_param = args.out_param or args.param.replace(".param", f"_int{args.bits}.param")
        out_bin = args.out_bin or args.bin.replace(".bin", f"_int{args.bits}.bin")
        run_ncnnllm2int(tool_path, args.param, args.bin, out_param, out_bin, args.bits, args.block, args.method)
        print(f"Quantized files saved to:\n  {out_param}\n  {out_bin}")
    elif args.model:
        model_dir = args.model.rstrip("/\\")
        output_dir = args.output or (model_dir + f"_int{args.bits}")
        quantize_model_dir(model_dir, output_dir, tool_path, args.bits, args.block, args.method)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
