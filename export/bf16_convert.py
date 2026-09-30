# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Convert fp32 ncnn weights into bf16 storage using ncnn's `ncnnoptimize`.

In this ncnn fork `ncnnoptimize`'s last argument maps to storage_type as::

    flag 0 / anything else -> storage_type = 0  -> fp32
    flag 1 or 65536        -> storage_type = 1  -> fp16   (magic 0x01306B47)
    flag 2                 -> storage_type = 2  -> bf16   (magic 0x01348B83)

(see ncnn/tools/ncnnoptimize.cpp and ncnn/tools/modelwriter.h)

int8 is NOT produced here - it comes from ncnn's own LLM quantization tool
`ncnnllm2int` (see quantize_model.py).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

FP32 = 0
FP16 = 1
BF16 = 2

MAGIC_FP16 = 0x01306B47
MAGIC_BF16 = 0x01348B83
MAGIC_INT8 = 0x000D4B38

STORAGE_TYPES = {0: "fp32", 1: "fp16", 2: "bf16"}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_ncnnoptimize(custom_path=None):
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    env_path = os.environ.get("NCNNOPTIMIZE")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    for base in (os.getcwd(), ROOT):
        for name in ["tools/ncnnoptimize.exe", "tools/ncnnoptimize"]:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return os.path.abspath(path)

    which_path = shutil.which("ncnnoptimize")
    if which_path:
        return os.path.abspath(which_path)

    patterns = [
        os.path.join(ROOT, "build", "**", "ncnnoptimize*"),
        os.path.join(ROOT, "build_cmake", "**", "ncnnoptimize*"),
        os.path.join(ROOT, "ncnn", "build**", "tools", "ncnnoptimize*"),
        os.path.join(ROOT, "..", "ncnn", "build**", "tools", "ncnnoptimize*"),
    ]
    for pattern in patterns:
        for m in glob.glob(pattern, recursive=True):
            if os.path.isfile(m) and not m.endswith((".dir", ".obj")):
                return os.path.abspath(m)
    return None


def run_ncnnoptimize(tool_path, in_param, in_bin, out_param, out_bin, flag=BF16):
    args = [
        tool_path,
        os.path.abspath(in_param),
        os.path.abspath(in_bin),
        os.path.abspath(out_param),
        os.path.abspath(out_bin),
        str(flag),
    ]
    print("Running: " + " ".join(args))
    res = subprocess.run(args)
    if res.returncode != 0:
        raise RuntimeError("ncnnoptimize exited with error code %d" % res.returncode)


def detect_storage_type(bin_path):
    """Best-effort detection of the weight storage type of a ncnn bin file.

    NOTE: this only inspects the **first** weight chunk. ncnn writes some chunks
    without any tag (``ModelWriter::fwrite_weight_data``), and whether a given
    layer uses the tagged writer depends on the layer type, so a file can be
    bf16 through and through while still starting with four raw float bytes.
    Every working decoder asset in this repo does exactly that.  Treat the
    result as a hint and rely on :func:`check_size_ratio` for the real verdict.
    """
    with open(bin_path, "rb") as f:
        magic = f.read(4)
    tag = int.from_bytes(magic, "little")
    # The first 4 bytes of every *tagged* weight chunk carry the magic.
    if tag == MAGIC_BF16:
        return "bf16"
    if tag == MAGIC_FP16:
        return "fp16"
    if tag == MAGIC_INT8:
        return "int8"
    if tag == 0:
        return "fp32"
    return "unknown(0x%08x)" % tag


def check_size_ratio(src_bin_or_size, dst_bin, flag):
    """The reliable way to confirm a conversion actually changed something.

    Chunk tags are per-chunk (and optional), so the file size ratio is the only
    cheap signal that holds for every graph: fp16/bf16 halve the weight bytes,
    fp32 keeps them.  A conversion that silently did nothing leaves the ratio at
    1.0 and is reported as an error.
    """
    a = src_bin_or_size if isinstance(src_bin_or_size, int) else os.path.getsize(src_bin_or_size)
    b = os.path.getsize(dst_bin)
    ratio = b / a if a else 0.0
    want = 1.0 if flag == FP32 else 0.5
    ok = abs(ratio - want) <= 0.02
    return a, b, ratio, want, ok


def replace_with_retry(src, dst, max_retries=30, delay=1.0):
    """Replace ``dst`` with ``src``.

    On Windows ``os.replace`` (== MoveFileEx REPLACE_EXISTING) needs DELETE
    access on ``dst``. When another process holds ``dst`` open with a
    read/memory-mapped handle -- e.g. Windows Search indexing a multi-GB weight
    file, an antivirus scan, or the IDE file watcher -- that DELETE is refused
    (WinError 5 / EBUSY), and the failure is *persistent*, not transient, so a
    plain retry loop never succeeds.

    The file can however still be opened for writing and truncated, so we fall
    back to an in-place overwrite (copyfile + remove tmp). That only needs write
    access and works even while the other process keeps its handle. This keeps
    the canonical filename instead of renaming around the lock.
    """
    last = None
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            last = e
            try:
                shutil.copyfile(src, dst)   # open dst for writing (truncate)
                os.remove(src)              # tmp is ours, never locked
                return
            except (PermissionError, OSError) as e2:
                last = e2
                if attempt < max_retries - 1:
                    print(f"  replace({os.path.basename(dst)}) blocked "
                          f"(attempt {attempt + 1}/{max_retries}: {e}); "
                          f"in-place overwrite also failed: {e2}")
                    time.sleep(delay)
                    delay = min(delay * 1.3, 10.0)
                else:
                    raise
    if last:
        raise last


def convert_pairs(pairs, tool_path, flag=BF16, expected=None):
    """Convert a list of (param, bin) pairs into the requested storage type.

    Conversion always goes through temp files first so an interrupted run never
    leaves a half-written param/bin pair behind.
    """
    for param, rng_bin in pairs:
        # remember the size before converting; that is all the ratio check needs,
        # so there is no reason to duplicate multi-GB files on disk
        src_size = os.path.getsize(rng_bin)

        tmp_p = param + ".tmp.param"
        tmp_b = rng_bin + ".tmp.bin"
        run_ncnnoptimize(tool_path, param, rng_bin, tmp_p, tmp_b, flag)
        a, b, ratio, want, ok = check_size_ratio(src_size, tmp_b, flag)
        if not ok:
            for temporary in (tmp_p, tmp_b):
                if os.path.exists(temporary):
                    os.remove(temporary)
            raise RuntimeError(
                f"{rng_bin}: expected a ~{want:.2f}x size change for "
                f"{STORAGE_TYPES[flag]} but got ratio={ratio:.3f}; "
                "refusing to replace the source files.")
        replace_with_retry(tmp_p, param)
        replace_with_retry(tmp_b, rng_bin)

        a, b, ratio, want, ok = check_size_ratio(src_size, rng_bin, flag)
        hint = detect_storage_type(rng_bin)
        print(f"  {os.path.basename(rng_bin)}: {a/(1024*1024):.2f} MB -> "
              f"{b/(1024*1024):.2f} MB (ratio={ratio:.3f}, want~{want}), "
              f"head tag hint={hint}")
        if not ok:
            raise RuntimeError(
                f"{rng_bin}: expected a ~{want:.2f}x size change for "
                f"{STORAGE_TYPES[flag]} but got ratio={ratio:.3f}. Either "
                f"ncnnoptimize did not convert anything, or the file is "
                f"partially untagged - inspect it by hand.")


def convert_model_dir(model_dir, tool_path, flag=BF16, expected=None):
    """Convert every *.ncnn.param/*.ncnn.bin pair referenced by model.json."""
    model_json_path = os.path.join(model_dir, "model.json")
    with open(model_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seen_bins = set()
    pairs = []
    for key, value in config.get("params", {}).items():
        if not key.endswith("_param"):
            continue
        bin_key = key.replace("_param", "_bin")
        if bin_key not in config["params"]:
            continue
        param_file = value
        bin_file = config["params"][bin_key]
        if bin_file in seen_bins:
            continue  # tied weights (proj_out reusing embed_token.bin)
        seen_bins.add(bin_file)
        pairs.append((os.path.join(model_dir, param_file),
                      os.path.join(model_dir, bin_file)))

    vision_cfg = config.get("setting", {}).get("vision", {})
    if isinstance(vision_cfg, dict) and vision_cfg.get("type") not in (None, "close"):
        for key, value in vision_cfg.items():
            if not (isinstance(key, str) and key.endswith("_param")):
                continue
            bin_key = key.replace("_param", "_bin")
            if bin_key not in vision_cfg:
                continue
            param_file = value
            bin_file = vision_cfg[bin_key]
            if "vision_embed_pos" in param_file:
                continue  # MemoryData stays in fp32
            if bin_file in seen_bins:
                continue
            seen_bins.add(bin_file)
            pairs.append((os.path.join(model_dir, param_file),
                          os.path.join(model_dir, bin_file)))

    if not pairs:
        raise ValueError(f"no param/bin pairs found in {model_dir}")

    print(f"=== Converting {model_dir} to {STORAGE_TYPES[flag]} ===")
    convert_pairs(pairs, tool_path, flag=flag, expected=expected)

    if "storage" in config:
        del config["storage"]
        with open(model_json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Cleaned storage field in {model_json_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert ncnn fp32 weights with ncnnoptimize.")
    ap.add_argument("--model", "-m", help="model directory containing model.json")
    ap.add_argument("--param", help="single param file")
    ap.add_argument("--bin", help="single bin file")
    ap.add_argument("--out-param", default="")
    ap.add_argument("--out-bin", default="")
    ap.add_argument("--flag", type=int, default=BF16, choices=[FP32, FP16, BF16],
                    help="0=fp32, 1=fp16, 2=bf16 (default 2)")
    ap.add_argument("--detect", action="store_true", help="only detect storage type of --bin")
    ap.add_argument("--ncnnoptimize", default="", help="path to ncnnoptimize executable")
    args = ap.parse_args()

    if args.detect:
        if not args.bin:
            ap.error("--detect requires --bin")
        print(detect_storage_type(args.bin))
        return

    tool = find_ncnnoptimize(args.ncnnoptimize)
    if not tool:
        print("Error: ncnnoptimize not found!", file=sys.stderr)
        sys.exit(1)
    print(f"Using ncnnoptimize: {tool}")

    if args.model:
        convert_model_dir(args.model, tool, flag=args.flag)
    elif args.param and args.bin:
        op = args.out_param or args.param
        ob = args.out_bin or args.bin
        source_size = os.path.getsize(args.bin)
        temp_dir = tempfile.mkdtemp(prefix="ncnn_bf16_")
        temp_param = os.path.join(temp_dir, os.path.basename(op) + ".param")
        temp_bin = os.path.join(temp_dir, os.path.basename(ob) + ".bin")
        try:
            run_ncnnoptimize(tool, args.param, args.bin, temp_param, temp_bin, args.flag)
            a, b, ratio, want, ok = check_size_ratio(source_size, temp_bin, args.flag)
            if not ok:
                raise RuntimeError(
                    f"expected a ~{want:.2f}x size change, got ratio={ratio:.3f}; "
                    "refusing to replace the output files")
            replace_with_retry(temp_param, op)
            replace_with_retry(temp_bin, ob)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"storage={detect_storage_type(ob)} -> {ob}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
