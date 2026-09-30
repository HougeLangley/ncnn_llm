"""Numerically verify an ncnn param/bin pair against a reference dump.

Why this exists: ncnn weight files do NOT carry a magic tag in front of every
weight chunk (``ModelWriter::fwrite_weight_data`` writes them untagged), so
sniffing the first 4 bytes of a .bin to decide whether it holds fp32/bf16/int8
is unreliable - every working decoder asset in this repo starts with four raw
float bytes.  Running the graph and comparing numbers is the only real check.

Metrics reported (all against the same reference tensor):

  max|d|     absolute worst-case deviation
  rel_l2     ||got - ref||_2 / ||ref||_2   <- the primary criterion
  cos        cosine similarity
  meanrel    mean elementwise relative error (only useful for bounded values)

Usage::

    python export/check_graph.py <param> <bin> <ref.npz> [--rel-l2 0.02]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llama_export import ncnn_infer  # noqa: E402


def load_ref(ref_path):
    with np.load(ref_path) as z:
        inputs = {name: z[name] for name in z.files if name.startswith("in")}
        if "target" not in z.files or not inputs:
            raise RuntimeError(f"reference archive has no target/in* tensors: {ref_path}")
        return inputs, z["target"]


def compare(param, bin_, ref_path, out_name="out0"):
    inputs, target = load_ref(ref_path)
    got = np.asarray(ncnn_infer(param, bin_, inputs, out_name=out_name))
    if got.shape != target.shape:
        raise RuntimeError(
            f"shape mismatch for {param}: got {got.shape}, expected {target.shape}")

    g = got.astype(np.float64)
    t = target.astype(np.float64)
    diff = np.abs(g - t)
    rel_l2 = float(np.linalg.norm(g - t) / (np.linalg.norm(t) + 1e-12))
    cos = float(np.dot(g.ravel(), t.ravel()) /
                (np.linalg.norm(g.ravel()) * np.linalg.norm(t.ravel()) + 1e-12))
    denom = np.maximum(np.abs(t), 1e-6)
    return {
        "max_abs": float(diff.max()),
        "rel_l2": rel_l2,
        "cos": cos,
        "meanrel": float((diff / denom).mean()),
        "maxrel": float((diff / denom).max()),
        "shape": got.shape,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("param")
    ap.add_argument("bin")
    ap.add_argument("ref")
    ap.add_argument("--out-name", default="out0")
    ap.add_argument("--rel-l2", type=float, default=0.02,
                    help="pass threshold on relative L2 error "
                         "(1e-3 suits fp32, 2e-2 suits bf16)")
    ap.add_argument("--cos", type=float, default=0.999)
    args = ap.parse_args()

    m = compare(args.param, args.bin, args.ref, args.out_name)
    ok = m["rel_l2"] < args.rel_l2 and m["cos"] > args.cos
    print(f"  {os.path.basename(args.param)}: shape={m['shape']} "
          f"max|d|={m['max_abs']:.3e} rel_l2={m['rel_l2']:.3e} "
          f"cos={m['cos']:.6f} meanrel={m['meanrel']:.3e} "
          f"-> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
