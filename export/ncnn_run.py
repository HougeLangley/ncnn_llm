# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Run one ncnn graph and dump the result.

This exists as a *standalone process* on purpose: torch and ncnn each ship
their own copy of the OpenMP runtime.  Loading both into one interpreter on
Windows deadlocks the moment ncnn's threads touch the graph (observed: CPU time
frozen at user=44s/sys=28s, single remaining thread, no progress for 20+ min).
Keeping ncnn in a child process that never imports torch sidesteps it.

Usage::

    python export/ncnn_run.py --param x.ncnn.param --bin x.ncnn.bin \
        --inputs in.npz --outputs out.npz [--out-name out0]
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--inputs", required=True, help="npz holding every input blob")
    ap.add_argument("--outputs", required=True, help="npz to write the result into")
    ap.add_argument("--out-name", default="out0")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import ncnn
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    if args.threads > 0:
        net.opt.num_threads = args.threads

    if net.load_param(args.param) != 0:
        raise SystemExit("failed to load param " + args.param)
    if net.load_model(args.bin) != 0:
        raise SystemExit("failed to load bin " + args.bin)

    z = np.load(args.inputs)
    with net.create_extractor() as ex:
        for name in z.files:
            arr = np.ascontiguousarray(z[name], dtype=np.float32)
            ret = ex.input(name, ncnn.Mat(arr).clone())
            if ret != 0:
                raise SystemExit(f"input {name} failed ({ret})")
        ret, out = ex.extract(args.out_name)
        if ret != 0:
            raise SystemExit(f"extract {args.out_name} failed ({ret})")

    np.savez(args.outputs, **{args.out_name: np.array(out)})
    print(f"[ncnn_run] ok -> {args.out_name} {tuple(np.array(out).shape)}")


if __name__ == "__main__":
    main()
