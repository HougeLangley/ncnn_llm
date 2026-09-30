# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Turn a plain full-sequence ncnn decoder into a KV-cache capable one.

pnnx exports ``F.scaled_dot_product_attention(q, k, v, attn_mask)`` as a 4-input /
1-output ``SDPA`` layer that recomputes attention over the whole sequence every
step. ncnn's SDPA layer natively supports incremental decoding with signature
``6 inputs (q, k, v, mask, cache_k, cache_v) -> 3 outputs (out, out_cache_k,
out_cache_v)`` and params ``5=has_mask``, ``6=scale``, ``7=kvcache_enabled``.

This module rewrites every SDPA op to that signature and adds a single
``Input kv_cache`` layer producing ``cache_k{i}`` / ``cache_v{i}`` for every
layer, exactly the layout ncnn_llm's runtime expects::

    Input kv_cache 0 N cache_k0 cache_v0 cache_k1 cache_v1 ...

It is the generalized version of the original ``hunyuan_ocr_add_kvcache.py``
and also understands the already-converted signature (idempotent).
"""

import argparse
import os


def find_input_layer_indices(lines):
    """Return indices of ``Input`` layer lines inside the param body."""
    return [i for i, l in enumerate(lines) if l.split() and l.split()[0] == "Input"]


def add_kvcache_to_param(path, head_dim=None, scale=None, backup=True):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    lines = raw.split("\n")
    if lines[0].strip() != "7767517":
        raise SystemExit("not an ncnn param file: " + path)

    if backup:
        bak = path + ".nokv"
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as f:
                f.write(raw)

    header = lines[1].split()
    orig_layers, orig_blobs = int(header[0]), int(header[1])
    body = lines[2:]

    def noutputs(line):
        p = line.split()
        return int(p[3]) if len(p) >= 4 else 0

    # 1) rewrite every SDPA layer
    sdpa_count = 0
    already = False
    new_body = []
    for line in body:
        parts = line.split()
        if not parts or parts[0] != "SDPA":
            new_body.append(line)
            continue
        name = parts[1]
        nin, nout = int(parts[2]), int(parts[3])
        blobs = parts[4:4 + nin + nout]
        params = parts[4 + nin + nout:]
        pd = {}
        for p in params:
            if "=" in p:
                k, v = p.split("=", 1)
                pd[k] = v

        if nin in (5, 6) and nout == 3:
            already = True
            sdpa_count += 1
            new_body.append(line)
            continue

        if nin not in (3, 4) or nout != 1:
            raise SystemExit(
                f"unexpected SDPA signature {nin}-in/{nout}-out in {path}; "
                "refusing to guess. (expect 3/1 or 4/1 full-sequence, "
                "or 5/3 or 6/3 kvcache)"
            )

        has_mask = nin == 4
        q, k, v = blobs[0], blobs[1], blobs[2]
        mask = blobs[3] if has_mask else None
        out = blobs[nin]

        # attention_scale = 1 / sqrt(head_dim). Leaving it unset is what makes
        # ncnn fall back to a shape-derived default; stating it explicitly keeps
        # the graph self-describing.
        pd["5"] = "1" if has_mask else "0"
        if scale is not None:
            pd["6"] = scale
        pd["7"] = "1"                       # enable kvcache

        idx = sdpa_count
        nin_new = [q, k, v]
        if has_mask:
            nin_new.append(mask)
        nin_new += [f"cache_k{idx}", f"cache_v{idx}"]
        nout_new = [out, f"out_cache_k{idx}", f"out_cache_v{idx}"]
        pstr = " ".join(f"{key}={pd[key]}" for key in sorted(pd, key=lambda x: int(x)))
        new_body.append(
            "%-24s %-24s %d %d %s %s"
            % ("SDPA", name, len(nin_new), len(nout_new),
               " ".join(nin_new + nout_new), pstr)
        )
        sdpa_count += 1

    if sdpa_count == 0:
        raise SystemExit("no SDPA layer found in " + path)

    # 2) insert / refresh the grouped `Input kv_cache` layer
    cache_blobs = []
    for i in range(sdpa_count):
        cache_blobs += [f"cache_k{i}", f"cache_v{i}"]
    input_line = "%-24s %-24s 0 %d %s" % ("Input", "kv_cache", len(cache_blobs),
                                          " ".join(cache_blobs))

    inputs = [i for i, l in enumerate(new_body)
              if l.split() and l.split()[0] == "Input"]
    if not inputs:
        raise SystemExit("no Input layer found in " + path)

    # idempotent: replace an already present kv_cache Input instead of stacking
    # another one on top (the exporter may be re-run over an existing output)
    existing = [i for i in inputs if len(new_body[i].split()) >= 4
                and new_body[i].split()[1] == "kv_cache"]
    if existing:
        new_body[existing[0]] = input_line
    else:
        new_body.insert(inputs[-1] + 1, input_line)

    # 3) recompute header
    layer_lines = [l for l in new_body if l.strip()]
    new_layers = len(layer_lines)
    new_blobs = sum(noutputs(l) for l in layer_lines)

    out = [lines[0], "%d %d" % (new_layers, new_blobs)] + new_body
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))

    return {
        "sdpa": sdpa_count,
        "already_converted": already,
        "old": (orig_layers, orig_blobs),
        "new": (new_layers, new_blobs),
        "path": path,
    }


def main():
    ap = argparse.ArgumentParser(description="Add KV-cache inputs/outputs to an ncnn decoder param.")
    ap.add_argument("param", nargs="?", default="", help="path to *.ncnn.param")
    ap.add_argument("--scale", type=str, default=None,
                    help="explicit attention scale to write into SDPA param 6 (default: leave unset => 1/sqrt(head_dim))")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if not args.param:
        ap.print_help()
        return
    info = add_kvcache_to_param(args.param, scale=args.scale, backup=not args.no_backup)
    print("[kvcache] SDPA layers: %d (already converted: %s)" % (info["sdpa"], info["already_converted"]))
    print("[kvcache] header %s -> %s" % (info["old"], info["new"]))
    print("[kvcache] wrote %s" % info["path"])


if __name__ == "__main__":
    main()
