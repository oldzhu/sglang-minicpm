#!/usr/bin/env python3
"""Detailed analysis of 'Other' category kernels in profiling traces."""
import gzip
import json
import os
import sys
from collections import defaultdict


def is_known_category(name):
    lower = name.lower()
    known_keywords = [
        "marlin", "gptq", "cutlass", "gemm", "gemv", "matmul",
        "fused_recurrent", "flash", "fmha",
        "rotary", "rope", "silu",
        "memcpy", "memset", "embedding", "gather",
        "chunk_fwd",
    ]
    for kw in known_keywords:
        if kw in lower:
            return True
    return False


def guess_triton_purpose(name):
    """Guess what a triton kernel does from its fused name."""
    lower = name.lower()
    if "mm" in lower and ("sigmoid" in lower or "mul" in lower):
        return "GEMM+activation (torch.compile fused)"
    if "mm" in lower and "permute" in lower:
        return "GEMM+permute (torch.compile fused)"
    if "mm" in lower:
        return "GEMM (torch.compile fused)"
    if "rmsnorm" in lower or ("mean" in lower and "pow" in lower and "rsqrt" in lower):
        return "RMSNorm (fused)"
    if "copy_" in lower and ("mean" in lower or "add" in lower):
        return "RMSNorm+residual (fused)"
    if "sigmoid" in lower:
        return "Sigmoid (output gate)"
    if "index_put" in lower or "index_copy" in lower:
        return "State scatter/gather"
    if "index" in lower:
        return "Indexing"
    if "copy" in lower or "convert" in lower:
        return "Type cast / copy"
    if "sum" in lower:
        return "Reduction (sum)"
    if "mul" in lower and "copy_" in lower:
        return "Residual scale (mul+copy)"
    if "reduce" in lower:
        return "Reduction"
    return "Unknown"


def main():
    profile_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/minicpm_profile"

    for fname in sorted(os.listdir(profile_dir)):
        if not fname.endswith(".trace.json.gz"):
            continue
        fpath = os.path.join(profile_dir, fname)
        with gzip.open(fpath, "rt") as f:
            data = json.load(f)

        events = data if isinstance(data, list) else data.get("traceEvents", [])

        others = defaultdict(float)
        others_count = defaultdict(int)
        total = 0.0

        for ev in events:
            if ev.get("cat") not in ("kernel", "gpu_memcpy"):
                continue
            name = ev.get("name", "")
            dur = ev.get("dur", 0)
            total += dur
            lower = name.lower()

            # Check normalization kernels separately (may contain "norm" but also other keywords)
            if "norm" in lower or "rms" in lower:
                continue
            if "fla" in lower or "gla" in lower:
                continue

            if not is_known_category(name):
                others[name] += dur
                others_count[name] += 1

        other_total = sum(others.values())
        print(f"\n{'='*80}")
        print(f"=== {fname}")
        print(f"    Other = {other_total/1000:.1f}ms / Total = {total/1000:.1f}ms ({other_total/total*100:.1f}%)")

        for name, dur in sorted(others.items(), key=lambda x: -x[1])[:15]:
            pct = dur / total * 100
            cnt = others_count[name]
            guess = guess_triton_purpose(name)
            print(f"  {pct:5.1f}% {dur/1000:7.1f}ms ({cnt:3d}x) [{guess:30s}] {name[:120]}")


if __name__ == "__main__":
    main()
