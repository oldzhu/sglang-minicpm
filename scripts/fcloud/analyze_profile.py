#!/usr/bin/env python3
"""Analyze sglang torch profiler traces and produce kernel breakdown."""
import gzip
import json
import os
import sys
from collections import defaultdict


def categorize_kernel(name):
    lower = name.lower()
    if "marlin" in lower or "gptq" in lower or "cutlass" in lower:
        return "GEMM (Marlin/GPTQ)"
    if "gemm" in lower or "gemv" in lower or "matmul" in lower:
        return "GEMM (other)"
    if "fla" in lower or "gla" in lower or "fused_recurrent" in lower:
        return "FLA/SimpleGLA"
    if "chunk" in lower and ("fwd" in lower or "bwd" in lower or "state" in lower):
        return "FLA/SimpleGLA"
    if "flash" in lower or "fmha" in lower:
        return "FlashAttention"
    if "norm" in lower or "rms" in lower:
        return "RMSNorm"
    if "rotary" in lower or "rope" in lower:
        return "RoPE"
    if "silu" in lower:
        return "Activation (SiLU)"
    if "memcpy" in lower or "memset" in lower:
        return "Memory ops"
    if "embedding" in lower or "gather" in lower:
        return "Embedding/Gather"
    return "Other"


def analyze_trace(fpath):
    with gzip.open(fpath, "rt") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    kernel_times = defaultdict(float)
    kernel_counts = defaultdict(int)
    total_gpu_time = 0

    for ev in events:
        cat = ev.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "cuda_runtime"):
            if cat == "cuda_runtime":
                continue
        if cat not in ("kernel", "gpu_memcpy"):
            continue
        name = ev.get("name", "unknown")
        dur = ev.get("dur", 0)
        kernel_times[name] += dur
        kernel_counts[name] += 1
        total_gpu_time += dur

    return kernel_times, kernel_counts, total_gpu_time


def main():
    profile_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/minicpm_profile"

    for fname in sorted(os.listdir(profile_dir)):
        if not fname.endswith(".trace.json.gz"):
            continue
        fpath = os.path.join(profile_dir, fname)
        kernel_times, kernel_counts, total_gpu_time = analyze_trace(fpath)

        print(f"\n{'='*80}")
        print(f"=== {fname} ===")
        print(f"Total GPU kernel time: {total_gpu_time/1000:.1f} ms")
        print(f"Unique kernels: {len(kernel_times)}")

        print(f"\nTop 25 kernels by total time:")
        sorted_kernels = sorted(kernel_times.items(), key=lambda x: -x[1])
        for name, dur in sorted_kernels[:25]:
            pct = dur / total_gpu_time * 100 if total_gpu_time else 0
            cnt = kernel_counts[name]
            cat = categorize_kernel(name)
            print(f"  {pct:5.1f}% {dur/1000:8.1f}ms ({cnt:4d}x) [{cat:20s}] {name[:120]}")

        # Category breakdown
        categories = defaultdict(float)
        for name, dur in kernel_times.items():
            categories[categorize_kernel(name)] += dur

        print(f"\nCategory breakdown:")
        for cat, dur in sorted(categories.items(), key=lambda x: -x[1]):
            pct = dur / total_gpu_time * 100 if total_gpu_time else 0
            print(f"  {pct:5.1f}% {dur/1000:8.1f}ms  {cat}")


if __name__ == "__main__":
    main()
