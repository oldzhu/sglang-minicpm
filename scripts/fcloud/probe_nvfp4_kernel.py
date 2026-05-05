"""
Probe (a)+(b): NVFP4 kernel dispatch + microbench on SM120.

Confirms:
  - is_sm120_supported() == True
  - flashinfer.fp4_quantize / mm_fp4 are importable and run end-to-end
  - measures latency of (i) BF16 reference matmul, (ii) fp4_quantize + mm_fp4 end-to-end,
    (iii) mm_fp4 only with pre-quantized inputs.
  - reports the per-step quantize overhead as a fraction of total FP4 time.
"""

import time
import torch


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per iter


def main():
    print("torch:", torch.__version__)
    print("cuda device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability:", cap)

    try:
        from sglang.srt.utils import is_sm120_supported
        print("is_sm120_supported():", is_sm120_supported())
    except Exception as e:
        print("is_sm120_supported import failed:", e)

    try:
        from flashinfer import fp4_quantize as fi_fp4_quantize
        from flashinfer import mm_fp4 as fi_mm_fp4
        print("flashinfer fp4_quantize / mm_fp4: OK")
    except Exception as e:
        print("flashinfer import failed:", e)
        return

    # Typical projection shape for MiniCPM-SALA-90 hidden=4096, ffn intermediate ~10880
    shapes = [
        ("decode-1   x qkv  ", 1, 4096, 4096),
        ("decode-1   x ffn  ", 1, 4096, 10880),
        ("prefill-512x qkv  ", 512, 4096, 4096),
        ("prefill-512x ffn  ", 512, 4096, 10880),
        ("chunk-4096 x qkv  ", 4096, 4096, 4096),
        ("chunk-4096 x ffn  ", 4096, 4096, 10880),
        ("chunk-65536x qkv  ", 65536, 4096, 4096),
        ("chunk-65536x ffn  ", 65536, 4096, 10880),
    ]

    print()
    print(f"{'shape':<22} {'M':>6} {'K':>6} {'N':>6}   "
          f"{'BF16 ms':>9} {'FP4 e2e':>9} {'mm_fp4':>9} {'quant':>9} {'q% e2e':>7} "
          f"{'speedup':>8}")
    print("-" * 110)

    for label, M, K, N in shapes:
        try:
            x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

            # BF16 baseline
            t_bf16 = cuda_time(lambda: torch.matmul(x_bf16, w_bf16.t()))

            # Static input/weight scales (real loader produces these too)
            input_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            input_scale_inv = 1.0 / input_scale

            # Pre-quantize weights once (this is what the loader does)
            w_fp4, w_scale = fi_fp4_quantize(w_bf16, input_scale_inv)
            alpha = (input_scale * 1.0).to(torch.float32)

            # End-to-end: fp4_quantize per call + mm_fp4
            def fp4_e2e():
                xf, xs = fi_fp4_quantize(x_bf16, input_scale_inv)
                return fi_mm_fp4(xf, w_fp4, xs, w_scale, alpha,
                                 torch.bfloat16, backend="cutlass")

            # Pre-quantized x: mm_fp4 only
            xf_pre, xs_pre = fi_fp4_quantize(x_bf16, input_scale_inv)

            def fp4_mm_only():
                return fi_mm_fp4(xf_pre, w_fp4, xs_pre, w_scale, alpha,
                                 torch.bfloat16, backend="cutlass")

            def fp4_quant_only():
                return fi_fp4_quantize(x_bf16, input_scale_inv)

            t_e2e = cuda_time(fp4_e2e)
            t_mm = cuda_time(fp4_mm_only)
            t_q = cuda_time(fp4_quant_only)
            q_pct = 100.0 * t_q / t_e2e if t_e2e > 0 else 0.0
            speedup = t_bf16 / t_e2e if t_e2e > 0 else 0.0

            print(f"{label:<22} {M:>6d} {K:>6d} {N:>6d}   "
                  f"{t_bf16:>9.4f} {t_e2e:>9.4f} {t_mm:>9.4f} {t_q:>9.4f} {q_pct:>6.1f}% "
                  f"{speedup:>7.2f}x")
        except Exception as e:
            print(f"{label:<22} ERROR: {e}")


if __name__ == "__main__":
    main()
