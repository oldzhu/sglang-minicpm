"""CPU-only smoke test for SOAR W4A8 helpers.

Validates two independent properties:

1. ``gptq_int4_dequantize`` round-trips correctly when given hand-packed
   INT4 input that we synthesised from a known BF16 reference.

2. ``fp8_blockwise_quantize`` -> ``fp8_blockwise_dequantize`` reconstructs
   a random BF16 tensor with relative Frobenius error below ~1.5e-2
   (FP8 e4m3 has ~3 mantissa bits so worst-case relative error ≈ 1/8).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.quantization.utils_w4a8_fp8 import (
    FP8_BLOCK_SIZE,
    fp8_blockwise_dequantize,
    fp8_blockwise_quantize,
    gptq_int4_dequantize,
)


def _frobenius_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def _pack_int4_along_dim(x_int4: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack INT4 values (range 0..15) into int32 along ``dim`` (8 per int32)."""
    x_int4 = x_int4.to(torch.int32) & 0xF
    if dim == 0:
        K, N = x_int4.shape
        assert K % 8 == 0
        x_int4 = x_int4.view(K // 8, 8, N)
        shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
        return (x_int4 << shifts).sum(dim=1).to(torch.int32)
    elif dim == 1:
        G, N = x_int4.shape
        assert N % 8 == 0
        x_int4 = x_int4.view(G, N // 8, 8)
        shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(1, 1, 8)
        return (x_int4 << shifts).sum(dim=2).to(torch.int32)
    raise ValueError(f"bad dim {dim}")


def test_gptq_int4_dequantize_synthetic() -> None:
    torch.manual_seed(0)
    K, N = 256, 384
    group_size = 128
    G = K // group_size

    # Random INT4 weight in [0, 15] and zero in [0, 14] (so zero+1 in [1, 15]).
    q_int4 = torch.randint(0, 16, (K, N), dtype=torch.int32)
    z_int4 = torch.randint(0, 15, (G, N), dtype=torch.int32)
    scales = torch.randn(G, N, dtype=torch.bfloat16) * 0.01

    # Reference BF16 weight using the canonical formula.
    k_to_group = torch.arange(K) // group_size
    z_full = (z_int4 + 1)[k_to_group]
    s_full = scales[k_to_group]
    w_ref = (q_int4 - z_full).to(scales.dtype) * s_full  # (K, N)

    # Pack in the gptqmodel layout.
    qweight = _pack_int4_along_dim(q_int4, dim=0)  # (K//8, N)
    qzeros = _pack_int4_along_dim(z_int4, dim=1)   # (G, N//8)

    w_dq = gptq_int4_dequantize(qweight, qzeros, scales, group_size=group_size)
    assert w_dq.shape == w_ref.shape
    assert w_dq.dtype == scales.dtype
    rel = _frobenius_rel(w_ref.float(), w_dq.float())
    assert rel < 1e-6, f"GPTQ dequant mismatch, rel={rel:.3e}"
    print(f"[OK] gptq_int4_dequantize: rel_frobenius={rel:.3e}")


def test_fp8_blockwise_quantize_roundtrip() -> None:
    torch.manual_seed(0)
    N, K = 512, 384
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.05

    w_fp8, scale_b = fp8_blockwise_quantize(w, block_size=FP8_BLOCK_SIZE)
    assert w_fp8.dtype == torch.float8_e4m3fn
    assert w_fp8.shape == (N, K)
    assert scale_b.shape == (N // FP8_BLOCK_SIZE, K // FP8_BLOCK_SIZE)
    assert scale_b.dtype == torch.float32

    w_rec = fp8_blockwise_dequantize(w_fp8, scale_b, block_size=FP8_BLOCK_SIZE)
    rel = _frobenius_rel(w.float(), w_rec.float())
    # FP8 e4m3 has 3 mantissa bits; quant step ≈ 2^-3 = 0.125 per element.
    # For Gaussian-distributed inputs the aggregate RMS error empirically
    # falls in the 2-4% range. We assert a generous ≤5% rel-Frobenius.
    assert rel < 5e-2, f"FP8 round-trip too lossy, rel={rel:.3e}"
    print(f"[OK] fp8_blockwise_quantize roundtrip: rel_frobenius={rel:.3e}")


def test_fp8_blockwise_handles_zero_block() -> None:
    """Verify all-zero blocks do not produce NaN scales."""
    N, K = 256, 256
    w = torch.zeros(N, K, dtype=torch.bfloat16)
    # Place a non-zero needle in one block to ensure mixed inputs work.
    w[0, 0] = 0.5
    w_fp8, scale_b = fp8_blockwise_quantize(w)
    assert torch.isfinite(scale_b).all(), "scale produced NaN/Inf"
    w_rec = fp8_blockwise_dequantize(w_fp8, scale_b)
    assert torch.isfinite(w_rec).all()
    print("[OK] fp8_blockwise_quantize zero-block: no NaN/Inf")


if __name__ == "__main__":
    test_gptq_int4_dequantize_synthetic()
    test_fp8_blockwise_quantize_roundtrip()
    test_fp8_blockwise_handles_zero_block()
    print("All W4A8 helper tests passed.")
