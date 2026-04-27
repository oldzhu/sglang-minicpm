"""SOAR W4A8 helpers: GPTQ INT4 dequantization + FP8 blockwise quantization.

This module is loaded by the W4A8 dispatch path in
``python/sglang/srt/layers/quantization/gptq.py`` when
``SOAR_W4A8_FP8_GEMM=1`` is set.

Design choices:
- All math is pure PyTorch (CPU + CUDA compatible) so the routines can be
  unit-tested locally without GPU.
- The GPTQ unpacking assumes the standard ``gptqmodel`` save layout
  (``desc_act=False``, 4-bit, group_size=128). Validation against the
  ``gptqmodel.GPTQModel.from_quantized`` reload happens during fcloud
  integration testing.
- FP8 e4m3 ``saturate`` to ±448; we pre-clamp before casting to avoid relying
  on undefined casting behaviour.

Public functions
----------------
- :func:`gptq_int4_dequantize` -- unpack ``qweight`` + ``qzeros`` + ``scales``
  into a dense BF16 weight tensor in ``(in_features, out_features)`` layout
  (matching what GPTQModel saves).
- :func:`fp8_blockwise_quantize` -- given a 2D BF16 weight in ``(N, K)``
  layout (compatible with ``cutlass_w8a8_block_fp8_linear_with_fallback``),
  return ``(weight_fp8, weight_fp8_scale)`` with ``128x128`` blocks.
"""
from __future__ import annotations

from typing import Tuple

import torch

# FP8 e4m3 max absolute value (saturate at ±448).
FP8_E4M3_MAX = 448.0
FP8_BLOCK_SIZE = 128


def gptq_int4_dequantize(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Vectorised dequantisation for ``gptqmodel`` 4-bit ``desc_act=False`` format.

    Saved tensor shapes (per a single ``Linear`` with ``in_features = K`` and
    ``out_features = N``):

    - ``qweight`` -- ``(K // 8, N)`` ``int32``; each ``int32`` packs 8 INT4
      values along the input dim, in ascending bit positions
      (``bits 0-3`` = first row of K).
    - ``qzeros``  -- ``(K // group_size, N // 8)`` ``int32``; each ``int32``
      packs 8 INT4 zero points along the output dim.
    - ``scales``  -- ``(K // group_size, N)`` BF16 / FP16.

    Returns:
        ``w`` -- ``(K, N)`` tensor in ``scales.dtype`` (BF16 typically).
    """
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be torch.int32, got {qweight.dtype}")
    if qzeros.dtype != torch.int32:
        raise TypeError(f"qzeros must be torch.int32, got {qzeros.dtype}")

    K_packed, N = qweight.shape
    K = K_packed * 8
    G = qzeros.shape[0]
    if scales.shape != (G, N):
        raise ValueError(
            f"scales shape {tuple(scales.shape)} mismatches "
            f"(num_groups={G}, out_features={N})"
        )
    if qzeros.shape != (G, N // 8):
        raise ValueError(
            f"qzeros shape {tuple(qzeros.shape)} mismatches "
            f"(num_groups={G}, out_features // 8 = {N // 8})"
        )
    if K != G * group_size:
        raise ValueError(
            f"in_features ({K}) must equal num_groups * group_size "
            f"({G} * {group_size} = {G * group_size})"
        )

    device = qweight.device
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)  # (8,)

    # Unpack qweight along K axis: (K_packed, N) -> (K_packed, 8, N) -> (K, N)
    q_unpacked = (qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF
    q_unpacked = q_unpacked.to(torch.int32).reshape(K, N)

    # Unpack qzeros along N axis: (G, N//8) -> (G, N//8, 8) -> (G, N)
    z_unpacked = (qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF
    z_unpacked = z_unpacked.to(torch.int32).reshape(G, N)
    # gptqmodel / auto_gptq store ``zero - 1`` to keep the same packed value
    # for symmetric and asymmetric modes; restore the actual zero point.
    z_unpacked = z_unpacked + 1

    # Broadcast zero/scale from per-group to per-row.
    k_to_group = (
        torch.arange(K, device=device, dtype=torch.long) // group_size
    )  # (K,)
    zeros_full = z_unpacked[k_to_group]  # (K, N) int32
    scales_full = scales[k_to_group]      # (K, N) bf16/fp16

    w = (q_unpacked - zeros_full).to(scales.dtype) * scales_full
    return w


def fp8_blockwise_quantize(
    w: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise a 2D weight to FP8 e4m3 with 128x128 block scales.

    Args:
        w: ``(N, K)`` float tensor (typically BF16). ``N`` and ``K`` must be
            multiples of ``block_size``. The shape convention matches what
            ``cutlass_w8a8_block_fp8_linear_with_fallback`` expects (it does
            ``weight.T`` internally before calling
            ``fp8_blockwise_scaled_mm``).
        block_size: side length of the square scale blocks. Default 128.

    Returns:
        ``weight_fp8`` -- ``(N, K)`` ``torch.float8_e4m3fn``.
        ``weight_fp8_scale`` -- ``(N // block_size, K // block_size)`` fp32.
    """
    if w.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(w.shape)}")

    N, K = w.shape
    if N % block_size != 0 or K % block_size != 0:
        raise ValueError(
            f"weight shape ({N}, {K}) not divisible by block_size {block_size}"
        )

    # Reshape into blocks: (N/B, B, K/B, B) so we can reduce over axes (1, 3).
    w_f32 = w.float()
    w_blocked = w_f32.view(N // block_size, block_size, K // block_size, block_size)
    block_amax = w_blocked.abs().amax(dim=(1, 3))  # (N/B, K/B)

    # Avoid division-by-zero on all-zero blocks.
    safe_amax = block_amax.clamp(min=1e-12)
    weight_fp8_scale = (safe_amax / FP8_E4M3_MAX).to(torch.float32)  # (N/B, K/B)

    # Broadcast scale back to (N, K) for the divide.
    scale_full = weight_fp8_scale.repeat_interleave(
        block_size, dim=0
    ).repeat_interleave(block_size, dim=1)  # (N, K)

    w_scaled = (w_f32 / scale_full).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    weight_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return weight_fp8, weight_fp8_scale


def fp8_blockwise_dequantize(
    weight_fp8: torch.Tensor,
    weight_fp8_scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Reference dequant used by the round-trip test.

    Returns ``weight_fp8.float() * scale_full`` where ``scale_full`` is
    repeat-broadcast from ``weight_fp8_scale``.
    """
    N, K = weight_fp8.shape
    scale_full = weight_fp8_scale.repeat_interleave(
        block_size, dim=0
    ).repeat_interleave(block_size, dim=1)
    return weight_fp8.float() * scale_full
