"""True W4A8 helpers: FP8 activation quantizer + INT4 weight dequant.

This module supports the true W4A8 path (INT4 weight storage + FP8 activation +
FP8 QMMA at 296 TF on SM120). It is loaded by the W4A8 dispatch path in
``python/sglang/srt/layers/quantization/gptq.py`` when
``SOAR_W4A8_REAL_FP8_GEMM=1`` is set.

Design:
- Activation quantization happens online (per forward pass) with per-token
  (per-row) scaling. Input BF16 (M, K) → FP8 e4m3 (M, K) + float32 scales
  (M, 1).
- Weight dequantization (INT4→FP8) happens per forward pass initially
  (Python/PyTorch, temp tensor). This is the validation scaffolding — it
  WILL be replaced by a fused CUDA kernel that unpacks INT4→FP8 inside
  the GEMM mainloop, eliminating the temp-FP8 HBM round-trip.
- Both activation and weight scales follow the 128×128 blockwise convention
  expected by the SM120 cutlass FP8 blockwise GEMM.

Public functions
----------------
- :func:`quantize_activation_fp8_per_token` — BF16 activations → FP8 e4m3
  with per-token scales.
- :func:`dequantize_weight_int4_to_fp8` — GPTQ INT4 packed weights → FP8
  e4m3 with 128×128 blockwise scales.
"""
from __future__ import annotations

from typing import Tuple

import torch

# FP8 e4m3 max absolute value (saturate at ±448).
FP8_E4M3_MAX = 448.0
# Block size for blockwise quantization (matches SM120 cutlass tile).
FP8_BLOCK_SIZE = 128


def quantize_activation_fp8_per_token(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 activations to FP8 e4m3 with per-token scaling.

    Args:
        x: Input activation tensor in ``(M, K)`` layout, dtype BF16 or FP16.

    Returns:
        Tuple of ``(x_fp8, scales)`` where:
        - ``x_fp8``: ``(M, K)`` tensor of ``torch.float8_e4m3fn``.
        - ``scales``: ``(M, 1)`` tensor of ``torch.float32`` (per-row max / 448.0).
    """
    # Per-row max absolute value.
    amax = torch.amax(torch.abs(x), dim=1, keepdim=True).float()  # (M, 1)
    # Clamp small values to avoid division by zero.
    amax = torch.clamp(amax, min=1e-12)
    # Scale = amax / FP8_E4M3_MAX so that scaled values are in [-448, 448].
    scales = amax / FP8_E4M3_MAX  # (M, 1)
    # Quantize: x_fp8 = round(x / scales).to(fp8_e4m3).
    # We do the division in float32 for precision, then cast.
    x_scaled = (x.float() / scales.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scales


def dequantize_weight_int4_to_fp8(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize GPTQ INT4 packed weights to FP8 e4m3 with blockwise scales.

    This is the **temporary validation** path. It produces a full FP8 weight
    tensor that is then fed to the existing SM120 FP8 blockwise GEMM kernel.
    The FP8 weight tensor is materialized in HBM → there is a ~5× weight
    traffic penalty vs the Marlin baseline at decode. This path exists to
    validate activation quantization accuracy before the fused kernel is ready.

    The fused kernel (Day 2-3) will eliminate the temp FP8 weight HBM traffic
    by unpacking INT4→FP8 inside the GEMM mainloop.

    Args:
        qweight: ``(K // 8, N)`` int32 packed INT4 (GPTQ format).
        qzeros: ``(K // group_size // 8, N)`` int32 packed zero-points.
        scales: ``(K // group_size, N)`` BF16/FP16 per-group scales.
        group_size: GPTQ group size (default 128).

    Returns:
        Tuple of ``(weight_fp8, weight_fp8_scales)`` where:
        - ``weight_fp8``: ``(N, K)`` ``torch.float8_e4m3fn``, column-major
          contiguous.
        - ``weight_fp8_scales``: ``(N // 128, K // 128)`` float32 blockwise
          scales for the SM120 FP8 blockwise GEMM.
    """
    # Step 1: Unpack INT4 → BF16 in (K, N) layout.
    w_bf16_kn = _unpack_gptq_int4(qweight, qzeros, scales, group_size)

    # Step 2: Transpose to (N, K) for the GEMM kernel (column-major B).
    w_bf16_nk = w_bf16_kn.t().contiguous()

    # Step 3: Blockwise quantize BF16 → FP8 e4m3 with 128×128 blocks.
    N, K = w_bf16_nk.shape
    assert N % FP8_BLOCK_SIZE == 0, f"N={N} not divisible by {FP8_BLOCK_SIZE}"
    assert K % FP8_BLOCK_SIZE == 0, f"K={K} not divisible by {FP8_BLOCK_SIZE}"

    # Reshape into blocks of (N/128, 128, K/128, 128).
    w_blocks = w_bf16_nk.view(
        N // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
        K // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE,
    )
    # Per-block max absolute value → scales.
    w_amax = torch.amax(torch.abs(w_blocks), dim=(1, 3), keepdim=False)  # (n_blk, k_blk)
    w_scales = w_amax.float() / FP8_E4M3_MAX  # (n_blk, k_blk)

    # Quantize each block.
    w_scaled = w_blocks.float() / w_scales[:, None, :, None].float()
    w_scaled = w_scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    # Reshape back to (N, K).
    w_fp8 = w_fp8.view(N, K).contiguous()

    return w_fp8, w_scales


def _unpack_gptq_int4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Vectorised unpack of GPTQ 4-bit ``desc_act=False`` format → BF16 (K, N).

    Saved tensor shapes (per a single ``Linear`` with ``in_features=K`` and
    ``out_features=N``):

    - ``qweight`` -- ``(K // 8, N)`` ``int32``; each ``int32`` packs 8 INT4
      values along the input dim, in ascending bit positions.
    - ``qzeros`` -- ``(K // group_size // 8, N)`` ``int32``; packed zero-points.
    - ``scales`` -- ``(K // group_size, N)`` BF16/FP16.

    Returns:
        ``(K, N)`` dense BF16 weight tensor.
    """
    # Detect Marlin-packed format: after GPTQ→Marlin repack, qweight may have
    # shape [K//16, N*2] (16 columns per int32) and qzeros may be empty [0]
    # (zero points baked into qweight). Detect and normalize.
    if qzeros.numel() == 0:
        # Marlin format: zeros are already incorporated in qweight.
        # qweight shape: [K//16, N*2] — unpack to [K//8, N] then to [K, N].
        K_div_16, N2 = qweight.shape
        N = N2 // 2
        K = K_div_16 * 16
        groups = K // group_size

        # Unpack Marlin qweight: [K//16, N*2] → [K//8, N] → [K, N]
        # Each int32 packs 8 weights across 2 columns × 4-bit.
        qweight_kn8 = qweight.view(K // 8, N)  # reinterpret
        shifts = torch.arange(0, 8, device=qweight.device) * 4
        qweight_expanded = qweight_kn8.unsqueeze(1)  # (K//8, 1, N)
        qweight_unpacked = (
            (qweight_expanded >> shifts.view(1, 8, 1)) & 0xF
        ).reshape(K, N).to(scales.dtype)

        # Broadcast scales: (groups, N) → (K, N).
        scales_broadcast = scales.repeat_interleave(group_size, dim=0)

        # No zero-point correction needed (already in qweight).
        w_kn = qweight_unpacked.float() * scales_broadcast.float()
        return w_kn.to(scales.dtype)

    # Standard GPTQ format (non-Marlin): qweight [K//8, N], qzeros [groups//8, N].
    K_div_8, N = qweight.shape
    K = K_div_8 * 8
    groups = K // group_size

    # Unpack qweight: (K//8, N) → (K, N).
    shifts = torch.arange(0, 8, device=qweight.device) * 4  # [0, 4, 8, ..., 28]
    # (K//8, 1, N) & (8,) → (K//8, 8, N) → (K, N).
    qweight_expanded = qweight.unsqueeze(1)  # (K//8, 1, N)
    qweight_unpacked = (
        (qweight_expanded >> shifts.view(1, 8, 1)) & 0xF
    ).reshape(K, N)

    # Unpack qzeros: (groups//8, N) → (groups, N) → broadcast to (K, N).
    qzeros_unpacked = (
        (qzeros.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    ).reshape(groups, N)
    # Broadcast zeros to (K, N): each group of group_size rows shares the zero-point.
    qzeros_broadcast = qzeros_unpacked.repeat_interleave(group_size, dim=0)  # (K, N)

    # Broadcast scales: (groups, N) → (K, N).
    scales_broadcast = scales.repeat_interleave(group_size, dim=0)  # (K, N)

    # Dequantize: (INT4 - zero_point) * scale.
    w_kn = (qweight_unpacked.float() - qzeros_broadcast.float()) * scales_broadcast.float()
    return w_kn.to(scales.dtype)


def compute_per_token_quant_error(
    x_bf16: torch.Tensor,
    x_fp8: torch.Tensor,
    scales: torch.Tensor,
) -> float:
    """Compute mean squared error between BF16 input and FP8→BF16 round-trip.

    Args:
        x_bf16: Original BF16 input (M, K).
        x_fp8: FP8 quantized values (M, K).
        scales: Per-token scales (M, 1).

    Returns:
        Mean squared error across all elements.
    """
    # Dequantize: x_bf16_approx = x_fp8.float() * scales.float().
    x_dequant = x_fp8.float() * scales.float()
    mse = torch.mean((x_bf16.float() - x_dequant) ** 2).item()
    return mse
