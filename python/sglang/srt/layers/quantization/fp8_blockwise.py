"""FP8 blockwise quantization for SM120 UMMA.

This module provides FP8 blockwise quantization using the existing SM120 FP8 kernel
from sgl-kernel (fp8_blockwise_scaled_mm).

Key properties:
- Weights: Pre-quantized offline to (N, K) float8_e4m3fn with (N/128, K/128) scales
- At inference: weight.t() gives col-major (K, N) required by the kernel (zero-copy)
- Activations: Quantized per-row per-128-K-block at inference time
- Kernel: Uses SM120 UMMA tcgen05.mma.ws.sync (already in sgl-kernel)
- No new CUDA code needed

Kernel requirements (fp8_blockwise_scaled_mm):
  mat_a: (M, K) row-major float8_e4m3fn
  mat_b: (K, N) col-major float8_e4m3fn  [stride(0)==1]
  scales_a: (M, K/128) col-major float32  [stride(0)==1, or 1D vector for M=1]
  scales_b: (K/128, N/128) col-major float32  [stride(0)==1]

To satisfy col-major requirement without memory copy:
  weight saved as (N, K) row-major  →  weight.t() is (K, N) with stride(0)=1  ✓
  weight_scale saved as (N/128, K/128)  →  weight_scale.t() is (K/128, N/128) with stride(0)=1  ✓
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.base_config import LinearMethodBase, QuantizationConfig
from sglang.srt.model_loader.weight_utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


try:
    from sgl_kernel import fp8_blockwise_scaled_mm
except ImportError:
    fp8_blockwise_scaled_mm = None


class FP8BlockwiseConfig(QuantizationConfig):
    """Configuration for FP8 blockwise quantization (SM120 UMMA)."""

    def __init__(self, block_size: int = 128, fp8_dtype: str = "float8_e4m3fn"):
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    @classmethod
    def from_config(cls, config: dict) -> "FP8BlockwiseConfig":
        return cls(
            block_size=config.get("block_size", 128),
            fp8_dtype=config.get("fp8_dtype", "float8_e4m3fn"),
        )

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> Optional[LinearMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return FP8BlockwiseLinearMethod(self)
        return None

    def get_name(self) -> str:
        return "fp8_blockwise"


class FP8BlockwiseLinearMethod(LinearMethodBase):
    """FP8 blockwise linear layer using SM120 UMMA.

    Weight (N, K) float8_e4m3fn is stored in standard PyTorch layout.
    At forward time, weight.t() creates a col-major (K, N) view (zero-copy).
    """

    def __init__(self, config: FP8BlockwiseConfig):
        self.config = config
        self.block_size = config.block_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Register FP8 weight and scale parameters.

        Checkpoint stores:
          <prefix>.weight       (N, K) float8_e4m3fn   — standard PyTorch layout
          <prefix>.weight_scale (N/128, K/128) float32
        """
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # FP8 weight: (N, K) = (out, in) — same shape as plain nn.Linear
        weight = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"weight_loader": weight_loader})
        layer.register_parameter("weight", weight)

        # Block scales: (N/128, K/128)
        weight_scale = nn.Parameter(
            torch.empty(
                (output_size_per_partition + self.block_size - 1) // self.block_size,
                (input_size_per_partition + self.block_size - 1) // self.block_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FP8 blockwise GEMM via SM120 UMMA kernel.

        weight is (N, K) row-major; .t() creates (K, N) col-major view (stride[0]=1).
        weight_scale is (N/128, K/128); .t() creates (K/128, N/128) col-major view.
        Both transposes are zero-copy strided views.
        """
        if fp8_blockwise_scaled_mm is None:
            raise RuntimeError(
                "sgl_kernel.fp8_blockwise_scaled_mm not available. "
                "Ensure sgl-kernel is compiled with SM120 support."
            )

        scales_a, x_fp8 = _quantize_activation_fp8_blockwise(x)

        # Zero-copy col-major views required by the kernel
        mat_b = layer.weight.t()          # (K, N) col-major, stride[0]=1
        scales_b = layer.weight_scale.t() # (K/128, N/128) col-major, stride[0]=1

        out = fp8_blockwise_scaled_mm(
            x_fp8,           # (M, K) float8_e4m3fn, row-major
            mat_b,           # (K, N) float8_e4m3fn, col-major
            scales_a,        # (M, K/128) float32, col-major (or vector for M=1)
            scales_b,        # (K/128, N/128) float32, col-major
            out_dtype=torch.bfloat16,
        )

        if bias is not None:
            out = out + bias
        return out


def _quantize_activation_fp8_blockwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation to FP8 blockwise format for kernel input.

    Returns:
        scales_a: (M, K/128) float32, col-major for M>1 (required by kernel)
        x_fp8:   (M, K) float8_e4m3fn, row-major
    """
    M, K = x.shape
    block_size = 128
    FP8_MAX = 448.0

    if K % block_size != 0:
        raise ValueError(f"Activation K={K} not divisible by block_size={block_size}")

    # Reshape for blockwise max: (M, K/128, 128)
    x_blocks = x.reshape(M, K // block_size, block_size)

    # Per-block max → row-major scales (M, K/128)
    max_abs = x_blocks.abs().amax(dim=2)
    scales_row = (max_abs / FP8_MAX).clamp(min=1e-12).to(torch.float32)

    # Kernel requires scales_a.stride(0)==1 (col-major) for M>1.
    # For M=1 the kernel accepts any contiguous 1-D-like tensor.
    if M > 1:
        # .t().contiguous().t() converts (M, K/128) row-major → col-major
        scales_a = scales_row.t().contiguous().t()
    else:
        scales_a = scales_row  # (1, K/128): is_contiguous_vector → passes kernel check

    # Quantize activations
    scale_expanded = scales_row.unsqueeze(2)  # (M, K/128, 1)
    x_scaled = (x_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)

    return scales_a, x_fp8
