"""FP8 blockwise quantization for SM120 UMMA.

This module provides FP8 blockwise quantization using the existing SM120 FP8 kernel
from sgl-kernel (fp8_blockwise_scaled_mm).

Key properties:
- Weights: Pre-quantized offline to FP8 blockwise format
- Activations: Quantized per-row per-128-K-block at inference time
- Kernel: Uses SM120 UMMA tcgen05.mma.ws.sync (already in sgl-kernel)
- No new CUDA code needed
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.base_config import LinearMethodBase, QuantizationConfig

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


try:
    from sgl_kernel import fp8_blockwise_scaled_mm
except ImportError:
    fp8_blockwise_scaled_mm = None


class FP8BlockwiseConfig(QuantizationConfig):
    """Configuration for FP8 blockwise quantization (SM120 UMMA).
    
    This quantization uses 8-bit FP8 E4M3 weights with blockwise scaling (128×128 blocks),
    matching the SM120 UMMA tile size.
    """

    def __init__(
        self,
        block_size: int = 128,
        fp8_dtype: str = "float8_e4m3fn",
    ):
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    @classmethod
    def from_config(cls, config: dict) -> FP8BlockwiseConfig:
        """Load from quantization_config dict."""
        return cls(
            block_size=config.get("block_size", 128),
            fp8_dtype=config.get("fp8_dtype", "float8_e4m3fn"),
        )

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> Optional[LinearMethodBase]:
        """Get the quantization method for this layer."""
        from sglang.srt.layers.linear import LinearBase
        
        if isinstance(layer, LinearBase):
            return FP8BlockwiseLinearMethod(self)
        return None

    def get_name(self) -> str:
        return "fp8_blockwise"


class FP8BlockwiseLinearMethod(LinearMethodBase):
    """FP8 blockwise linear layer using SM120 UMMA.
    
    Weights are pre-quantized to FP8 (shape N×K, col-major) with blockwise scales.
    Activations are quantized per-row per-128-K-elements at inference time.
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
        """Create FP8 blockwise weights for the layer.
        
        Expected to find pre-quantized weights:
        - layer.weight_fp8: (N, K) float8_e4m3fn
        - layer.weight_fp8_scale: (N/128, K/128) float32
        
        These are loaded from the preprocessed model (preprocess_model.py).
        """
        # Weights should already be loaded from model files
        # Just validate they exist
        if not hasattr(layer, "weight_fp8"):
            raise ValueError(
                f"Layer {layer} missing weight_fp8 (expected from FP8 blockwise model)"
            )
        if not hasattr(layer, "weight_fp8_scale"):
            raise ValueError(
                f"Layer {layer} missing weight_fp8_scale (expected from FP8 blockwise model)"
            )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply FP8 blockwise GEMM.
        
        Args:
            layer: Linear layer with weight_fp8 and weight_fp8_scale attributes
            x: Input tensor (M, K) float16 or bfloat16
            bias: Optional bias tensor
        
        Returns:
            Output tensor (M, N) bfloat16
        """
        if fp8_blockwise_scaled_mm is None:
            raise RuntimeError(
                "sgl_kernel.fp8_blockwise_scaled_mm not available. "
                "Make sure sgl-kernel is compiled with SM120 support."
            )

        # Quantize activation to FP8 blockwise
        scales_a, x_fp8 = _quantize_activation_fp8_blockwise(x)

        # Call SM120 FP8 GEMM kernel
        out = fp8_blockwise_scaled_mm(
            x_fp8,                    # (M, K) float8_e4m3fn
            layer.weight_fp8,         # (N, K) float8_e4m3fn, col-major
            scales_a,                 # (M, K/128) float32
            layer.weight_fp8_scale,   # (N/128, K/128) float32
            out_dtype=torch.bfloat16,
        )

        if bias is not None:
            out = out + bias

        return out


def _quantize_activation_fp8_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation to FP8 blockwise format.
    
    Args:
        x: Activation tensor (M, K) float16 or bfloat16
    
    Returns:
        scales_a: (M, K/128) float32 scales
        x_fp8: (M, K) float8_e4m3fn quantized activation
    """
    M, K = x.shape
    block_size = 128
    FP8_MAX = 448.0
    
    if K % block_size != 0:
        raise ValueError(f"Activation K={K} not divisible by block_size={block_size}")
    
    # Reshape to (M, K/128, 128) for blockwise max
    x_blocks = x.reshape(M, K // block_size, block_size)
    
    # Compute max absolute value per block
    max_abs = x_blocks.abs().amax(dim=2)  # (M, K/128)
    
    # Compute scales
    scales_a = (max_abs / FP8_MAX).clamp(min=1e-12).to(torch.float32)
    
    # Quantize
    scale_expanded = scales_a.unsqueeze(2)  # (M, K/128, 1)
    x_scaled = (x_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    
    return scales_a, x_fp8
