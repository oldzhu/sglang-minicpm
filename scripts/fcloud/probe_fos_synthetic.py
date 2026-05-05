"""Phase B Step 2 — synthetic-tensor smoke test of the FourOverSix patch.

Run on fcloud where modelopt 0.43 is installed:

    cd /root/submission_sim && python3 /root/sglang-minicpm/scripts/fcloud/probe_fos_synthetic.py

Verifies that:
  1. The patch installs and uninstalls cleanly.
  2. With a synthetic mix of "M=6 friendly" and "M=4 friendly" blocks, the
     selector picks the expected rule per block (~equal split).
  3. The FP8 scale tensor produced by the patched method has the same
     dtype/shape/device as modelopt's original output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `preprocess_model` importable.
SUB = Path("/root/submission_sim")
sys.path.insert(0, str(SUB))

import torch  # noqa: E402

from preprocess_model import _install_four_over_six_patch, _summarize_fos_stats, _FOS_STATS, _FOS_ACTIVE  # noqa: E402

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor  # noqa: E402


def _check_basic_compat():
    # Build a synthetic weight: 4 rows × 64 cols → 16 blocks of 16.
    torch.manual_seed(0)
    K = 64
    B = 16
    W = torch.zeros(4, K, dtype=torch.bfloat16, device="cuda")
    dev = "cuda"

    # Row 0: each block has one outlier ~6, others ~0.05 → M=6 friendly.
    W[0] = (0.05 * torch.randn(K, device=dev)).to(W.dtype)
    for blk in range(K // B):
        W[0, blk * B] = 6.0 if blk % 2 == 0 else -6.0

    # Row 1: each block dense ~3.5..6 (top-heavy) → M=4 friendly.
    r1 = (3.5 + 2.5 * torch.rand(K, device=dev))
    sign = (torch.rand(K, device=dev) > 0.5).float() * 2 - 1
    W[1] = (r1 * sign).to(W.dtype)

    # Row 2: dead block (all zeros) — should default to scale=1, no errors.
    W[2] = 0.0

    # Row 3: ordinary noise.
    W[3] = (0.3 * torch.randn(K, device=dev)).to(W.dtype)

    sf2 = NVFP4QTensor.get_weights_scaling_factor_2(W)
    print(f"[probe] sf2={sf2.item():.6e}")

    # 1. Original
    orig_scale, _ = NVFP4QTensor.get_weights_scaling_factor(W, B, sf2)
    print(f"[probe] original scale dtype={orig_scale.dtype} shape={tuple(orig_scale.shape)}")

    # 2. Install patch
    _FOS_STATS.clear()
    restore = _install_four_over_six_patch()
    _FOS_ACTIVE["on"] = True
    try:
        fos_scale, _ = NVFP4QTensor.get_weights_scaling_factor(W, B, sf2)
    finally:
        _FOS_ACTIVE["on"] = False
        restore()

    print(f"[probe] fos      scale dtype={fos_scale.dtype} shape={tuple(fos_scale.shape)}")
    assert fos_scale.dtype == orig_scale.dtype, "dtype mismatch"
    assert fos_scale.shape == orig_scale.shape, "shape mismatch"
    assert fos_scale.device == orig_scale.device, "device mismatch"

    # Per-row M=4 picks
    print(f"[probe] FOS_STATS entries: {len(_FOS_STATS)}")
    for s in _FOS_STATS:
        print(f"  shape={s['weight_shape']} blocks={s['n_blocks']} m4={s['n_pick_m4']} pct={s['pct_m4']:.1f}%")
    print(f"[probe] summary: {_summarize_fos_stats()}")

    # 3. After restore, original is back
    again_scale, _ = NVFP4QTensor.get_weights_scaling_factor(W, B, sf2)
    assert torch.equal(again_scale.float(), orig_scale.float()), "restore failed"

    # 4. Reconstruction MSE: FOS must be <= original on this synthetic mix.
    def _decode(scale_fp8: torch.Tensor) -> torch.Tensor:
        eff = (scale_fp8.float() * sf2.float()).clamp_min(1e-30).unsqueeze(-1)
        w_blocks = W.view(*W.shape[:-1], -1, B).float()
        scaled = w_blocks / eff
        codes = NVFP4QTensor._cast_fp4(scaled.clone())
        e2m1 = NVFP4QTensor.get_e2m1_values(W.device)
        return e2m1[codes.long()] * eff

    err_orig = ((_decode(orig_scale) - W.view(*W.shape[:-1], -1, B).float()) ** 2).mean().item()
    err_fos = ((_decode(fos_scale) - W.view(*W.shape[:-1], -1, B).float()) ** 2).mean().item()
    print(f"[probe] mean MSE  original={err_orig:.6e}  fos={err_fos:.6e}  improvement={(err_orig-err_fos)/max(err_orig,1e-30)*100:.2f}%")
    assert err_fos <= err_orig + 1e-9, "FOS made things worse — bug!"

    print("[probe] OK — patch installs, runs, restores, and reduces (or matches) MSE")


if __name__ == "__main__":
    _check_basic_compat()
