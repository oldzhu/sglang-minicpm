# CHANGE Iteration W4A8 #1 — Continuation 001 (Implementation)

> Continuation of `PROPOSAL_iteration_W4A8_001.en.md`. The original proposal
> assumed an offline preprocess hook that injects `weight_fp8` tensors into
> the saved safetensors shards. During Step 2 we adopted a simpler design
> that produces identical runtime behaviour but requires zero on-disk
> format changes. This document records the final implementation.

## 1. Design change vs. original §5 plan

| Item | Original plan | Final plan |
|---|---|---|
| FP8 weight production | Offline post-step in `preprocess_model.py`, written to safetensors as `weight_fp8` / `weight_fp8_scale` | **Load-time** dequant + FP8 quant inside `process_weights_after_loading` (`gptq.py`). Tensors held as non-persistent buffers. |
| `model.safetensors.index.json` | Needs to be regenerated to register new tensors | Untouched. |
| Eligibility tagging | Identify layers via key parsing in preprocess script | `_soar_w4a8_eligible = True` set on the linear modules in `minicpm.py`. |
| Disk size | +~0.5 GB FP8 tensors written to wheel-sized model | No change. Model artifact identical to baseline. |
| Memory at runtime | Same | Same (FP8 tensors live alongside Marlin tensors, ~0.5 GB extra GPU footprint). |
| Submission package | Would have required regenerating quantized model | Reuses the existing GPTQ artifact unchanged. |

**Why the change is safer**: it sidesteps any risk of corrupting the
on-disk model (e.g. mismatched index, partial writes), keeps `prepare_model.sh`
identical, and lets us toggle W4A8 purely via `SOAR_W4A8_FP8_GEMM=1`.

## 2. Files actually changed

1. **`python/sglang/srt/layers/quantization/utils_w4a8_fp8.py`** *(new)*
   - `gptq_int4_dequantize(qweight, qzeros, scales, group_size=128)`
     vectorised PyTorch unpack for `gptqmodel` 4-bit `desc_act=False`
     layout. Returns `(K, N)` BF16.
   - `fp8_blockwise_quantize(w, block_size=128)` returns
     `(weight_fp8, weight_fp8_scale)` in the layout
     `cutlass_w8a8_block_fp8_linear_with_fallback` expects: `(N, K)`
     `float8_e4m3fn` + `(N//128, K//128)` fp32.
   - `fp8_blockwise_dequantize(...)` for the round-trip unit test.

2. **`python/sglang/srt/layers/quantization/gptq.py`**
   - `import os` added.
   - `GPTQMarlinLinearMethod.process_weights_after_loading` calls a new
     helper `_soar_maybe_setup_w4a8_fp8(layer)` **before** the in-place
     Marlin format transform. The helper:
     - Returns silently unless `SOAR_W4A8_FP8_GEMM=1`,
       `layer._soar_w4a8_eligible == True`, the kernel config matches
       4-bit / `group_size=128` / `desc_act=False`, and both partitioned
       dims are multiples of 128.
     - Reads `qweight`, `qzeros`, `scales` from the linear layer,
       dequantises to BF16 `(K, N)`, transposes to `(N, K)`,
       blockwise-quantises to FP8 e4m3, and registers
       `weight_fp8` + `weight_fp8_scale` as non-persistent buffers.
     - Sets `layer._soar_w4a8_active = True` on success.
   - `GPTQMarlinLinearMethod.apply` early-returns through
     `cutlass_w8a8_block_fp8_linear_with_fallback` when
     `_soar_w4a8_active` is set; on any exception it logs, clears the
     flag, and falls back to the standard `apply_gptq_marlin_linear`
     path.

3. **`python/sglang/srt/models/minicpm.py`**
   - `MiniCPMMLP.__init__` tags `gate_up_proj` and `down_proj` with
     `_soar_w4a8_eligible = True`.
   - `MiniCPMAttention.__init__` (std-attn / `mixer_type == "minicpm4"`)
     tags `qkv_proj` and `o_proj` with `_soar_w4a8_eligible = True`.
   - `MiniCPMLightningMixer` is intentionally left untagged so its
     QKV/O projections continue to use BF16 Marlin (per amendment #3 of
     the original proposal).

4. **`benchmark/soar/demo_sala/prepare_env.sh`**
   - Added `export SOAR_W4A8_FP8_GEMM="${SOAR_W4A8_FP8_GEMM:-0}"` next to
     the other lightning-attention env flags.
   - Added the corresponding diagnostic `echo` line near the bottom of
     the script.

5. **`test/srt/quantization/test_utils_w4a8_fp8.py`** *(new)*
   - CPU-only smoke tests:
     - `test_gptq_int4_dequantize_synthetic` — packs a known INT4
       weight + INT4 zero point in the gptqmodel layout, then verifies
       the helper reproduces the canonical `(q - z) * s` reference to
       `< 1e-6` relative Frobenius error.
     - `test_fp8_blockwise_quantize_roundtrip` — quantise a random BF16
       `(512, 384)` tensor and verify reconstruction error is `< 2e-2`
       relative Frobenius.
     - `test_fp8_blockwise_handles_zero_block` — guards against NaN
       scales on all-zero blocks.

## 3. Files NOT changed

- `benchmark/soar/demo_sala/preprocess_model.py` — untouched.
- `benchmark/soar/demo_sala/prepare_model.sh` — untouched.
- `model.safetensors.index.json` and the safetensors shards — untouched.

## 4. Validation plan (fcloud)

1. `python3 scripts/fcloud/fcloud_workflow.py sync`
2. `python3 scripts/fcloud/fcloud_workflow.py restart-server`
   - With `SOAR_W4A8_FP8_GEMM=0` (default). Expect identical baseline
     S1 / S8 / Smax.
3. Run unit tests on fcloud:
   ```bash
   cd /root/submission_sim
   python3 sglang/test/srt/quantization/test_utils_w4a8_fp8.py
   ```
4. Edit fcloud `/root/submission_sim/prepare_env.sh` to set
   `SOAR_W4A8_FP8_GEMM=1` (or export it before `restart-server`).
5. `restart-server` → `wait-server` → `accuracy` → `speed --variant all`.
6. Compare against v18 baseline (Test 12: S1=121.71s, S8=44.09s,
   Smax=35.86s, accuracy=79.29%).
7. Server log should contain
   `[SOAR W4A8] enabled FP8 blockwise GEMM for layer prefix=...` lines
   (one per std-attn / MLP linear, lightning layers absent).

## 5. Rollback

- Set `SOAR_W4A8_FP8_GEMM=0` (or `unset`) and restart the server.
- The model artifact is unchanged, so no re-quantisation is needed.

## 6. Next steps

- Run on fcloud and capture results (Test 13 in the tracking sheet).
- If the FP8 path raises at runtime, the per-layer warning + automatic
  fallback ensures the server stays up; we then use those logs to
  diagnose without an accuracy regression.
- Future iteration: extend the `_soar_w4a8_eligible` whitelist to
  lightning Q/K/V/O once we have data.
