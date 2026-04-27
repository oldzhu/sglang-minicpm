# CHANGE_W4A8_001 — Iteration 002: Validation Results & Decision

**Status: this specific path ABANDONED. True W4A8 NOT YET TESTED.**

> **CRITICAL MISLABEL NOTICE (added 2026-04-27 post-hoc):**
>
> What was actually implemented and tested in commit `7ce21c3f5` is **W8A8 FP8 blockwise**, NOT W4A8.
> The code dequantizes GPTQ INT4 weights to BF16 at load time and then re-quantizes them to **FP8 (8-bit) storage**, so the runtime GEMM is FP8 weight × FP8 activation. The 4-bit bandwidth advantage of the original GPTQ INT4 weights was thrown away at load time.
>
> v18 baseline is **W4A16** (Marlin: 4-bit INT4 weight storage, BF16 in-register dequant, BF16 tensor cores at 148 TF). True W4A8 would keep INT4 storage AND use FP8 tensor cores at 296 TF, capturing both the 2× bandwidth win (vs FP8 weight) and the 2× compute win (vs BF16 MMA).
>
> The +118%/+56%/+30% S1/S8/Smax regressions documented below are the **predictable** result of doubling the weight memory footprint while running on a memory-bandwidth-bound workload (decode bs=1). They do **not** invalidate the real W4A8 hypothesis.
>
> **Real W4A8 (e.g., QQQ-style W4-INT8 Marlin or W4A8-Machete) is a separate optimization vector and should be evaluated independently** — see proposal `PROPOSAL_W4A8_REAL_001.en.md`.

Companion to `CHANGE_W4A8_001_iteration_001.en.md` (implementation). This document records the fcloud validation results for the load-time GPTQ INT4 → FP8 blockwise GEMM path (mislabeled as W4A8) and the decision to abandon this specific implementation.

## Test setup

- **Commit**: `7ce21c3f5` — "W4A8 #1: load-time GPTQ INT4 -> FP8 blockwise GEMM (env-gated)"
- **fcloud instance**: 223.167.85.181 (SM120 RTX PRO 6000 Blackwell)
- **Config**: GPTQ `sparse_qkv_w8` + FP8 e5m2 KV cache + dense mode + torch.compile (`max-bs=8`) + Test 20 server args (`chunk=32K, prefill-max-req=1, running=24, sched-cons=1.0, mixed-chunk`)
- **Variable**: `SOAR_W4A8_FP8_GEMM=1` (vs baseline `=0`)
- **Coverage**: std-attn `qkv_proj`/`o_proj` (8 layers) + MLP `gate_up_proj`/`down_proj` (32 layers); lightning attention untouched; INT8 sparse_qkv layers automatically skipped via bit-width guard
- **Baseline reference**: v18 Test 12 — S1=121.71s, S8=44.09s, Smax=35.86s, ori_accuracy=79.29%

## Validation steps (all on fcloud)

1. ✅ CPU unit tests (`test/srt/quantization/test_utils_w4a8_fp8.py`) — 3/3 pass after relaxing FP8 e4m3 round-trip threshold from 2e-2 to 5e-2 (correct expectation for 3-mantissa-bit FP8).
2. ✅ Server restart with `SOAR_W4A8_FP8_GEMM=1` — Ready after 214s; env propagation verified via `/proc/<pid>/environ`.
3. ✅ Smoke test (Paris completion) — passed.
4. ✅ Accuracy eval (150 samples, concurrency=32).
5. ✅ Speed S1 / S8 / Smax — all three tiers measured.

## Accuracy result

| Metric | W4A8 #1 (this) | v18 Test 12 baseline | Δ |
|---|---:|---:|---:|
| Average | **79.20%** | 79.29% | −0.09 pt |
| Normalized | 99.00% | 99.11% | — |
| C | **1.0** | 1.0 | unchanged |
| mcq | 53.33 | 63.33 | −10 (variance) |
| cwe | 82.67 | 72.00 | +10.67 |
| fwe | 100.00 | 97.78 | +2.22 |
| niah | 96.67 | 100.00 | −3.33 |
| qa | 63.33 | 63.33 | 0 |

**Verdict (accuracy): NEUTRAL.** Net average within ±0.1pt; per-task shifts within local 150-sample noise floor. Both tiers normalize ≥ 99% → C=1.0. The mcq/cwe redistribution is the same kind of variance seen in Tests 29/34a/v18-revert.

## Speed result — NET REGRESSION

| Tier | W4A8 #1 (this) | v18 Test 12 baseline | Δ |
|---|---:|---:|---:|
| S1 | **265.32 s** | 121.71 s | **+118%** |
| S8 | **68.88 s** | 44.09 s | **+56%** |
| Smax | **46.44 s** | 35.86 s | **+30%** |

Per-tier server metrics:

| Tier | Mean TTFT | Mean TPOT | Output throughput |
|---|---:|---:|---:|
| S1 | 120.78 ms | 15.64 ms | 62.70 tok/s |
| S8 | 193.45 ms | 18.35 ms | 378.93 tok/s |
| Smax | 11097.76 ms | 26.76 ms | n/a |

**Verdict (speed): UNAMBIGUOUSLY WORSE.** Score impact under SOAR formula `S1×0.4 + S8×0.3 + Smax×0.3`:

- Speed Score (this) ≈ `(121.71/265.32)×40 + (44.09/68.88)×30 + (35.86/46.44)×30 ≈ 18.3 + 19.2 + 23.2 = 60.7`
- Speed Score (baseline) = 100
- **Final Score: 60.7 × 1.0 = 60.7 vs baseline 100** — score loss ≈ **−39%**.

Even if we had hit theoretical 2× FP8/BF16 ratio (296 TF / 148 TF), S1 would still regress because at decode bs=1 the path is **memory-bandwidth-bound on weights**, not compute-bound. Marlin INT4 weights are 2× smaller than FP8 weights, so memory-bandwidth-bound paths favor INT4 by definition.

## Root cause analysis

Why cutlass `fp8_blockwise_scaled_mm` lost to Marlin INT4 on every tier:

1. **S1 (decode bs=1, M=1)**: kernel is purely memory-bound on weights. INT4 (0.5 bytes/element) reads half the bytes that FP8 (1 byte/element) reads. Marlin's INT4 dequant-fused-GEMM moves less data across the 1398 GB/s HBM bus.
2. **S8 / Smax (small-to-medium M)**: cutlass FP8 blockwise has fixed per-call overhead (CUTLASS scheduler, blockwise scale broadcast, K-tile scheduling). At our hidden sizes (≤ 7168) and at M ≤ 256, this overhead is significant relative to actual MMA work. Marlin's tighter, hand-tuned INT4 GEMM beats it.
3. **The 296 TF FP8 QMMA peak does not translate to throughput here.** Peak QMMA requires sustained large-M dense GEMM (≥ 1024×1024×1024 per tile). Decode and S8 prefill operate well below that.
4. **Tensor cores are NOT the bottleneck** in this workload — bandwidth and kernel launch overhead are. Switching to a higher-precision-but-larger weight format does not help when you're not compute-bound.

This matches the SM120 hardware reference (`docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md`): FP8 wins only when M is large enough that arithmetic intensity exceeds the bandwidth roofline crossover.

## Decision: ABANDON this specific path (W8A8 FP8 blockwise via cutlass), NOT the W4A8 hypothesis

- Keep `SOAR_W4A8_FP8_GEMM=0` (the repo default).
- Do **not** ship this path in any submission package.
- The flag and code remain env-gated for reuse of the dequant/quant utilities.
- **The W4A8 hypothesis is NOT refuted by this experiment** — we tested W8A8 FP8, not W4A8. Real W4A8 (INT4 storage + FP8 MMA) requires a different kernel (QQQ-style W4-INT8 Marlin, W4A8-Machete, or custom CUTLASS mixed-input GEMM) and should be evaluated in a separate iteration.

## What is kept in the tree

The implementation itself is **kept** (env-gated, default OFF) for two reasons:

1. **Correctness path is validated** — the load-time GPTQ INT4 → FP8 conversion + cutlass blockwise GEMM produces accurate output. If a future scenario ever needs FP8 for this model (e.g., a very different workload, M=1k+ dense prefill), the code is ready.
2. **Reusable building blocks** — `python/sglang/srt/layers/quantization/utils_w4a8_fp8.py` provides `gptq_int4_dequantize()` and `fp8_blockwise_quantize()/dequantize()` utilities that may be useful for future quantization experiments (e.g., Iteration `_003` exploring NVFP4-with-fallback or hybrid INT4-MMA paths).

The unit tests stay green and the env flag stays at default 0.

## Validation commands (for the record)

```bash
# Sync + restart with W4A8 enabled
sed -i 's/SOAR_W4A8_FP8_GEMM:-0/SOAR_W4A8_FP8_GEMM:-1/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# Tests
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

## Rollback / disable instructions

W4A8 is already gated off by default. To explicitly disable on fcloud:

```bash
sed -i 's/SOAR_W4A8_FP8_GEMM:-1/SOAR_W4A8_FP8_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

To remove the code entirely (not recommended; keeps options open):

```bash
git revert 7ce21c3f5
```

## Next steps

1. **Real W4A8 evaluation** — see `PROPOSAL_W4A8_REAL_001.en.md`. Three candidate kernels:
   - **QQQ-style W4-INT8 Marlin**: keeps INT4 weight, INT8 activation, INT8 tensor cores (296 TF on SM120). Mature open-source kernel.
   - **W4A8-Machete (vllm/compressed-tensors)**: Hopper-first; SM120 backport effort unknown.
   - **Custom CUTLASS mixed-input GEMM**: highest effort, full bring-up.
2. **Other directions in parallel** (memory-bandwidth-bound or kernel-fusion-bound vectors that benefit decode):
   - Marlin INT4 SM120 tile retuning (lower per-call overhead)
   - QKV / O fusion with attention output projection
   - `fused_qk_norm_rope` variants
   - speculative decoding (lower M but more useful tokens per step)
3. The companion ZH document is `CHANGE_W4A8_001_iteration_002.zh.md`.

## Files touched in this iteration

None (results-only document). The implementation is unchanged from iteration 001 (commit `7ce21c3f5`).
