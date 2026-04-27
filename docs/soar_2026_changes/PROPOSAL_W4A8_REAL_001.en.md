# PROPOSAL: True W4A8 — Real INT4-storage + FP8-compute kernel

**Status: PROPOSAL (awaiting approval).** No code changes will be made until approved.

## Background and motivation

CHANGE_W4A8_001 (commit `7ce21c3f5`) was **mislabeled**. The implementation did **W8A8 FP8 blockwise** — at load time we dequantized GPTQ INT4 weights to BF16 and re-quantized to **FP8 (8-bit) storage**, then ran cutlass FP8 × FP8 GEMM. This doubled the weight memory footprint vs the Marlin INT4 baseline (W4A16).

Result on fcloud (SM120 RTX PRO 6000):
- S1 +118%, S8 +56%, Smax +30% vs v18 baseline (W4A16 Marlin).

This regression is the **predictable consequence** of giving up the 2× weight-bandwidth advantage of INT4 on a memory-bandwidth-bound decode workload. It does **not** refute the W4A8 hypothesis.

**True W4A8** keeps INT4 weight storage *and* uses FP8/INT8 tensor cores for the MMA, capturing both wins:
- 2× weight bandwidth (INT4 vs FP8/BF16 storage)
- 2× compute throughput (FP8 QMMA / INT8 IMMA = 296 TF on SM120 vs BF16 = 148 TF)

Theoretical S1 improvement vs Marlin W4A16:
- Same weight bandwidth as Marlin (INT4 storage retained)
- FP8/INT8 activation = 2× smaller → reduces activation memory traffic
- Faster MMA on sustained large-M (matters more for prefill/S8/Smax than decode)
- Realistic gain estimate: **5–15% on S1 (decode)**, **10–25% on S8/Smax (prefill mix)**

## Rule-compliance check (SOAR constraints)

- **Quantize on-site, ≤ 5h**: weight dequant/requant happens at load time per submission; activation quantization happens per-forward. Both are deterministic, fast, and well within budget.
- **2GB submission**: kernel is built into sgl-kernel wheel; no model artifact size growth.
- **Apache 2.0 / reproducible / explainable**: all candidate kernels below have permissive licenses.
- **No reliance on forbidden tricks**: no prefix cache abuse, no eval harness modification.

## Risk to accuracy/stability

| Path | Activation precision | Accuracy risk |
|---|---|---|
| QQQ-style W4-INT8 Marlin | INT8 (8-bit symmetric per-token) | **Medium**: INT8 activation has been validated extensively in vllm/QQQ; per-token symmetric quant is well-behaved. Risk ~ same as FP8 e5m2 KV (already in baseline). |
| W4A8-Machete (FP8 e4m3 activation) | FP8 e4m3 (per-token scale) | **Low–Medium**: FP8 e4m3 has 3 mantissa bits but per-token scaling makes it competitive with BF16 for transformer activations. |
| Custom CUTLASS mixed-input | configurable | depends on choice |

Acceptance criterion: **normalized accuracy ≥ 99% (C=1.0)** on local public set, same as v18 baseline. If the kernel candidate drops C below 1.0, abandon that specific kernel and try next.

## Three candidate paths

### Option A: QQQ-style W4-INT8 Marlin (RECOMMENDED first attempt)

- **Source**: https://github.com/IST-DASLab/marlin (W4A8 fork) and https://github.com/HandH1998/QQQ
- **What it does**: Marlin-style INT4 weight + INT8 activation kernel. Per-token symmetric INT8 activation quantization. Uses INT8 IMMA tensor cores (296 TF on SM120, same as FP8 QMMA peak).
- **Why best first**: closest to existing Marlin code path (we already use Marlin for INT4); kernel is mature, has ampere/ada/hopper instantiations.
- **SM120 effort**: Marlin SM120 tile table already added in CHANGE_0125. Adding W4-INT8 instantiations requires a few new tile tuples in the same dispatch table. Estimated **medium effort**.

### Option B: W4A8-Machete (vllm/compressed-tensors)

- **Source**: https://github.com/vllm-project/vllm `vllm/model_executor/layers/quantization/utils/machete_utils.py` and CUTLASS-based kernel.
- **What it does**: CUTLASS mixed-input GEMM with INT4 weight (packed) + FP8 e4m3 activation. Hopper-optimized.
- **SM120 effort**: Hopper/SM90 first-class; SM120 (Blackwell) backport status unclear. May or may not work out-of-the-box. **High risk on effort.**
- **Why considered**: native FP8 activation is more model-friendly than INT8 for some transformer architectures.

### Option C: Custom CUTLASS mixed-input GEMM

- **What**: Build a CUTLASS 3.x mixed-input GEMM (INT4 weight, FP8 activation, BF16 accumulator+output) with per-K-block dequant in the warp prologue.
- **Effort**: Highest. Requires CUTLASS expertise and tuning for SM120.
- **When**: Only if A and B both fail.

## Recommended plan

1. **Iteration 1** (this proposal, if approved): try **Option A — QQQ-style W4-INT8 Marlin** first. It is the closest to existing infrastructure.
2. If Option A is gated by SM120 instantiation work too large or accuracy regresses, fall back to **Option B — Machete**.
3. **Option C** held as last resort.

## Detailed implementation plan (Option A — to be filled in next iteration if approved)

Files that would change (no code yet — proposal stage):

- `sgl-kernel/csrc/`: add W4-INT8 Marlin kernel files (or fold into existing `gptq_marlin.cu` with new template instantiations for INT8 activation).
- `sgl-kernel/cmake/`: add SM120 W4A8 tile table.
- `python/sglang/srt/layers/quantization/gptq.py`: add `_soar_maybe_setup_w4a8_int8` helper that:
  - Verifies layer is INT4 GPTQ (skip INT8 sparse_qkv via existing bit-width guard).
  - Keeps existing Marlin INT4 weight format intact (no requant!).
  - Adds per-token INT8 activation quantizer (online, in `apply()`).
  - Calls new `marlin_gemm_w4a8_int8_kernel(...)` from sgl-kernel.
- `python/sglang/srt/models/minicpm.py`: reuse existing `_soar_w4a8_eligible` tags from CHANGE_W4A8_001 (no MiniCPM model changes needed).
- `benchmark/soar/demo_sala/prepare_env.sh`: add new env flag `SOAR_W4A8_INT8_GEMM` (separate from the now-deprecated `SOAR_W4A8_FP8_GEMM`).

## Validation commands (planned)

Same workflow as iteration 001:

```bash
# CPU unit test for activation quantizer + dequant utilities
python3 test/srt/quantization/test_w4a8_int8_quantizer.py

# fcloud
python3 scripts/fcloud/fcloud_workflow.py setup
sed -i 's/SOAR_W4A8_INT8_GEMM:-0/SOAR_W4A8_INT8_GEMM:-1/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Result success/failure criteria

| Metric | Pass | Fail |
|---|---|---|
| Accuracy (normalized) | ≥ 99% (C=1.0) | < 99% (any C drop) |
| S1 | ≤ 121.71s (Marlin baseline) | > 121.71s |
| S8 | ≤ 44.09s | > 44.09s |
| Smax | ≤ 35.86s | > 35.86s |

If accuracy passes but speed fails, **try INT8 activation block size tuning** before abandoning. If accuracy fails on any kernel, abandon that kernel; try next option.

## Rollback instructions

W4A8-INT8 path will be env-gated `SOAR_W4A8_INT8_GEMM=0` by default. To disable:
```bash
sed -i 's/SOAR_W4A8_INT8_GEMM:-1/SOAR_W4A8_INT8_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```
Code revert: `git revert <iteration-2-commit-sha>`.

## Open questions awaiting user decision

1. **Approve Option A** (QQQ-style W4-INT8 Marlin) as the first real W4A8 attempt? **Yes / No**
2. If yes, are there constraints on **kernel source** (must port from QQQ repo? must be from-scratch CUTLASS? acceptable to vendor a kernel under Apache-2.0?)
3. Estimated time/effort budget for kernel bring-up before falling back to Option B?

This proposal makes **no code changes** until you approve.

---

**Companion ZH document**: `PROPOSAL_W4A8_REAL_001.zh.md`.
