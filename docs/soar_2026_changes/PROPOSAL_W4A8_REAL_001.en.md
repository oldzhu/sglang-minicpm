# PROPOSAL: True W4A8 — Real INT4-storage + FP8-compute kernel

**Status: PROPOSAL (awaiting approval).** No code changes will be made until approved.

## Background and motivation

CHANGE_W4A8_001 (commit `7ce21c3f5`) was **mislabeled**. The implementation did **W8A8 FP8 blockwise** — at load time we dequantized GPTQ INT4 weights to BF16 and re-quantized to **FP8 (8-bit) storage**, then ran cutlass FP8 × FP8 GEMM. This doubled the weight memory footprint vs the Marlin INT4 baseline (W4A16).

Result on fcloud (SM120 RTX PRO 6000):
- S1 +118%, S8 +56%, Smax +30% vs v18 baseline (W4A16 Marlin).

This regression is the **predictable consequence** of giving up the 2× weight-bandwidth advantage of INT4 on a memory-bandwidth-bound decode workload. It does **not** refute the W4A8 hypothesis.

**True W4A8** keeps INT4 weight storage *and* uses 8-bit tensor cores for the MMA, capturing both wins:
- 2× weight bandwidth (INT4 vs FP8/BF16 storage)
- 2× compute throughput vs BF16, **but only via FP8 QMMA = 296 TF on SM120**

### SM120 dtype availability (CRITICAL — drives kernel choice)

Per `docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md` (the authoritative hardware reference for this competition):

| Tensor core dtype | TFLOPS on SM120 | Status |
|---|---|---|
| BF16/FP16 | 148 | listed |
| **FP8 (e4m3 / e5m2)** | **296** | **listed, guaranteed** |
| FP4 | 593 | listed |
| **INT8** | **NOT LISTED** | unverified — likely throttled vs Ada/Hopper on Blackwell consumer |

This is decisive: **INT8 IMMA throughput on SM120 is not enumerated in the official spec.** NVIDIA's Blackwell consumer (GB202) datasheet does include INT8 tensor cores, but throughput may be reduced relative to FP8 QMMA. Picking INT8 activation gives no guaranteed compute win on this hardware. **FP8 activation is the safe target.**

Theoretical S1 improvement vs Marlin W4A16 (with FP8 activation path):
- Same weight bandwidth as Marlin (INT4 storage retained)
- FP8 activation = 2× smaller than BF16 → reduces activation memory traffic
- FP8 QMMA at 296 TF = 2× BF16 (matters more for prefill/S8/Smax than decode)
- Realistic gain estimate: **5–15% on S1 (decode)**, **10–25% on S8/Smax (prefill mix)**

## Rule-compliance check (SOAR constraints)

- **Quantize on-site, ≤ 5h**: weight dequant/requant happens at load time per submission; activation quantization happens per-forward. Both are deterministic, fast, and well within budget.
- **2GB submission**: kernel is built into sgl-kernel wheel; no model artifact size growth.
- **Apache 2.0 / reproducible / explainable**: all candidate kernels below have permissive licenses.
- **No reliance on forbidden tricks**: no prefix cache abuse, no eval harness modification.

## Risk to accuracy/stability

| Path | Activation precision | Accuracy risk |
|---|---|---|
| W4-FP8 (Machete-style or custom CUTLASS) | FP8 e4m3 (per-token scale) | **Low–Medium**: FP8 e4m3 has 3 mantissa bits but per-token scaling makes it competitive with BF16 for transformer activations; widely deployed in production stacks. |
| W4-INT8 Marlin (QQQ-style) | INT8 (8-bit symmetric per-token) | **Medium**: INT8 activation is well-validated in vllm/QQQ. Risk ~ same as FP8 e5m2 KV (already in baseline). **But gated by INT8 IMMA throughput question on SM120.** |
| Custom CUTLASS mixed-input | configurable | depends on choice |

Acceptance criterion: **normalized accuracy ≥ 99% (C=1.0)** on local public set, same as v18 baseline. If the kernel candidate drops C below 1.0, abandon that specific kernel and try next.

## Three candidate paths (REORDERED — FP8 first to guarantee SM120 compute win)

### Option A (RECOMMENDED): W4 + FP8 activation — Machete-style or custom CUTLASS mixed-input

- **Source**: vllm Machete kernel (https://github.com/vllm-project/vllm `csrc/quantization/machete/`); upstream CUTLASS 3.x mixed-input examples.
- **What it does**: Mixed-input GEMM with **INT4 packed weight** + **FP8 e4m3 activation**, dequantized into FP8 register pair feeding **FP8 QMMA at 296 TFLOPS** on SM120. INT4 storage is retained end-to-end (HBM → L2 → SMEM → register).
- **Why first**: SM120 FP8 QMMA throughput is **explicitly listed at 296 TF in the official hardware reference**. This is the only 8-bit MMA path with a guaranteed compute win on this hardware.
- **SM120 effort**: Machete is Hopper/SM90 first-class; SM120 (Blackwell) backport is the main risk. CUTLASS 3.x has SM120 collective builders for FP8 (used by sgl-kernel `cutlass_w8a8_fp8`), so the mixed-input variant is plausible but needs bring-up. Estimated **medium-to-high effort** dominated by tile/instruction selection on SM120.

### Option B (FALLBACK): W4 + INT8 activation — QQQ-style Marlin

- **Source**: https://github.com/IST-DASLab/marlin (W4A8 fork) and https://github.com/HandH1998/QQQ
- **What it does**: Marlin-style INT4 weight + INT8 activation kernel using INT8 IMMA tensor cores.
- **Caveat**: SM120 INT8 IMMA throughput is **NOT enumerated** in the official hardware reference. Before choosing this option we MUST run a microbenchmark on the fcloud GPU (`cutlass_profiler` INT8 GEMM at large M) to confirm INT8 IMMA ≥ FP8 QMMA. If it lands at BF16 rate (148 TF), this option only delivers the bandwidth win — same as Marlin W4A16 — and is not worth the engineering cost.
- **Why fallback**: closest to existing Marlin code path; mature kernel with ampere/ada/hopper instantiations; CHANGE_0125 already added Marlin SM120 tile table. **Lowest engineering effort once INT8 IMMA throughput is verified.**

### Option C (LAST RESORT): Custom CUTLASS mixed-input GEMM

- **What**: Build a CUTLASS 3.x mixed-input GEMM from scratch (INT4 weight, FP8 activation, BF16 accumulator+output) with per-K-block dequant in the warp prologue, tuned for SM120.
- **Effort**: Highest. Requires CUTLASS expertise and full SM120 tuning sweep.
- **When**: Only if A (Machete bring-up) and B (INT8 IMMA microbenchmark + QQQ port) both fail.

## Recommended plan

1. **Phase 0 (cheap microbenchmark before any kernel work)**: run `cutlass_profiler` on fcloud to measure SM120 INT8 IMMA throughput at large M. Result decides whether Option B is even viable.
2. **Iteration 1** (if approved): try **Option A — W4-FP8 Machete-style** first. FP8 QMMA is the only path with guaranteed 296 TF on SM120.
3. If Option A bring-up cost is prohibitive AND Phase 0 confirmed INT8 IMMA ≈ FP8 QMMA, fall back to **Option B — QQQ W4-INT8 Marlin**.
4. **Option C** held as last resort.

## Detailed implementation plan (Option A — to be filled in next iteration if approved)

Files that would change (no code yet — proposal stage):

- `sgl-kernel/csrc/`: add W4-FP8 mixed-input kernel (port Machete or custom CUTLASS 3.x mixed-input collective for SM120).
- `sgl-kernel/cmake/`: add SM120 W4A8-FP8 tile table; reuse existing CUTLASS SM120 FP8 build flags.
- `python/sglang/srt/layers/quantization/gptq.py`: add `_soar_maybe_setup_w4a8_fp8_real` helper that:
  - Verifies layer is INT4 GPTQ (skip INT8 sparse_qkv via existing bit-width guard).
  - **Keeps INT4 weight storage intact** (repack to Machete format if required by kernel; no upcast to FP8 storage — that was the iteration_001 bug).
  - Adds per-token FP8 e4m3 activation quantizer (online, in `apply()`).
  - Calls new `machete_gemm_w4a8_fp8(...)` (or equivalent) from sgl-kernel.
- `python/sglang/srt/models/minicpm.py`: reuse existing `_soar_w4a8_eligible` tags from CHANGE_W4A8_001 (no MiniCPM model changes needed).
- `benchmark/soar/demo_sala/prepare_env.sh`: add new env flag `SOAR_W4A8_FP8_REAL_GEMM` (distinct from the now-deprecated `SOAR_W4A8_FP8_GEMM` which gated the W8A8 mislabel path).

## Validation commands (planned)

Same workflow as iteration 001:

```bash
# CPU unit test for FP8 activation quantizer + dequant utilities
python3 test/srt/quantization/test_w4a8_fp8_real_quantizer.py

# fcloud
python3 scripts/fcloud/fcloud_workflow.py setup
sed -i 's/SOAR_W4A8_FP8_REAL_GEMM:-0/SOAR_W4A8_FP8_REAL_GEMM:-1/' /root/submission_sim/prepare_env.sh
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

W4A8-FP8-real path will be env-gated `SOAR_W4A8_FP8_REAL_GEMM=0` by default. To disable:
```bash
sed -i 's/SOAR_W4A8_FP8_REAL_GEMM:-1/SOAR_W4A8_FP8_REAL_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```
Code revert: `git revert <iteration-2-commit-sha>`.

## Open questions awaiting user decision

1. **Approve Option A** (W4-FP8 mixed-input GEMM, FP8 QMMA at guaranteed 296 TF on SM120) as the first real W4A8 attempt? **Yes / No**
2. If yes, are there constraints on **kernel source**: (a) port vllm Machete to SM120, (b) write a custom CUTLASS 3.x mixed-input collective from scratch, or (c) acceptable to vendor an Apache-2.0 kernel from a third-party repo?
3. Should we run **Phase 0 INT8 IMMA microbenchmark** on SM120 first (cheap, <1h fcloud time) to keep Option B (W4-INT8 Marlin) as a real fallback? Or skip Phase 0 and commit fully to FP8?
4. Estimated time/effort budget for Option A bring-up before escalating to Option C (full custom CUTLASS)?

This proposal makes **no code changes** until you approve.

---

**Companion ZH document**: `PROPOSAL_W4A8_REAL_001.zh.md`.
