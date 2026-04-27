# Proposal — 1-day CUTLASS W4-FP8 spike to validate the FP8 ceiling on SM120

**Date**: 2026-04-27 17:30
**Status**: Proposal only. **No code changes yet.** Awaiting user approval.
**Predecessor**: `ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md` (full kernel = 3–4 weeks).

## 1. Objective and expected gain

**De-risk** the parked W4-FP8 dense GEMM Option A by spending 1 day to measure whether a CUTLASS-based W4-FP8 reference can actually approach the dense FP8 ceiling on SM120 (281 TF measured in Phase 0). If yes, we know the 3–4 week investment has a real upper bound. If no (e.g., dequant overhead caps it at <200 TF), the priority order is settled and we don't waste the 3–4 weeks.

**This is NOT a production kernel.** It's a measurement spike — a synthetic-data benchmark to find the FP8 ceiling for the W4-weight × FP8-activation shape we'd actually use.

## 2. Rule-compliance check

N/A — this is a one-day **measurement** task. No model change, no submission package change, no accuracy risk. Pure benchmark on synthetic data.

## 3. Risk

- Zero accuracy/stability risk (no production change).
- Risk: 1 fcloud-hour of compute spent on a spike that could yield "inconclusive".
- Mitigation: clear pass/fail thresholds defined upfront (see § 6).

## 4. Implementation plan

### 4.1 Approach
Use a CUTLASS 3.x example as the starting point. CUTLASS has:
- `examples/65_distributed_gemm` and `examples/55_hopper_int4_fp8_gemm/` (Hopper SM90 W4-FP8 reference — closest existing impl).
- For SM120: adapt the SM90 example by changing `arch::Sm90` → `arch::Sm120` and using SM120-compatible MMA atom (Blackwell warp-level MMA, NOT warpgroup).

### 4.2 Steps (~1 fcloud working day)

| # | Step | Time |
|---|---|---|
| 1 | Locate `examples/55_hopper_int4_fp8_gemm/` in CUTLASS submodule (or upstream clone) | 0.5 h |
| 2 | Make a self-contained `bench_w4fp8_sm120.cu` that calls the CUTLASS template at `M=16384, N=14336, K=4096` (matches Phase 0 mat shape) | 2 h |
| 3 | Build with `nvcc -arch=sm_120 -O3 -std=c++17 -lcudart` on fcloud | 1 h |
| 4 | If SM90 atom doesn't compile on SM120: fallback to a hand-rolled minimal W4→FP8 dequant kernel that calls `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` directly, with synthetic INT4 weight + FP8 activation | 2 h |
| 5 | Run 100 iters, compute median TFLOPS, log result | 0.5 h |
| 6 | Document outcome in `RESULT_W4_FP8_CUTLASS_SPIKE_<date>.{en,zh}.md` | 1 h |
| | **Total** | **~7 h, single fcloud session** |

### 4.3 Files (all ephemeral, no commit to sglang)
- `/root/bench_w4fp8_sm120.cu` (fcloud-only)
- `/root/run_w4fp8_spike.sh` (fcloud-only)
- Local doc: `docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_<date>.{en,zh}.md` (committed to repo)

## 5. Validation commands (fcloud)

```bash
# (User starts fcloud)
# Agent uploads bench_w4fp8_sm120.cu via fcloud_exec.py
ssh fcloud "cd /root && nvcc -arch=sm_120 -O3 -std=c++17 bench_w4fp8_sm120.cu -lcudart -o bench_w4fp8 && ./bench_w4fp8"
```

## 6. Pass/fail criteria

| Measured TFLOPS at M=16384,N=14336,K=4096 | Decision |
|---|---|
| **≥ 250 TF** (≥ 89% of dense FP8 281 TF) | **Green light**. W4-FP8 ceiling is real. Pursue full kernel as future iteration. |
| **180–249 TF** | **Yellow**. FP8 lever exists but ~30% of peak lost to dequant. Marginal vs NVFP4 KV — defer. |
| **< 180 TF** | **Red**. Dequant overhead dominates. **Kill** W4-FP8 dense permanently. |

## 7. Rollback

N/A — no production change.

## 8. Next-step suggestions

- If green: schedule W4-FP8 full kernel as iteration after NVFP4 KV.
- If yellow/red: remove W4-FP8 from the optimization roadmap; focus on NVFP4 KV + other catalog items.

## 9. Cross-references

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md)
- [PROPOSAL_NVFP4_KV_CACHE_20260427_1730.en.md](PROPOSAL_NVFP4_KV_CACHE_20260427_1730.en.md)
- CUTLASS upstream: https://github.com/NVIDIA/cutlass `examples/55_hopper_int4_fp8_gemm/`
