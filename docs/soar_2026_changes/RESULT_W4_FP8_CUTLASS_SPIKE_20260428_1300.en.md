# RESULT — W4-FP8 CUTLASS spike on SM120

**Date**: 2026-04-28 13:00
**Status**: Executed on fcloud (RTX 6000D sm_120, CUDA 12.8, torch 2.9.1+cu128). **Verdict: RED.**
**Predecessor**: [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)
**Code**: [spike_w4fp8/bench_w4fp8_sm120.cu](spike_w4fp8/bench_w4fp8_sm120.cu) + [run_w4fp8_spike.sh](spike_w4fp8/run_w4fp8_spike.sh)

## What this measures

A hand-rolled CUDA kernel that, in the inner K-loop, performs the W4 → FP8 dequant chain (unpack int4 → subtract zero-point → multiply bf16 group scale → cast bf16x2 to FP8 e4m3x2 via PTX cvt) and then issues `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. This isolates the **dequant tax** that a real W4-FP8 kernel would pay on top of the dense FP8 ceiling (281 TF measured in Phase 0).

The kernel is intentionally minimal — no TMA, no swizzle, no software pipelining — so the measured TFLOPS is a **lower bound** on what a tuned implementation could achieve. If even this scaffold hits ≥250 TF, a real production kernel is viable; if it can't crack 180 TF, the dequant chain is the bottleneck and we should kill the W4-FP8 path.

## Builds

| Step | Command |
|---|---|
| Upload | scp `bench_w4fp8_sm120.cu` and `run_w4fp8_spike.sh` to `/root/` on fcloud |
| Build+run | `bash /root/run_w4fp8_spike.sh` |

Expected build flags: `nvcc -arch=sm_120 -O3 -std=c++17 -Xptxas -v`. Verify nvcc supports `sm_120` — needs CUDA toolkit 12.8 or newer.

## Pass/fail criteria (from proposal §6)

| Measured TFLOPS at M=16384,N=14336,K=4096 | Verdict |
|---|---|
| **≥ 250 TF** (≥ 89% of dense FP8 281 TF) | **GREEN** — pursue full W4-FP8 kernel |
| **180–249 TF** | **YELLOW** — marginal, defer behind NVFP4 KV |
| **< 180 TF** | **RED** — kill W4-FP8 dense permanently |

## Result table

| Shape | Avg ms/iter | TFLOPS | % of 281 TF FP8 ceiling | Verdict |
|---|---|---|---|---|
| M=16384 N=14336 K=4096 (proposal default) | 12.304 | **156.4** | **55.7%** | **RED** |
| M=2048 N=4096 K=4096 (small) | 0.441 | 155.8 | 55.4% | RED |
| M=1 N=14336 K=4096 (decode shape) | 0.004 | 32.6 | 11.6% | RED (BW-bound, expected) |
| M=8192 N=8192 K=8192 (square medium) | 7.032 | 156.4 | 55.6% | RED |

**Build info**: 23 registers, 0 spills, 1 barrier (ptxas -v). All shapes converge to ~156 TF — i.e. compute-peak limited, not launch-overhead and not bandwidth (except the decode shape).

## Why RED — root-cause attribution

The scaffold's inner K-loop performs:

1. Load 32 packed int4 weights from smem (1 LDS.32 per 8 weights)
2. Unpack 8 int4 → 8 fp32 (manual shift + sub-zp + fmul scale, 8 IADDs + 8 FMULs)
3. Pack 8 fp32 → 4 e4m3 pairs via 4× `cvt.rn.satfinite.e4m3x2.f32` PTX instructions
4. One `mma.sync.aligned.m16n8k32` (4096 FMAs / warp / iter)

The MMA alone, if it ran in isolation at 281 TF, would take **~7 ms** for the 16384·14336·4096 problem. Measured time is 12.3 ms, so dequant overhead is **~5.3 ms = 43% of cycles**.

## Caveats — this is a lower bound

1. **Unfused dequant**: production kernels use `lop3.b32` to unpack 8 int4 + bias + cast to fp16 in **3 PTX instructions** (vs. our ~16). A Marlin-class fused unpack would cut dequant cost by ~3-4×, recovering most of the 43% gap.
2. **No TMA / no async copies**: smem load is on the critical path; a real kernel uses TMA + 2-3-stage software pipelining to hide it under MMA.
3. **Synthetic smem (lane-indexed reuse)**: this **maximizes** MMA utilization, so the 156 TF ceiling is essentially "with-dequant compute peak" — a more realistic kernel would also pay L2/HBM bandwidth tax, but a tuned one would hide it.
4. **PTX cvt path**: we use `cvt.rn.satfinite.e4m3x2.f32` (2 fp32 → 1 fp8x2). The bf16-direct path doesn't exist on SM120, so this is the supported route. Cost ~1 instr per 2 values.

**Realistic upper bound for a tuned production W4-FP8 kernel on SM120**: ~200-240 TFLOPS (70-85% of 281 TF), based on Marlin-W4A16 typically achieving 75-85% of peak BF16 on Hopper/Ada.

## Caveats / honesty notes

1. **Synthetic data**: kernel reads garbage from gmem and accumulates garbage. The MMA throughput is real, but the result is uncorrelated with anything. Cannot validate correctness from this spike — only timing.
2. **No TMA / no swizzle**: a tuned kernel would close maybe 10–20% of any gap to FP8 peak vs this scaffold.
3. **Single tile per warp**: the scaffold runs a fixed K_LOOP_ITERS=64 inner loop on shared smem, with grid sized so total FMAs ≈ M·N·K. This is a fair-cost proxy for a real GEMM but doesn't model SMEM-bandwidth or L2-bandwidth pressure that a full GEMM would feel.
4. **dequant chain**: this is the **most expensive** version — bf16 multiply then bf16→FP8 cast. A more aggressive impl could fuse zero-point absorption into the scale (eliminate the subtract), or use lop3.b32 tricks for parallel int4 unpack. So if the spike hits YELLOW, a tuned kernel could plausibly clear GREEN.

## Decision (post-result)

**Verdict: RED → park W4-FP8 dense kernel work indefinitely.**

Rationale:
- Lower-bound spike at 55.7% of FP8 ceiling means a tuned kernel could plausibly reach 200-240 TF (~75-85%), translating to **~+15-40% on prefill GEMMs only** vs. current W4A16 Marlin (~140-170 TF-equiv).
- Decode is bandwidth-bound (the 32 TF result reflects HBM, not compute) — **W4-FP8 gives ~zero gain at decode**, which is most of S₁ and S₈.
- End-to-end estimated gain: **~5-12% on prefill-heavy paths only**, in exchange for **3-4 weeks of CUTLASS-class kernel engineering** plus calibration risk on accuracy.
- Compare to NVFP4 KV cache (next priority): ~3 days of plumbing (80% in tree per survey), saves ~44% KV memory, helps S∞ directly via larger batches at long context. Much higher ROI.

**Action items**:
1. ✅ Update [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) — mark W4-FP8 dense as RED/parked.
2. ✅ Proceed with NVFP4 KV cache P2 plumbing (`--force-dense-minicpm` smoke first per [SURVEY_NVFP4_KV_P1_20260428_1130.en.md](SURVEY_NVFP4_KV_P1_20260428_1130.en.md)).
3. Revisit W4-FP8 only if a vendor (NVIDIA / CUTLASS upstream) publishes a tuned SM120 W4-FP8 kernel we can drop in (zero engineering cost).

## Cross-references

- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md) — 281 TF FP8 ceiling reference
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md) — full kernel cost analysis (3–4 weeks)
