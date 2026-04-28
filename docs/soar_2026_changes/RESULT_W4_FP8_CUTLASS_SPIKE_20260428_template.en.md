# RESULT — W4-FP8 CUTLASS spike on SM120 (template, fill after fcloud run)

**Date**: TBD
**Status**: Awaiting fcloud execution.
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

## Result table (fill after run)

| Shape | Avg ms/iter | TFLOPS | % of 281 TF FP8 ceiling | Notes |
|---|---|---|---|---|
| M=16384 N=14336 K=4096 (proposal default) | TBD | TBD | TBD% | primary verdict |
| M=2048 N=4096 K=4096 (small) | TBD | TBD | TBD% | smaller tile, may show launch overhead |
| M=1 N=14336 K=4096 (decode shape) | TBD | TBD | N/A | bandwidth-bound; FLOPS not meaningful |
| M=8192 N=8192 K=8192 (square medium) | TBD | TBD | TBD% | check K-scaling |

## Caveats / honesty notes

1. **Synthetic data**: kernel reads garbage from gmem and accumulates garbage. The MMA throughput is real, but the result is uncorrelated with anything. Cannot validate correctness from this spike — only timing.
2. **No TMA / no swizzle**: a tuned kernel would close maybe 10–20% of any gap to FP8 peak vs this scaffold.
3. **Single tile per warp**: the scaffold runs a fixed K_LOOP_ITERS=64 inner loop on shared smem, with grid sized so total FMAs ≈ M·N·K. This is a fair-cost proxy for a real GEMM but doesn't model SMEM-bandwidth or L2-bandwidth pressure that a full GEMM would feel.
4. **dequant chain**: this is the **most expensive** version — bf16 multiply then bf16→FP8 cast. A more aggressive impl could fuse zero-point absorption into the scale (eliminate the subtract), or use lop3.b32 tricks for parallel int4 unpack. So if the spike hits YELLOW, a tuned kernel could plausibly clear GREEN.

## Next-step decision tree

- **GREEN** → schedule W4-FP8 full-kernel iteration AFTER NVFP4 KV cache lands in production.
- **YELLOW** → defer W4-FP8 indefinitely; revisit only if NVFP4 KV doesn't deliver expected gain. Document YELLOW reason in [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).
- **RED** → permanently remove W4-FP8 dense from the optimization roadmap. Update catalog.

## Cross-references

- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md](PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md) — 281 TF FP8 ceiling reference
- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md) — full kernel cost analysis (3–4 weeks)
