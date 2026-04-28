# Analysis — Can we reuse the existing QServe W4A8 INT8 kernel for W4A8 FP8?

**Date**: 2026-04-27 17:30
**Context**: User asked whether the existing W4A8 INT8 kernels in `sgl-kernel/csrc/gemm/qserve_w4a8_per_*_gemm.cu` can be turned into W4A8 FP8 by simply replacing INT8 with FP8.
**Status**: Analysis only. No code change.

## TL;DR

**No, it is not a simple substitution.** The MMA opcode change is one line, but the W4 dequant inner loop — the dominant cost of the kernel — has to be rewritten end-to-end because QServe specifically chose integer arithmetic so the dequant feeds INT8 IMMA without leaving the integer pipeline. Estimated effort: **~3–4 weeks** of CUDA work (single experienced engineer, optimistic case).

## 1. What the existing INT8 kernel does

[sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu](sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu) — QServe-style dense W4 weight × A8 activation GEMM, SM80+:

- MMA: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (line 136)
- Accumulator: `int32_t`
- W4 dequant: pack-4-int4 → mask `0x0F0F0F0F` / `0xF0F0F0F0>>4` → subtract `int8` zero → multiply by `int8` group scale → result stays as `int8` (lines 280–285)
- Group scales / zeros: `int8_t scales_i8`, `int8_t zeros`
- Epilogue: `int32 → fp16` with per-token + per-channel scale multiply

Critical design: **the entire compute pipeline stays in integer arithmetic** so the dequantized weights can directly feed INT8 IMMA. This is QServe's "two-level scaling" — group scales chosen so per-group max fits in int8 after subtracting zero, then a per-channel float scale absorbs the rest in the epilogue.

## 2. What changes for FP8 e4m3 path

| Stage | INT8 (current) | FP8 (needed) |
|---|---|---|
| MMA opcode | `s32.s8.s8.s32` | `f32.e4m3.e4m3.f32` |
| Accumulator | `int32_t` | `float` |
| **W4 dequant inner loop** | int4 → int8 by `(w − z) * s` integer | int4 → fp16 → multiply fp16 group scale → **encode FP8 e4m3 bit pattern** (sign + 4-bit exp bias 7 + 3-bit mantissa, with saturation/clipping) |
| Group scales storage | int8 | fp16 / bf16 |
| Epilogue | int32 → fp16 with scale | fp32 → bf16 with scale (mostly compatible) |

The MMA line is trivial. The **dequant rewrite** is non-trivial because:
- Integer subtract+multiply collapses into 2 ops; FP8 encode requires bit-manipulation to build e4m3 fields with proper exponent biasing and saturation.
- Register pressure changes (fp16 intermediates instead of int8 register packing).
- Group scales become fp16 — load/broadcast paths differ.

## 3. Other necessary changes

1. **Architecture target**: current `__CUDA_ARCH__ >= 800` (Ampere). FP8 QMMA `m16n8k32.f32.e4m3.e4m3.f32` requires **sm_89 / 90+ / 120** and CUDA 12.0+. Need a separate file (or `#if` guard) — cannot reuse one source for both.
2. **SM120 tile retuning**: CTA/WARP/STAGES were tuned for INT8 IMMA on A100/H100. Register pressure differs for FP8. New tuning sweep on RTX 6000D required.
3. **Activation FP8 e4m3 per-token quantizer**: `sgl_per_token_quant_fp8` already exists in sgl-kernel. Drop-in.
4. **Loader path**: GPTQ checkpoint already has fp16 group scales — actually easier than synthesizing int8 scales.
5. **Bindings + Python wiring**: new op in [sgl-kernel/csrc/common_extension.cc](sgl-kernel/csrc/common_extension.cc), Python wrapper, sglang quant linear method (mirror `gptq_marlin`).
6. **Numerics validation**: FP8 e4m3 mantissa = 3 bits. Outliers matter. Need accuracy sweep on `perf_public_set.jsonl` to stay > 99% normalized for C=1.0.

## 4. Implementation work breakdown

| # | Task | Days |
|---|---|---|
| 1 | Fork `qserve_w4a8_per_group_gemm.cu` → new `qserve_w4a8fp8_per_group_gemm.cu` | 0.5 |
| 2 | Rewrite W4-int4 → FP8 e4m3 dequant inner loop (the hard part) | 4–6 |
| 3 | Swap MMA opcode + fp32 accumulator + epilogue | 1 |
| 4 | Wire activation FP8 quantizer | 0.5 |
| 5 | Loader: keep INT4 weights, fp16 group scales, FP8 path | 1 |
| 6 | sgl-kernel binding + Python op registration | 0.5 |
| 7 | sglang linear method (mirror `gptq_marlin`) | 1 |
| 8 | SM120 tile/stage tuning sweep | 2–3 |
| 9 | Numerical validation vs W4A16 baseline (≥ 99% normalized) | 2 |
| 10 | S1/S8/Smax benchmarks + iteration | 2 |
| | **Optimistic total** | **14–18** |

Realistic with debug iterations: **3–4 weeks**.

## 5. Expected gain

- BF16 = 142 TF measured; FP8 = 281 TF measured (Phase 0) → **~2× compute peak**
- Decode is **weight-bandwidth-bound** (W4 packed storage is identical to W4A16). S1 sees **+5–15%** only.
- Prefill / large-batch is **compute-bound** → **+30–60%** at high concurrency / long context.
- Net official-eval gain estimate (S1 40% + S8 30% + S∞ 30%): **+8–15% performance score** if accuracy holds.

## 6. Cost vs alternatives

| Dimension | W4-FP8 GEMM (this analysis) | NVFP4 KV cache | Other catalog items |
|---|---|---|---|
| Engineering effort | 3–4 weeks new CUDA kernel | ~1 week (storage + quant; reuse path) | varies |
| Novelty risk | Untested on SM120; no reference impl | Champion combo uses it; refs exist | varies |
| Accuracy risk | High (FP8 e4m3 3-bit mantissa) | Medium (KV sensitivity) | varies |
| Compute lever | +2× FP8 TFLOPS in MMA | None — KV dequant to BF16 | varies |
| Memory lever | Activations 2× smaller (vs BF16) — small decode win | KV 2× smaller vs FP8 (4× vs BF16) — **big win at long context** | varies |
| Champion-combo evidence | None | Yes | n/a |

## 7. Clarifications added 2026-04-28

### 7.1 What "W4A8 #1" actually was vs the v18 baseline

A prior test labeled "W4A8 #1" (commit `7ce21c3f5`) was actually **W8A8 FP8**, not INT8 and not real W4A8. Process flow:

| Stage | v18 baseline (W4A16 BF16) | W4A8 #1 mislabel (W8A8 FP8) | Real W4A8 FP8 (parked Option A) |
|---|---|---|---|
| Weight HBM dtype | INT4 packed | **FP8 e4m3 (inflated 2×)** | INT4 packed |
| Dequant in K-loop | INT4 → BF16 | none | INT4 → FP8 e4m3 |
| Activation | BF16 (no quant) | BF16 → FP8 e4m3 per-token | BF16 → FP8 e4m3 per-token |
| MMA | BF16×BF16 → FP32, 148 TF | FP8×FP8 → FP32, 281 TF peak | FP8×FP8 → FP32, 281 TF peak |
| Weight bytes/param | 0.5 B | 1.0 B | 0.5 B |
| Result | Reference | **+118%/+56%/+30% regression** (decode is weight-BW-bound; weight bytes doubled) | Hypothesis: best of both worlds |

The regression of "W4A8 #1" is consistent with doubling weight HBM bytes on a bandwidth-bound decode workload. It does NOT invalidate real W4A8.

We never tested **W8A8 INT8** end-to-end on the model. INT8 only appeared as the synthetic Phase 0 microbench (136 TF, killed before any model-level test).

### 7.2 Phase 0 vs CUTLASS spike (Proposal B)

| Test | Inputs at MMA | Measures | Number |
|---|---|---|---|
| Phase 0 (done) | FP8 × FP8 (no dequant in K-loop) | Hardware FP8 ceiling | 281 TF |
| Proposal B (proposed) | INT4 packed → in-kernel dequant → FP8 × FP8 | FP8 ceiling **after** W4→FP8 dequant overhead | TBD |

Proposal B measures the dequant tax that Phase 0 deliberately did not include.

## 8. Recommendation

1. **Park W4-FP8 dense GEMM** as Option A. Worth revisiting only after NVFP4 KV lands.
2. **Pursue NVFP4 KV cache as next iteration** — see `PROPOSAL_NVFP4_KV_CACHE_20260427_1730.{en,zh}.md`.
3. Optionally, run a **cheap CUTLASS W4-FP8 spike** to validate the analytical FP8 ceiling before committing 3–4 weeks — see `PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.{en,zh}.md`.

## Cross-references

- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md)
- [RESEARCH_w4a8_kernel_landscape_20260427_1530.en.md](RESEARCH_w4a8_kernel_landscape_20260427_1530.en.md)
- [PROPOSAL_W4A8_REAL_001.en.md](PROPOSAL_W4A8_REAL_001.en.md)
- [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)
- [sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu](../../sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu)
- [sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu](../../sgl-kernel/csrc/gemm/qserve_w4a8_per_chn_gemm.cu)
