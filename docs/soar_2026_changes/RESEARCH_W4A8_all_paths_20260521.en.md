# RESEARCH: W4A8 Fused GEMM — All Paths Compared (excluding two-step)

**Date**: 2026-05-21
**Scope**: Exhaustive survey of every viable direction for a fused INT4→FP8
dequant + GEMM kernel on SM120 (Blackwell) at >148 TFLOPS.
Two-step approaches are excluded (previously proven +118% regression).

---

## TL;DR Summary

| # | Path | TFLOPS | Effort | Risk | Viable? |
|---|------|--------|--------|------|---------|
| A | **Create SM120 mixed-input cutlass builder** | ~280 | 2-3 weeks | Medium | ✅ BEST |
| B | **Fix raw PTX kernel (current w4a8_fp8_qmma.cu)** | ~280 | 1-2 weeks | HIGH | ⚠️ |
| C | **Port SM100 UMMA mixed-input → SM120 tcgen05** | ~280 | 1-2 weeks | Medium | ✅ |
| D | **CuTeDSL Python codegen → C++ kernel** | ~280 | 1 week | HIGH | ⚠️ |
| E | **Optimize wmma (no tcgen05)** | ~190 max | 3-5 days | Low | ❌ |
| F | **INT8 dequant + INT8 IMMA** | ~136 | 3-5 days | Low | ❌ |

---

## A: Create SM120 Mixed-Input Cutlass Builder (BEST)

### What exists
- NVIDIA cutlass has `sm120_blockscaled_mma_builder.inl` (305 lines) — FP8×FP8 with block scaling
- NVIDIA cutlass has `sm120_mma_builder.inl` — standard FP16/BF16/TF32
- sgl-kernel has `sm90_gmma_builder_mixed_input.inl` (280 lines) — SM90 mixed-input reference
- sgl-kernel has `CollectiveBuilderMixedInput` stub (48 lines) that delegates to SM90 builder

### What's needed
Create a new file `sm120_mixed_input_mma_builder.inl` (~350-400 lines) that:
1. Accepts `ElementPairB = cute::tuple<QuantType, ScaleType, ZeroType>` (INT4 + scales + zeros)
2. Detects narrow operand (4-bit < 8-bit) and routes through dequant transform
3. Uses `rr_blockscaled_op_selector_sm120()` for tcgen05 QMMA instruction selection
4. Adapts pipeline stage count for mixed-input SMEM overhead
5. Uses existing SM120 epilogue (no changes needed)

### Key advantage
- Leverages ALL existing cutlass infrastructure: TMA, TMEM, tcgen05.mma, warp specialization, software pipelining
- The SM120 blockscaled builder already handles FP8×FP8 QMMA correctly
- Only need to inject INT4→FP8 dequant into the weight loading path
- No raw PTX — cutlass generates correct PTX automatically

### Reference pattern (from SM90 mixed-input builder)
```cpp
// Detect narrow operand
static constexpr bool IsSubbyteA = cute::sizeof_bits_v<ElementA> < 8;
using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, ElementA>;
// Store dequant scales alongside weight data
using ElementPairB = cute::tuple<QuantType, ScaleType, ZeroType>;
// MMA uses the dequantized type (FP8)
using ElementBMma = float_e4m3_t;
```

### SMEM budget (SM120: 101KB)
| Buffer | Size | Note |
|--------|------|------|
| Weight (INT4 packed) | 8 KB | 128×128 × 0.5 B/elem |
| Weight scale + zero | ~8 KB | Per-group scales |
| Dequantized FP8 weight | 16 KB | May be eliminated if dequant feeds MMA directly |
| Activation FP8 | 16 KB | TMA-loaded |
| TMA descriptors | ~512 B | 4 descriptors |
| Pipeline buffers | ~32 KB | Double-buffered |
| Epilogue | ~16 KB | Output staging |
| **Total** | **~96 KB** | Fits in 101KB |

---

## B: Fix Raw PTX Kernel (`w4a8_fp8_qmma.cu`)

### Current state
- Written (440 lines, committed)
- Fails ptxas compilation: `tcgen05.alloc` syntax wrong, `elect_one.sync` syntax wrong
- No SMEM swizzling, no double-buffering, no TMA for global→SMEM

### What needs fixing
1. **TMEM allocation**: Must use `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_ptr], num_cols` — writes to SMEM, not registers
2. **elect_one.sync**: Syntax is `elect_one.sync p, 0xFFFFFFFF;` (comma, not pipe)
3. **TMA descriptors**: The `fill_tma_desc_2d()` format is unvalidated — high risk of HW mismatch
4. **Warp specialization**: cutlass uses 6-7 specialized warps; our single-warpgroup may not meet HW requirements
5. **No swizzling**: bank conflicts on SMEM reads by tcgen05

### Verdict
**Viable but risky.** Each PTX syntax error requires a compile-test cycle on fcloud (~5 min each). The TMA descriptor format alone could take days to get right. If any one of these is wrong, the kernel produces garbage output with no diagnostics.

---

## C: Port SM100 UMMA Mixed-Input → SM120 tcgen05

### What exists
NVIDIA cutlass has `sm100_mixed_input_umma_builder.inl` (~350 lines) — a FULL mixed-input builder for SM100 (Blackwell UMMA instructions). This handles:
- INT4/FP8/FP16 mixed-width operands
- Scale + zero-point dequant in the mainloop
- TMA-based data loading
- Warp-specialized scheduling

### Key difference: SM100 UMMA vs SM120 tcgen05
- SM100 UMMA: Uses `SM100_MMA_F8F6F4_SS` or `SM100_MMA_F16BF16_TS` etc.
- SM120 tcgen05: Uses different instruction selection via `rr_op_selector_sm120()`
- Both are Blackwell! The UMMA builder might actually work on SM120 with minor changes.
- The SM120 blockscaled builder uses `OpClassBlockScaledTensorOp` while mixed-input uses `OpClassTensorOp`

### Verdict
**Very promising.** The SM100 mixed-input builder already handles all the INT4→FP8 dequant logic. Porting to SM120 mainly means changing the MMA instruction selector from UMMA to tcgen05, and adapting pipeline stages for SM120's smaller SMEM.

---

## D: CuTeDSL Python → C++ Code Generation

### What exists
NVIDIA cutlass has Python CuTeDSL examples for SM120 mixed-input GEMM:
`examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/`
- `mixed_input_gemm.py` — basic INT4×FP8 with tcgen05
- `grouped_mixed_input_gemm.py` — grouped variant
- `grouped_mixed_input_gemm_acc_scale.py` — accumulated scales variant

### How it works
These Python scripts use the CuTe DSL (Domain Specific Language) to describe the kernel at a high level. The framework then generates C++ CUDA code that compiles with cutlass. It handles TMA, TMEM, tcgen05.mma, warp specialization automatically.

### Feasibility for our use case
- **Pro**: Would generate correct PTX automatically — no syntax errors
- **Con**: Requires the CuTeDSL Python environment (cutlass build deps, MLIR, etc.)
- **Con**: Generated code is tightly coupled to the cutlass Python framework
- **Con**: Difficult to integrate into sglang's cmake build system
- **Unknown**: Whether the generated code can be compiled as a standalone .so

### Verdict
**Too complex for our build pipeline.** Would require pulling the entire NVIDIA cutlass Python ecosystem as a build dependency.

---

## E: Optimize wmma Kernel (No tcgen05)

### Current wmma kernel (`w4a8_fp8_fused_gemm.cu`)
- 128 threads, 4 warps, m16n16k16 wmma
- INT4→FP16 dequant in SMEM
- 148 TFLOPS theoretical, ~100-120 TFLOPS actual

### Possible optimizations (without tcgen05)
| Optimization | Gain | Effort |
|---|---|---|
| Double-buffering SMEM (K-tile overlap) | +10-20% | Medium |
| `cp.async` for global→SMEM | +5-10% | Medium |
| Larger tiles (reduce launch overhead) | +5% | Low |
| Better occupancy | +0-5% | Low |
| **Combined ceiling** | **~190 TFLOPS** | — |

### Why this is NOT enough
- Baseline Marlin: 148 TFLOPS theoretical
- Optimized wmma: ~190 TFLOPS max
- tcgen05 QMMA: ~280 TFLOPS (measured)
- **Gap to tcgen05: ~90 TFLOPS (47% more throughput)**
- The entire point of W4A8 is to exploit FP8 QMMA for 2× speedup

---

## F: INT8 Dequant + INT8 IMMA

### Measured INT8 throughput on SM120
From `PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md`:
- INT8 IMMA: ~136 TFLOPS (same as BF16)
- FP8 QMMA: ~275 TFLOPS (2× INT8)

### Why not
INT8 tcgen05 on SM120 caps at the same ceiling as BF16. FP8 QMMA is the only path to >200 TFLOPS on this hardware.

---

## Implementation Plan: Path A (Recommended)

### Step 1: Reference study (1 day)
- Read NVIDIA cutlass `sm100_mixed_input_umma_builder.inl` thoroughly
- Read sgl-kernel `sm90_gmma_builder_mixed_input.inl` 
- Understand how INT4 dequant is injected into the mainloop

### Step 2: Create SM120 mixed-input builder (3-5 days)
- New file: `sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm120_mixed_input_mma_builder.inl`
- Follow SM90 mixed-input pattern for operand detection and dequant setup
- Follow SM120 blockscaled builder for tcgen05 instruction selection
- Adapt SMEM budgets (SM120 has 101KB vs SM90's 232KB)

### Step 3: Wiring (1 day)
- Add to `collective_builder_mixed_input.hpp`: `#include` the new SM120 builder
- Add new kernel file similar to `fp8_blockwise_gemm_kernel.cu` but for W4A8
- Register torch op in `common_extension.cc`

### Step 4: Build & test (2-3 days)
- Full sgl-kernel wheel build on fcloud (needs cutlass FetchContent)
- Correctness: single tile → multi tile → model dimensions
- Speed: S1/S8/Smax benchmarks

### Estimated total: 7-10 days

---

## Fallback: Path C (Port SM100 UMMA)

If Path A's blockscaled op class creates complications (OpClassBlockScaledTensorOp vs OpClassTensorOp), fall back to porting the SM100 UMMA mixed-input builder. This uses `OpClassTensorOp` which is closer to the SM90 mixed-input pattern.

The SM100 builder already handles all dequant logic; only the instruction selector needs SM120 adaptation.

---

## References

- `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu` — raw PTX kernel (blocked)
- `sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu` — working wmma kernel
- `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` — working SM120 FP8 GEMM
- `sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl` — SM90 reference
- `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` — SM90 W4A8 grouped
- NVIDIA cutlass: `sm100_mixed_input_umma_builder.inl` — SM100 mixed-input
- NVIDIA cutlass: `sm120_blockscaled_mma_builder.inl` — SM120 blockscaled
- NVIDIA cutlass: `examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/` — Python DSL examples
