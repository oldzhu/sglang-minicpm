# DECISION: Option B vs Option C — Full Deep Comparison
## SM120 FP8 GEMM Path Analysis for MiniCPM-SALA GPTQ

**Date**: 2026-04-21  
**Status**: Option B selected for immediate work; Option C documented for future deep optimization  
**Baseline**: GPTQ (sparse_qkv_w8) + FP8 KV cache + dense mode — S1=121.71s, S8=44.09s, Smax=35.86s  
**Reference Hardware**: SM120 RTX PRO 6000 — 593 TFLOPS FP8, 296 TFLOPS BF16, 1398 GB/s

---

## Background: Why We Are Here

Profiling (CHANGE_0120) revealed:
- **GEMM = 85.3% of prefill time, 63.5% of decode time**
- Current Marlin GPTQ W4 kernel uses SM80 `mma.sync.aligned.m16n8k16` — cannot access SM120 FP8/FP4 TFLOPs
- SM120 FP8 hardware offers **2× more TFLOPS** than BF16, **4× more** than INT8-emulated paths
- Key bottleneck: compute-bound prefill is leaving 400+ TFLOPS of SM120 hardware idle

Phase 1 investigation (PROPOSAL_fp8_w4_dequant_gemm.md) surveyed TRT-LLM and confirmed the CUTLASS path is viable. Decision matrix (DECISION_fp8_w4_implementation_options.md) rejected Option A (TRT-LLM wheel, too large) and selected Options B and C.

### Critical Discovery: SM120 FP8 GEMM Already Exists in sgl-kernel

During Option B investigation, we found that **sgl-kernel already has a fully working SM120 FP8 blockwise GEMM**:

```
File: sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu
Function: sm120_fp8_blockwise_dispatch_shape<OutType>(out, a, b, scales_a, scales_b)
Guard: #if defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
Python API: fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype)
```

The SM120 kernel uses CUTLASS 3.x with:
- `cutlass::arch::Sm120` arch tag
- `MmaTileShape = Shape<128, 128, 128>` (uses SM120 UMMA warp-level MMA)
- `ScalesPerTile = Shape<128, 1, 1>` → one scale per row of A (per-128-K-block), per 128×128 block of B
- **Scale format for A**: `(M, K/128)` — per-row, per-K-block-of-128
- **Scale format for B**: `(K/128, N/128)` — per-K-block, per-N-block

This means Option B requires **zero CUDA kernel writing** — only weight conversion and model loading changes.

---

## The Two Paths Compared

### Option B: FP8 Blockwise GEMM via Existing SM120 Kernel

**Core Idea**:
1. `preprocess_model.py`: load GPTQ W4 model → dequantize W4 to FP16/BF16 → requantize to FP8 blockwise format (M×128-K-blocks, 128×128-N-blocks)
2. Store FP8 weights + block scales as new model weights
3. `linear.py` (model code): detect FP8 weights → quantize BF16 activation to FP8 (per-row, per-K-block) → call `fp8_blockwise_scaled_mm` → outputs BF16
4. No changes to sgl-kernel required — kernel already compiled for SM120

**What changes**:
| Component | Change |
|-----------|--------|
| `preprocess_model.py` | Add `gptq_to_fp8_blockwise()` conversion for linear layers |
| `python/sglang/srt/layers/linear.py` | Add FP8 blockwise dispatch path (`MiniCPMFP8Linear`) |
| `python/sglang/srt/models/minicpm.py` | Use `MiniCPMFP8Linear` when FP8 weights detected |
| `benchmark/soar/demo_sala/prepare_env.sh` | Add server args for FP8 linear path |
| No CUDA files changed | — |

**Scale conversion (GPTQ group_size=128 → blockwise 128×128)**:
- GPTQ weight scales: shape `(K/128, N)` (per-group per-column)
- Kernel needs: `scales_b` shape `(K/128, N/128)` (per-K-block per-N-block)
- Conversion: for each N-block of 128 columns within each K-group:
  - Dequantize the 128×128 W4 block to FP16
  - Find max absolute value → compute FP8 scale = max_abs / 448.0
  - Quantize to FP8 E4M3 = block / scale
- Memory: W4 (0.5B/param) → FP8 (1B/param) = 2× weight memory, but ~9B params → ~9GB (fits 84GB)
- Activation quantization: per-row, per-128-K-elements → scale = max(abs(x_row_block)) / 448.0

**Why blockwise (128×128) instead of per-tensor or per-token**:
- FP8 E4M3 has limited dynamic range (~±448)
- Per-tensor: single scale clips large variance → severe accuracy loss
- Per-token for A × per-column for B: needs (M) + (N) scales → but SM120 UMMA needs aligned block scales
- 128×128 blockwise: aligns with UMMA tile size, good accuracy/speed tradeoff

**Expected performance**:
| Metric | Current (Marlin W4) | After Option B | Gain |
|--------|---------------------|----------------|------|
| TFLOPS utilized | ~100-140 BF16 effective | ~350-450 FP8 | 2.5-3.5× |
| Prefill GEMM time | baseline | 25-35% reduction | ↑ |
| End-to-end S1 | 121.71s | ~85-100s est. | ↑ |
| End-to-end S8 | 44.09s | ~30-35s est. | ↑ |
| Weight memory | ~4.5GB (W4 9B model) | ~9GB (FP8 9B model) | 2× more |

Note: decode gain limited by memory bandwidth (1398 GB/s ceiling). Prefill is compute-bound and benefits most.

**Timeline**: 3-5 days  
**Risk**: Low-Medium (kernel is battle-tested; weight conversion + accuracy need validation)

---

### Option C: Custom SM120 W4A8 Fused Dequant GEMM Kernel

**Core Idea**:
Write a new CUDA kernel that:
1. Loads W4 GPTQ weights directly from memory (NOT dequantized — keeps memory footprint at ~4.5GB)
2. In-register: dequantize W4 to FP8 using GPTQ group scales
3. Use SM120 UMMA `tcgen05.mma.ws.sync` instruction with FP8 operands
4. Accumulate in FP32, output in BF16

This retains W4 memory footprint while gaining SM120 FP8 compute throughput.

**Why this is hard**:
- Must write custom CuTe mainloop that loads W4 and produces FP8 for UMMA
- GPTQ weight packing: 8 × INT4 values packed per INT32, with per-group dequantization
- SM120 UMMA requires specific memory alignment and register layout (4-register groups per warp lane)
- Must implement TMA-based prefetch for W4 data to hide latency
- Must handle GPTQ group_size=128 scale lookup without bank conflicts
- Cannot use existing CUTLASS CollectiveBuilder for fused dequant (not a supported path)
- Requires full custom mainloop in CuTe assembly-level primitives

**What changes**:
| Component | Change |
|-----------|--------|
| `sgl-kernel/csrc/gemm/w4a8_sm120/` | NEW: ~2000-3000 lines of CUDA kernel |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_w4a8_gemm_kernel.cu` | Main kernel |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_w4a8_tma_prefetch.cuh` | TMA helpers |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_dequant_fp8.cuh` | Dequant register utils |
| `sgl-kernel/python/sgl_kernel/gemm.py` | Add `gptq_w4a8_fp8_mm()` binding |
| `python/sglang/srt/layers/linear.py` | Dispatch to new kernel |
| Full rebuild of sgl-kernel required | ~4 hours first-time, ~3 min incremental |

**Kernel design challenges**:

1. **W4 memory layout for TMA**: GPTQ stores weights as `(K/8, N)` packed INT32s (8 INT4 per INT32). TMA requires aligned, non-strided tensors. Need custom TMA descriptor for packed W4.

2. **In-register dequantization pipeline**:
   ```
   // For each 128×128 tile:
   // 1. TMA-load 128×16 packed INT32 tile (equivalent to 128×128 INT4 weights)
   // 2. Lookup group scale: scales[k_group, n] for all n in tile
   // 3. Unpack INT4 pairs, subtract zero_point, multiply by scale → FP8
   // 4. Stage FP8 in shared memory for UMMA
   // 5. Run tcgen05.mma.ws.sync FP8×FP8 → FP32
   ```

3. **UMMA register layout**: SM120 UMMA (warp-level MMA) needs specific register layout. CuTe abstractions help, but custom mainloop means manual layout management.

4. **Pipeline depth**: Need software pipelining of TMA load (W4) + dequant + UMMA to fully utilize SM120 L1 bandwidth. Poor pipelining = poor occupancy = poor performance.

5. **Accuracy**: FP8 dequantization path for W4 introduces additional quantization noise. Must verify normalized accuracy stays > 99% for C=1.0.

**Expected performance**:
| Metric | Current (Marlin W4) | After Option C | Gain |
|--------|---------------------|----------------|------|
| TFLOPS utilized | ~100-140 BF16 effective | ~400-500 FP8 | 3-4× |
| Prefill GEMM time | baseline | 30-45% reduction | ↑ |
| Weight memory | ~4.5GB W4 | ~4.5GB W4 (unchanged!) | same |
| End-to-end S1 | 121.71s | ~75-95s est. | ↑ |
| End-to-end S8 | 44.09s | ~28-33s est. | ↑ |

**Timeline**: 4-8 weeks (depending on SM120 expertise and debugging time)  
**Risk**: High

---

## Head-to-Head Comparison Table

| Dimension | Option B: FP8 Blockwise | Option C: W4A8 Custom Kernel |
|-----------|------------------------|------------------------------|
| **Kernel writing** | None (existing kernel) | ~2000-3000 lines CUDA |
| **Timeline** | 3-5 days | 4-8 weeks |
| **Weight memory** | 2× (W4 → FP8) | Same (stays W4) |
| **Prefill TFLOPS** | 350-450 FP8 | 400-500 FP8 |
| **Prefill speedup** | ~2.5-3.5× on GEMM | ~3-4× on GEMM |
| **Decode speedup** | <10% (BW-bound) | <10% (BW-bound) |
| **Accuracy risk** | Low (FP8 well-tested) | Medium (new dequant path) |
| **Build risk** | None | High (UMMA, TMA, pipeline) |
| **Debugging risk** | Low | High |
| **Correctness validation** | Easy (existing tests) | Needs numerical testing |
| **Dependency risk** | None | None (all in sgl-kernel) |
| **Submission size** | Slightly larger wheel | Same or slightly larger |
| **SOAR compliance** | ✅ | ✅ |

---

## Performance Analysis by Inference Stage

### S1 (single request, max-concurrent=1) — 40% weight

**Prefill phase** (dominant for long prompts):
- Compute-bound for large batch/sequence lengths
- Option B: ~2.5-3× speedup on GEMM component (85% of time) → 50-60% overall prefill reduction
- Option C: ~3-4× speedup on GEMM → 60-70% overall prefill reduction
- Δ (C vs B): +10-15% better prefill for C

**Decode phase** (dominant for short prompts / streaming):
- Memory-bandwidth bound: 1398 GB/s ceiling
- FP8 weights = 1 byte → 2× more data to load vs W4 (0.5 bytes)
- Option B: **slightly slower decode** than W4 (more bandwidth for weights)
- Option C: **same decode as current** (W4 stays, no extra bandwidth)
- Δ (C vs B): C is significantly better for decode

### S8 (8 concurrent requests) — 30% weight

**Prefill phase**: larger effective batch → more compute-bound
- Option B: ~55-65% prefill reduction
- Option C: ~65-75% prefill reduction

**Decode phase**: 8 requests × decode steps → bandwidth × 8
- Option B: each request's decode loads FP8 weights → still BW-bound but 2× more per step
- Option C: stays W4 → same BW as baseline

### S∞ (unlimited concurrency) — 30% weight

**Prefill dominant**: large batches → strongly compute-bound
- Option B: peak benefit, ~60-70% prefill reduction
- Option C: ~70-80% prefill reduction
- Both excellent at S∞

---

## Recommendation

### Immediate: Option B

**Reasons**:
1. SM120 FP8 kernel already exists → zero CUDA risk
2. 3-5 day implementation timeline fits SOAR competition pace
3. Expected to score meaningfully better in S1 (40% weight) and S∞ (30% weight)
4. Low risk to accuracy (FP8 blockwise is well-studied)
5. Leaves competition time for further optimization

### Future: Option C (if needed after B)

**When to start C**:
- After B is deployed and scored
- If gap to top 5 requires additional >20% improvement
- If profiling post-B shows decode is the new bottleneck (W4 for decode avoids FP8's 2× BW cost)
- If competition timeline allows 4+ weeks of kernel work

**How to approach C efficiently**:
1. **Start with CuTe mini-kernel** (1 SM, small tile, no pipelining) to validate correctness
2. **Add TMA prefetch** for W4 loading (Stage 2)
3. **Add multi-stage pipeline** (Stage 3)
4. **Tune tile sizes** for MiniCPM-SALA shapes (Stage 4)

Key reference for C: `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` for existing W4 dequant logic, `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` for SM120 UMMA templates.

---

## Option B Next Steps (Implementation Plan)

See `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` for full implementation details.

**High-level steps**:
1. Add `gptq_to_fp8_blockwise()` in `preprocess_model.py` — converts GPTQ layers to FP8 + block scales
2. Add `FP8BlockwiseLinear` module in `linear.py` — per-row-per-block activation quant + `fp8_blockwise_scaled_mm`
3. Plumb FP8 linear into `minicpm.py` conditional on weight dtype
4. Verify accuracy on local eval (`eval_model_001.py`)
5. Benchmark speed (S1, S8, Smax) on fcloud

**Key questions to answer in testing**:
- Does normalized accuracy stay > 99% with FP8 blockwise weights? (Target: C=1.0)
- What is actual TFLOPS utilization on SM120 for MiniCPM-SALA shapes?
- Does FP8 decode overhead from 2× weight bandwidth hurt S1 (which may be decode-heavy)?

---

## Option C Reference Implementation Notes

For future engineers implementing Option C, key code references:

**Existing W4 dequant logic** (adapt for FP8 output):
- `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` — GPTQ dequantization logic
- `sgl-kernel/csrc/gemm/gptq/qdq_4.cuh` — INT4 pack/unpack utilities

**SM120 UMMA setup** (reference template):
- `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` lines 206-353 — full SM120 kernel template
- `cutlass::arch::Sm120`, `cute::UMMA::Major::MN`

**TMA for W4 weights** (non-standard — packed INT32):
- Need custom `cute::Tensor` descriptor for (K/8, N) packed layout
- Reference: CuTe tutorial on non-standard layouts

**Target kernel structure**:
```
// Option C pseudocode
__global__ void gptq_w4a8_sm120_gemm_kernel(
    int8_t* A_fp8,          // (M, K) FP8 activation (pre-quantized per-token)
    int32_t* B_w4_packed,   // (K/8, N) GPTQ packed W4
    float* B_scales,        // (K/128, N) GPTQ group scales
    int8_t* B_zeros,        // (K/128, N) GPTQ zero points
    float* A_scales,        // (M, K/128) activation block scales
    bfloat16_t* C,          // (M, N) output
    int M, int N, int K
) {
    // Stage 1: TMA prefetch 128×128 W4 tile (=128×16 int32)
    // Stage 2: Unpack + dequant to FP8 in smem using GPTQ scales
    // Stage 3: tcgen05.mma.ws.sync fp8×fp8 → fp32 accumulator
    // Stage 4: Store fp32 acc + scale → bf16 output via TMA
}
```

**Expected tile configuration for MiniCPM-SALA shapes**:
- Standard linear layers: K≈4096-8192, N≈4096-16384
- Best tile: M=128, N=128, K=128 (one GPTQ group fits K-dimension exactly)
- SM occupancy target: 2 waves minimum (192 tiles for 96 SMs)

---

## Conclusion

Option B is the correct immediate choice:
- **Zero CUDA kernel risk** — existing SM120 FP8 kernel
- **3-5 days to implement and test**
- **Expected 30-50% speedup** on S1/S∞ (compute-bound prefill dominant)
- Option C is documented here for future reference with clear implementation guide

Both options are fully SOAR-compliant (no forbidden tricks, no external deps, reproducible).
