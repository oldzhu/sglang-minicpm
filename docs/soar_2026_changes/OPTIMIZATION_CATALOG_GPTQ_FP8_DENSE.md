# Complete Optimization Catalog: GPTQ + FP8 KV + Dense Mode

> **Created**: 2026-04-14 | **Updated**: 2026-04-20  
> **Baseline**: Test 12 — S1=121.71s, S8=44.09s, Smax=35.86s, ori_accuracy=79.29%, normalized=99.11%, C=1.0  
> **Best config (2026-04-20)**: prefill-max-req=4, sched-cons=0.8, chunk=65536, torch.compile(max-bs=8), mixed-chunk  
> **Official score**: 40.23 (#19) after new long-context dataset rerun  
> **Config**: `--quantization gptq_marlin --force-dense-minicpm --kv-cache-dtype fp8_e5m2`  
> **Architecture**: MiniCPM-SALA — 32 layers (8 standard attention + 24 SimpleGLA lightning/recurrent)

This document catalogs **every known speed optimization vector** for our baseline config, ordered from the top Python scheduling layer down to bottom CUDA kernels.

---

## Layer 1: Server Scheduling (Python — config only)

| ID | Optimization | Current | Proposed | Expected Gain | Effort | Risk | Code Change? |
|----|-------------|---------|----------|---------------|--------|------|--------------|
| **S1** | `--enable-torch-compile --torch-compile-max-bs 8` | Off | On | **5-7%** | Config only | OOM if max-bs too high | No |
| **S2** | `--enable-mixed-chunk` | Off | On | **3-5%** | Config only | Low | No |
| **S3** | `--prefill-max-requests` | 1 | 2-4 | **5-10%** (multi-request workload) | Config only | Medium (memory) | No |
| **S4** | `--schedule-conservativeness` | 1.0 | 0.95 | **3-5%** | Config only | Medium (occasional OOM) | No |
| **S5** | `--max-running-requests` | 20 | 24-32 | **2-3%** | Config only | Low | No |

### Details

**S1 — torch.compile**: Wraps `model.forward` with `torch.compile(mode="max-autotune-no-cudagraphs")` for batch sizes ≤ max-bs. Inductor backend applies operator fusion, reduced kernel launches, and auto-tuning. Compilation happens at server startup (+3-10 min). Test 12-VarC showed 5-7% speedup but OOM at max-bs=32. Try max-bs=8 first.

**S2 — Mixed chunking**: Allows decode requests from running_batch to be mixed with new prefill batch in the same scheduling round. Reduces idle GPU cycles between prefill and decode phases.

**S3 — Prefill max requests**: Currently forces ONE prefill request per scheduling round. Increasing to 2-4 packs more prefill work together, improving GPU utilization for bursty/small-request workloads. Monitor memory usage.

**S4 — Schedule conservativeness**: Controls token budget aggressiveness (1.0 = strict, 0.95 = 5% oversubscription). More aggressive packing means better throughput but risk of occasional OOM under peak load.

**S5 — Max running requests**: Hard limit on concurrent requests in flight. 20 is conservative; 24-32 may utilize KV cache memory more fully.

---

## Layer 2: Model Forward Pass (Python/PyTorch)

| ID | Optimization | What | Expected Gain | Effort | Risk | Code Change? |
|----|-------------|------|---------------|--------|------|--------------|
| **M1** | **Residual scale folding** | Fold fixed runtime scales (`residual_scale`, `scale_emb`, `1/scale_width`) into weights at load time to remove scalar kernels in forward path. | **1-3%** | Low (load-time only) | Medium (implementation-sensitive) | Yes (weight loading) |
| **M2** | **bf16 RoPE** (CHANGE_0075) | Remove float32 upcast in standard attention RoPE. Keep computation in bf16. | **1-2%** | Done | Low (needs test on correct config) | Done |
| **M3** | **In-place residual scale** (CHANGE_0075) | `hidden_states *= self.residual_scale` instead of `hidden_states = hidden_states * self.residual_scale` | **0.5-1%** | Done | Low | Done |

### Details

**M1 — Residual scale folding (TESTED 2026-04-17, commit f373fbade)**: Implemented by folding `residual_scale` into GPTQ Marlin `o_proj.scales`/`down_proj.scales`, `scale_emb` into `embed_tokens.weight`, and `1/scale_width` into `lm_head.weight`, with runtime scaling guards disabled after fold. Test 23 results: S1 112.55s (vs 113.67s, -1.0%), S8 41.04s (vs 41.07s, flat), Smax 34.58s (vs 34.15s, +1.3% slower); accuracy 78.64%, normalized 98.30%, C=0.96. Current verdict: **not submission-safe** (accuracy regression), keep as experimental branch only.

**M2 + M3 — CHANGE_0075**: Already committed (290e370e6). bf16 RoPE removes unnecessary float32→bf16→float32 casts. In-place `*=` avoids tensor copy. Both need testing on the correct dense+FP8 config (Tests 14-16 all ran on wrong sparse config).

---

## Layer 3: Attention Backends (Python + CUDA)

| ID | Optimization | What | Expected Gain | Effort | Risk | Code Change? |
|----|-------------|------|---------------|--------|------|--------------|
| **A1** | **SimpleGLA state contiguity** | Force batch state allocations contiguous in memory pool so fast I/O always triggers (currently falls back to loop-based gather/scatter when scattered) | **5-8% decode** | Medium | Low | Yes (memory_pool.py) |
| **A2** | **FLA chunk size tuning** | ~~Fork `fla` library, test chunk sizes 32/64/96/128/192 for prefill~~ | **0% (TESTED)** | Done | — | Done (CHANGE_0080) |
| **A3** | **Fuse state I/O into FLA kernel** | Patch `fla` to accept state pool pointers directly — eliminate separate load/store per decode step | **10-15% decode** | High | Medium | Yes (fla CUDA kernel) |
| **A4** | **Recurrent threshold tuning** | ~~`SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128` — test 64, 96, 192~~ | **0% (TESTED)** | Done | — | No |

### Details

**A1 — State contiguity**: The SimpleGLA backend has a fast path (`SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=1`) that uses direct tensor indexing for state load/store. But when batch layout is scattered in the memory pool, it falls back to a loop-based gather/scatter. Guaranteeing contiguous state allocation would ensure the fast path always triggers.

**A2 — FLA chunk size (TESTED, NO GAIN)**: Tested chunk_size=32/64/128 and threshold=64/128/256 on fcloud (Test 19, commit 23d1c8ecf). All configs produced identical results (S1 ~112.96s, S8 ~41.55s). Chunk size and threshold only affect prefill kernel selection, but the workload is decode-dominated. These parameters are at their optimal defaults.

**A3 — Fuse state I/O**: Currently, every decode step does: load state from pool → run kernel → store state back to pool. If the FLA kernel could accept a pointer to the state pool directly and read/write in-place, it would eliminate the load/store overhead entirely. This is the highest-potential single optimization but requires modifying the FLA CUDA kernel.

**A4 — Recurrent threshold (TESTED, NO GAIN)**: Tested threshold=64/128/256. No measurable impact. The threshold controls chunk vs recurrent kernel selection for prefill; decode always uses recurrent regardless of threshold.

### Critical Insight (UPDATED 2026-04-20 — PROFILING DATA)

> **PREVIOUS ASSUMPTION (DISPROVED)**: "The 24 SimpleGLA layers account for ~75% of forward pass time."
>
> **ACTUAL PROFILING RESULT**: On 25K-30K token inputs:
> - **GEMM (Marlin/GPTQ) = 85.3% of prefill** — the true bottleneck
> - FLA/SimpleGLA = 12.4% of prefill (chunk_fwd = 1%, FlashInfer attention = 8.8%, rest = 2.6%)
> - For decode: GEMM = 63.5%, torch.compile fused = 29.4%, FLA = 4.9%
>
> A1/A3 optimizations still have value for decode, but **GEMM optimization (Layer 5) is now the highest-impact target** for overall throughput, especially with the new 32K-512K token dataset.
>
> See `docs/soar_2026_changes/CHANGE_0120_profiling_analysis.en.md` for full profiling data.

---

## Layer 4: Custom CUDA Kernels (sgl-kernel)

| ID | Optimization | What | Expected Gain | Effort | Risk | Code Change? |
|----|-------------|------|---------------|--------|------|--------------|
| **K1** | **Fused QK-norm-RoPE** | Already enabled (`--enable-fused-qk-norm-rope`). Fuses Q norm + K norm + RoPE into 1 kernel for lightning layers. | ✅ Already on | — | — | — |
| **K2** | **Fused add + RMSNorm** | Already fused (`fused_add_rmsnorm_kernel.cu`). Residual add + RMSNorm in single kernel. | ✅ Already on | — | — | — |
| **K3** | **SiluAndMul** | Already fused. Gate activation + elementwise multiply in 1 kernel. | ✅ Already on | — | — | — |
| **K4** | **Fused RMSNorm + residual_scale** | Modify `fused_add_rmsnorm` kernel to accept an extra `residual_scale` scalar, multiply inline. Saves 2 kernel launches per layer (64 total). | **1-2%** | Medium | Low | Yes (CUDA kernel) |

### Details

**K4 — Fused RMSNorm + residual_scale**: The existing `sgl_fused_add_rmsnorm()` kernel takes `(input, residual, weight, eps)`. Adding a `residual_scale` scalar parameter would let it compute `output = rmsnorm(input + residual) * residual_scale` in a single kernel instead of two (rmsnorm + multiply). This is a relatively safe kernel modification — the scalar multiply is trivially fused into the epilogue.

---

## Layer 5: GPTQ Marlin GEMM (CUDA)

| ID | Optimization | What | Expected Gain | Effort | Risk | Code Change? |
|----|-------------|------|---------------|--------|------|--------------|
| **G1** | **Marlin kernel** | Already used — fused dequant + GEMM, highly optimized for H100/L40S | ✅ Already on | — | — | — |
| **G2** | **Workspace buffer pooling** | Reuse Marlin workspace buffers across layers instead of allocating per-call | **0.5-1%** | Low | Zero | Yes (linear.py) |
| **G3** | **Dense-specific requantization** | Recalibrate GPTQ with dense-mode attention (current calibration was for sparse mode). Could improve weight accuracy → allow more aggressive quantization. | **2-5%** | Very High (hours of calibration) | Medium (accuracy) | No (config) |

### Details

**G2 — Workspace pooling**: The Marlin GEMM allocates workspace buffers (small temporary GPU memory) per forward call. Allocating once and reusing across all layers would reduce allocator overhead slightly.

**G3 — Dense requantization**: **CONFIRMED MISMATCH** (2026-04-17): GPTQ calibration runs via `GPTQModel.load()` through HuggingFace transformers, which uses the model's **native sparse attention** (topk=64). However, sglang inference uses `--force-dense-minicpm` (dense attention). The `SOAR_GPTQ_ATTN_IMPL=flash_attention_2` env var only controls the HF attention *backend*, not sparse/dense mode. Since this is W4A16 (weights-only quantization), the impact is second-order — activations aren't quantized so the quantization grid mismatch is small. Fix would require adding `force_dense` support to the HF modeling code or monkey-patching during calibration. Low priority but worth doing if accuracy margin tightens.

---

## What's Already Maxed Out (No Further Gains)

| Component | Status |
|-----------|--------|
| Marlin GPTQ GEMM | At hardware limits — fused dequant + mixed-precision GEMM |
| Fused QK-norm-RoPE | Already on |
| Fused add + RMSNorm | Already on |
| SiluAndMul fusion | Already on |
| FP8 KV cache | Already on (standard attention only; not applicable to SimpleGLA state) |
| Lightning fast state I/O | Already on (`SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=1`) |
| Lightning fast output gate | Already on (`SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=1`) |
| Dense-as-sparse | Already on (`--dense-as-sparse`) |
| Radix cache | Already disabled (`--disable-radix-cache`) — correct for eval with `--flush-cache` |

---

## Priority Ranking (Bang-for-Buck)

### Tier 1: Test Immediately (config-only, zero code changes)
| Priority | ID | Optimization | Expected Gain | Status |
|----------|----|-------------|---------------|--------|
| 1 | S1 | `--enable-torch-compile --torch-compile-max-bs 8` | **5-7%** | ✅ DONE (Test 18, -7.4%/-6.6%/-1.5%) |
| 2 | S2 | `--enable-mixed-chunk` | **3-5%** | ✅ DONE (Test 20, Smax -4.0%) |
| 3 | A4 | Recurrent threshold tuning (64/96/192) | ~~1-3%~~ **0%** | ❌ TESTED — no gain |
| 4 | S3 | `--prefill-max-requests 4` | **5-10%** | ✅ DONE (Test 25A, S1 -8.2%) |
| 5 | S4 | `--schedule-conservativeness 0.8` | **3-5%** | ✅ DONE (Test 25A, part of S3 combo) |

### Tier 2: Quick Code Changes (1-2 days, low risk)
| Priority | ID | Optimization | Expected Gain | Status |
|----------|----|-------------|---------------|--------|
| 6 | M1 | Residual scale folding into weights | **1-3%** | ⚠️ Tested (Test 23): accuracy regression, not adopted |
| 7 | M2+M3 | CHANGE_0075 verify on correct config | **1.5-3%** | ✅ DONE (Test 17, no speed change but accuracy safe) |
| 8 | G2 | Workspace buffer pooling | **0.5-1%** | ⬜ Not started |

### Tier 3: Medium Code Changes (3-5 days, medium risk)
| Priority | ID | Optimization | Expected Gain | Status |
|----------|----|-------------|---------------|--------|
| 9 | **G4** | **Marlin GEMM SM120 auto-config** | **5-15% prefill** | ⬜ Not started — **NEW #1 PRIORITY (85.3% of prefill)** |
| 10 | **G5** | **FP8 W8A16 quantization** | **10-30% prefill** | ⬜ Not started — replace Marlin W4 with FP8 tensor cores |
| 11 | A1 | SimpleGLA state contiguity guarantee | ~~5-8% decode~~ **<1% prefill, ~3% decode** | ⬜ Not started — **deprioritized by profiling** |
| 12 | A2 | FLA chunk size tuning | ~~5-10%~~ **0%** | ❌ TESTED — no gain (Test 19) |
| 13 | K4 | Fused RMSNorm + residual_scale kernel | ~~1-2%~~ **<0.5%** | ⬜ Not started — **deprioritized by profiling** |

### Tier 4: Major Engineering (1-2 weeks, high effort)
| Priority | ID | Optimization | Expected Gain |
|----------|----|-------------|---------------|
| 12 | A3 | Fuse state I/O into FLA kernel | **10-15% decode** |
| 13 | G3 | Dense-specific GPTQ recalibration | **2-5%** |

---

## Theoretical Maximum Cumulative Gain

If all optimizations succeed (optimistic):
- Tier 1 (config): ~15-25%
- Tier 2 (quick code): ~3-7%
- Tier 3 (medium code): ~10-18%
- Tier 4 (major): ~12-20%
- **Total theoretical**: ~40-70% (multiplicative, not additive)

**Reality check (updated with profiling data)**: Profiling shows GEMM is 85.3% of prefill and 63.5% of decode. With the new 32K-512K dataset being prefill-dominant, the highest-leverage optimization is **faster GEMM** — either via Marlin kernel tuning for Blackwell SM120 or switching to FP8 W8A16 (native tensor cores). FLA/SimpleGLA optimization has much lower impact than originally estimated (~12.4% of prefill vs assumed 75%). With score at 40.23 (#19), we need **~37.7% faster** to reach #5 (64.58). This magnitude of speedup likely requires GEMM-level changes (FP8 quantization) rather than scheduling tuning or minor kernel fusions.

---

## Strategic Assessment (Updated 2026-04-20)

### New Dataset Impact
The competition deployed a new long-context speed dataset (2026-04-15): **68% of inputs are 32K-512K tokens**. This fundamentally shifts the optimization target from decode throughput to **prefill throughput**. Our config changes (chunk=65536, prefill-max-requests=4) showed no gain on old data (max ~7K token inputs) but will have significant impact on the new long-context workload.

### Config Tuning Status: EXHAUSTED
All Layer 1 (scheduling) optimizations have been tested:
- S1 torch.compile: +7.4% S1 ✅
- S2 mixed-chunk: +4.0% Smax ✅
- S3 prefill-max-requests=4: **+8.2% S1** ✅
- S4 schedule-conservativeness=0.8: combined with S3 ✅
- S5 max-running-requests=24: included ✅
- No further config-only gains available.

### NVFP4 Status: NOT VIABLE
Test 21 showed catastrophic accuracy failure (~12%) with W4A4 FP4 quantization. Model generates infinite `<think>` loops. FP4 is too aggressive for this reasoning architecture. Mixed-precision NVFP4 (per-layer exclusion) would require major engineering.

### Priority Paths Forward (REVISED per profiling data 2026-04-20)
1. **Submit with best config v19** (immediate) — prefill-max-req=4, sched-cons=0.8, chunk=65536
2. **Marlin GEMM profiling & tuning** (highest impact) — 85.3% of prefill; verify SM120 auto-config tile sizes, thread group selection
3. **FP8 weight quantization (W8A16)** (potentially transformative) — Replace Marlin W4 dequant+FP16 GEMM with native FP8 tensor cores (~2× GEMM throughput). Safe accuracy (8-bit)
4. **FlashInfer attention optimization** (8 standard layers) — 8.8% of prefill (each `BatchPrefillWithRaggedKVCacheKernel` call ~55ms)
5. **FLA chunk kernel optimization** (deprioritized) — only 1% of prefill (chunk_fwd_o + chunk_fwd_h = 50ms vs GEMM 4258ms)
6. **A1 state contiguity / A3 fused state I/O** (decode-focused) — 4.9% of decode; state I/O indexing ~3% of decode
7. **EAGLE3 spec decode** (if time) — diminished returns on prefill-dominant workload

---

## Blocked / Not Viable (Updated)

| Optimization | Reason |
|-------------|--------|
| NGRAM speculative decoding | Incompatible with SimpleGLA recurrent layers — corrupts state with no rollback |
| EAGLE speculative decoding | Requires SimpleGLA state save/restore (not implemented). **MVP approach identified**: sequential verify + state checkpoint/rollback. Work started in eagle3-spec-decode branch. Potential **30-50%** speedup if implemented. |
| Sparse attention mode | ~50% accuracy on fcloud — not viable as submission config |
| FP8 for SimpleGLA state | `fla` library kernels don't support FP8 scale parameters — external dequant would negate gains |
| Model architecture changes | Locked for competition |
| FLA chunk size tuning (A2) | **Tested**: zero impact across chunk_size=32/64/128 (Test 19) |
| Recurrent threshold tuning (A4) | **Tested**: zero impact across threshold=64/128/256 (Test 19) |

---

## Environment Variables Already Tuned

```bash
# Lightning layer optimizations (all ON)
SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=1
SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=1
SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128
SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
SOAR_ENABLE_FUSED_QK_NORM_ROPE=1

# Memory management
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6
```

---

## Testing Plan

When fcloud comes back online, test in this order:
1. **Test 17**: CHANGE_0075 (bf16 RoPE + in-place residual) on correct dense+FP8 config → verify accuracy
2. **Test 18**: Add `--enable-torch-compile --torch-compile-max-bs 8` → verify no OOM + measure speed
3. **Test 19**: Add `--enable-mixed-chunk` → measure speed delta
4. **Test 20**: Try `--prefill-max-requests 2` → measure speed delta
5. Stack winning combinations for final benchmark
