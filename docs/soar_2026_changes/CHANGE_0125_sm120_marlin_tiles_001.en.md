# CHANGE_0125: SM120 Marlin Tiles — Investigation & Test Results

**Continuation of CHANGE_0125_sm120_marlin_tiles**

## Test Results (Test 27, 2026-04-20)

### Accuracy

| Metric | Test 27 (CHANGE_0125) | Baseline (Test 25) | Delta |
|--------|----------------------|---------------------|-------|
| **Overall accuracy** | 77.18% | 79.00% | -2.3% |
| mcq | 56.67% | 53.33% | +3.3% |
| cwe | 83.67% | 85.00% | -1.3% |
| fwe | 98.89% | 100% | -1.1% |
| niah | 100% | 100% | 0% |
| qa | 46.67% | 56.67% | -10.0% |
| Duration | 3123s | 2988s | +4.5% |
| TPS | 468.74 | 423.75 | — |

The qa task dropped to 46.67% — this is **test variance**, not caused by CHANGE_0125 (see analysis below).

### Speed

| Tier | Test 27 | Baseline (Test 25B) | Delta |
|------|---------|---------------------|-------|
| S1 | 111.48s | 110.54s | +0.9% |
| S8 | 40.42s | 40.54s | -0.3% |
| Smax | 33.53s | 33.58s | -0.1% |

**Result: NEUTRAL** — no measurable speed change in any tier.

### Server Log Confirmation

Only **one** SM120 auto-config message was logged:
```
[sgl-kernel] SM120 Marlin auto-config enabled: M=20 N=4608 K=4096
  thread_m_blocks=2 thread_n=64 thread_k=128 num_threads=128
  source=score occupancy_blocks_per_sm=2 score=262.33
```

The selected config is `(thread_m_blocks=2, thread_n_blocks=4, thread_k_blocks=8)` — an **existing** tile `{4,8,128}`, NOT any of the 3 new tiles from CHANGE_0125. The new tiles were compiled but never selected by the scoring function.

---

## Root Cause Investigation: Why New Tiles Were Not Selected

### 1. The Scoring Formula

The SM120 scorer in `gptq_marlin.cu` (`score_sm120_candidate()`) computes:

```
score = fill_ratio  × 1000.0     (dominant: fraction of SM slots filled)
      + wave_ratio  × 100.0      (total work waves, capped at 4)
      + m_coverage  × 10.0       (fraction of M-tile actually used)
      + smem_fit    × 1.0        (smaller shared memory = better)
      + occupancy   × 0.5        (concurrent blocks per SM)
      + total_tiles × 0.01       (tiebreaker)
```

**`fill_ratio × 1000` completely dominates** — it measures what fraction of the GPU's total SM slots (SMs × occupancy_blocks_per_sm) are filled by tiles. Everything else combined maxes out at ~410 points. The scorer overwhelmingly prefers configs that **produce more tiles** to fill all SMs.

### 2. MiniCPM-SALA Weight Shapes

The model has 153 GEMM calls per forward pass:

| Layer | N | K | Count | Quant |
|-------|---|---|-------|-------|
| gate_up_proj (MLP) | 28672 | 4096 | 32 | W4 g128 |
| down_proj (MLP) | 4096 | 14336 | 32 | W4 g128 |
| std qkv_proj | 6144 | 4096 | 8 | W8 g128 |
| std o_proj | 4096 | 4096 | 8 | W4 g128 |
| lightning qkv_proj | 3072 | 4096 | 24 | W4 g128 |
| lightning o_proj | 4096 | 1024 | 24 | W4 g128 |
| z_proj | 1024 | 4096 | 24 | W4 g128 |
| lm_head | 150528 | 4096 | 1 | FP16 (not Marlin) |

Model architecture: 8 standard attention layers + 24 lightning (SimpleGLA) layers, all with MLP.

### 3. Tile Count Analysis: Why Wide Tiles Lose

The RTX PRO 6000 (SM120/Blackwell GB202) has **96 SMs**. The key question is how many tiles each config produces for a given GEMM shape.

For the logged shape **M=20, N=4608, K=4096** (std qkv_proj, `thread_m_blocks=2` → tile_m=32):

| Config | thread_n | n_tiles (N/thread_n) | m_tiles | total_tiles | fill_ratio (÷96) | Score (dominant) |
|--------|----------|----------------------|---------|-------------|-------------------|------------------|
| **{16,8,256}** NEW | 256 | 18 | 1 | **18** | 0.19 | **~188** |
| **{16,4,256}** NEW | 256 | 18 | 1 | **18** | 0.19 | **~188** |
| **{8,8,256}** NEW | 128 | 36 | 1 | **36** | 0.38 | **~375** |
| {8,4,128} existing | 128 | 36 | 1 | **36** | 0.38 | **~375** |
| **{4,8,128}** existing | 64 | 72 | 1 | **72** | 0.75 | **~750** ← WINNER |

The existing `{4,8,128}` wins by **2-4× in score** because it creates 72 tiles vs 18-36. With only 18 tiles from `thread_n=256`, 78 of 96 SMs would sit idle in the first wave.

### 4. Even the Largest N (28672) Can't Save Wide Tiles

For gate_up_proj (N=28672, most favorable case for wide tiles):

| Config | thread_n | n_tiles | fill_ratio (÷96) | wave_ratio × 100 |
|--------|----------|---------|-------------------|-------------------|
| {16,8,256} | 256 | 112 | 1.0 (full!) | 112/96 = 1.17 → **117** |
| {4,8,128} | 64 | 448 | 1.0 (full!) | 448/96 = 4.67 → capped **400** |

When fill_ratio ties at 1.0, `wave_ratio × 100` breaks the tie. The narrow tile gets 400 points vs 117 points — an insurmountable 283-point gap. The wide tile would need to win on `smem_fit` (weight 1×) and `occupancy` (weight 0.5×), which is impossible.

### 5. Minimum N for Wide Tiles to Win

For `thread_n=256` to be competitive, `total_tiles ≥ SMs` (96):
- `n_tiles = N / 256`, with `m_tiles = ceil(M / tile_m)`
- For decode (M=1, m_tiles=1): need N ≥ 96 × 256 = **24,576**
- For small prefill (M=20, m_tiles=1 at tile_m=32): need N ≥ **24,576**

MiniCPM-SALA's largest N dimension is 28,672 (gate_up_proj) — barely above the threshold. But even there, the narrow tile wins via wave_ratio. **No layer in MiniCPM-SALA can benefit from `thread_n=256` tiles.**

### 6. Template Parameter Physical Meaning

```
thread_m_blocks: M tiles, each 16 rows → coverage = thread_m_blocks × 16
thread_n_blocks: N tiles, each 16 cols → coverage = thread_n_blocks × 16
thread_k_blocks: K iteration, each 16 elements → K per step = thread_k_blocks × 16
num_threads: threads per threadblock (128 = 4 warps, 256 = 8 warps)
```

The MMA instruction is **`mma.sync.aligned.m16n8k16`** (SM80-era), NOT SM120's native warp-level MMA.

### 7. Why SM120 Tile Changes Fundamentally Cannot Help

Three independent reasons:

**A. Wrong instruction set**: Marlin uses SM80's `mma.sync.aligned.m16n8k16`. SM120 Blackwell has native warp-level MMA with higher throughput, but Marlin doesn't use it. Tile shape changes can't access SM120 hardware capabilities.

**B. Occupancy physics**: Wider tiles produce fewer thread blocks. With 96 SMs and MiniCPM's N ≤ 28,672, wide tiles always leave SMs idle. The scorer correctly identifies this.

**C. Memory-bound regime**: At small M (decode, M=1-8), GEMMs are memory-bandwidth-limited. The bottleneck is loading W4 weights from GDDR7 (1398 GB/s), not compute. More tiles spread memory access across more SMs, which is better.

---

## What Would Actually Help on SM120?

| Approach | Impact | Effort | Risk | Notes |
|----------|--------|--------|------|-------|
| **SM120 native warp-level MMA** | High (potentially 2×) | Very high | High | Rewrite Marlin kernel to use SM120 MMA instead of SM80 mma.sync |
| **TMA (Tensor Memory Accelerator)** | Medium | High | Medium | Async memory loads, better pipelining vs current shared memory staging |
| **QMMA (mxfp8)** | High (2× vs FP16) | Medium | Accuracy risk | SM120 has 296 TFLOPS FP8 vs 148 TFLOPS BF16. Requires FP8 GEMM path |
| **CUTLASS 3.x SM120 kernels** | High | Very high | Medium | Replace Marlin entirely with CUTLASS sm120 GEMM (NVIDIA's optimized path) |
| **TRT-LLM SM120 reference** | Potentially high | High | Medium | Competition hints: "TRTLLM will give user some reference usage" for SM120 |
| Scorer weight tuning | Negligible | Low | Low | Won't help — scorer is already making correct decisions |
| More tile instantiations | None | Low | None | Same problem — more tiles to choose from doesn't help if fundamentals are wrong |

### Key Hardware Specs (from competition PDF)

| Feature | Value |
|---------|-------|
| Architecture | SM120 (Blackwell) |
| BF16/FP16 Tensor Core | 148 TFLOPS |
| FP8 Tensor Core | 296 TFLOPS |
| FP4 Tensor Core | 593 TFLOPS |
| GPU Memory | 84 GB GDDR7 |
| Memory Bandwidth | 1398 GB/s |
| L2 Cache | 112 MB |
| MMA | Warp-level (NOT warpgroup) |
| TMA | Supported |
| QMMA | Supported (mxfp8) |

Key difference from Hopper (SM90): MMA is **warp-level** instead of warpgroup-level. Optimization strategies are "basically similar to Hopper" per competition organizers.

---

## Conclusions

1. **CHANGE_0125 is functionally neutral** — the new tiles compiled successfully but the scoring function correctly never selects them for MiniCPM-SALA's weight shapes.

2. **The scoring function is NOT broken** — it correctly identifies that narrow tiles (thread_n=64) maximize SM utilization for this model's N dimensions (1024-28672).

3. **Tuning the scoring function is a dead end** — the occupancy advantage of narrow tiles is physically real, not a scoring artifact.

4. **SM120 hardware advantages (native MMA, TMA, QMMA) are NOT accessible through tile changes** — Marlin uses SM80-era instructions regardless of tile configuration.

5. **To truly benefit from SM120**: Need kernel-level changes to use native SM120 MMA instructions, TMA, or switch to CUTLASS/TRT-LLM SM120 kernels. This is a major undertaking beyond tile tuning.

6. **Recommendation**: CHANGE_0125 can be kept (harmless) or reverted (saves ~1 minute compile time on incremental builds). The Marlin tile optimization path is exhausted for MiniCPM-SALA. Future GEMM optimization should focus on SM120-native kernel integration.
