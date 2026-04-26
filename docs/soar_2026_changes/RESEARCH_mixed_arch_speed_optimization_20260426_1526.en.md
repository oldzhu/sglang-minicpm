# Research note — Speed optimization for MiniCPM-SALA mixed architecture

**Date**: 2026-04-26 15:26
**Context**: User queries on (1) 4-bit quantization status, (2) 4-bit KV cache scope, (3) optimization vectors specific to the 24 lightning (linear-attention) layers vs the 8 standard-attention layers.
**Status**: Discussion / analysis only. No code change. Used as input for next-iteration proposal selection.

---

## 0. Architecture facts to anchor the discussion

| Layer type | Count | Compute model | KV / state | Complexity (seq len `N`) |
|---|---|---|---|---|
| **`minicpm4` (standard attention)** | **8** | `softmax(QKᵀ)V` with paged KV cache, RoPE, GQA | Paged KV (FP8 e5m2 today; budget bounded by 84 GB GDDR7) | Prefill **O(N²·d)** compute; Decode **O(N·d)** memory-bound (KV bandwidth) |
| **`lightning` (SimpleGLA / linear attention)** | **24** | `chunk_simple_gla` for prefill, `fused_recurrent_simple_gla` for decode; recurrent state update `S = g·S + kᵀv`, output `o = q·S` | **Fixed-size** state `(num_kv_heads, head_dim, head_dim)` per layer per request; **NO seq-len-dependent KV** | Prefill **O(N·d²)** ≈ linear in N; Decode **O(d²) per token, independent of N** |

**The single most important consequence**: as input length grows, the standard 8 layers dominate cost (KV grows linearly with N; lightning state stays constant). At the long-context regime that matters in official scoring, optimizing the 8 layers gives the biggest leverage. But the 24 lightning layers still dominate at small context / decode-with-short-history because they account for 75 % of the layer count and their per-token cost is constant `d²`.

Code references (validated 2026-04-26):
- Layer dispatch: `python/sglang/srt/models/minicpm.py:530` (`self.mixer_type = config.mixer_types[layer_id]`).
- Lightning kernel selection: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:1632-1672` — `chunk_simple_gla` vs `fused_recurrent_simple_gla` based on `forward_mode.is_decode()` or `_select_mode(forward_batch)`.
- Lightning state shape: `python/sglang/srt/models/minicpm.py:407` (`self.state_shape = (self.num_kv_heads, self.head_dim, self.head_dim)`).

---

## 1. User query #1 — "4-bit quantization (already W4A8?)"

**Not quite.** What we ship today (verified via `benchmark/soar/demo_sala/preprocess_model.py` + Marlin path):

- GPTQ **W4A16** for most linear layers (weight INT4, activation BF16).
- GPTQ **W8A16** for QKV of the 8 standard-attention ("sparse") layers (`SOAR_GPTQ_SPARSE_QKV_BITS=8`, `group_size=128`).
- KV cache **FP8 e5m2** (separate from weight quant).
- Activations are **BF16 throughout the GEMMs** (Marlin W4A16 / W8A16 kernels).

So we have **W4 / W8 weight-only**, NOT W4A8 and NOT W4A4. Real W4A8 (INT8 activations + INT4 weights via QQQ-style or Marlin-W4A8) and W4A4 (NVFP4) are separate optimization vectors:

| Variant | Weight | Activation | SM120 expected gain | Status |
|---|---|---|---|---|
| W4A16 (current) | INT4 | BF16 | 1× (baseline) | ✅ shipped |
| W8A16 (sparse QKV) | INT8 | BF16 | 0.6× (slower than W4A16) | ✅ kept for accuracy |
| **W4A8 INT8 act** | INT4 | INT8 | ~1.4× (FP8 GEMM at 296 TFLOPS) | ❌ not tried — needs activation calibration + Marlin W4A8 kernel path |
| **W4A8 FP8 act (mxfp8 / QMMA)** | INT4 | FP8 | ~1.5-1.8× (uses QMMA on SM120) | ❌ not tried — most promising on Blackwell |
| **NVFP4 W4A4** | FP4 | FP4 | ~2.5× theoretical (593 TFLOPS) | ❌ Test 21 catastrophic acc ~12 % — destroys reasoning |

**New optimization opportunity**: **W4A8 with FP8 activations**. SM120 has hardware QMMA for mxfp8. Using it for activations while keeping weights at GPTQ INT4 would roughly double GEMM throughput vs current W4A16, with much smaller expected accuracy loss than NVFP4 W4A4 (FP8 e4m3 activations preserve ~99 % of BF16 accuracy on most LLMs — and we already proved FP8 survives in our KV cache).

---

## 2. User query #2 — "4-bit KV cache (only for the 8 layers?)"

**Correct on both counts.** Lightning layers don't have a token-indexed KV cache — they have a fixed `d×d` recurrent state that doesn't benefit from "per-token quantization". So 4-bit KV applies **only to the 8 standard layers**.

Why 4-bit KV is interesting on SM120:
- KV bandwidth is the dominant decode bottleneck (RTX PRO 6000 = 1398 GB/s).
- Halving KV size from FP8→FP4 halves the bandwidth pressure.
- SGLang already has `MHATokenToKVPoolFP4` and `--kv-cache-dtype fp4_e2m1` upstream (`python/sglang/srt/mem_cache/memory_pool.py:1085`, auto-creation at `model_runner_kv_cache_mixin.py:583`). M2.0 ablation proposal (already drafted) addresses this.

**Why "mixed" matters specifically** (champion W5 finding restated for our model):
- First / last few layers are accuracy-critical (attention sinks, output decisions).
- Middle layers are surprisingly tolerant to FP4 KV.
- Our 8 std-attn layers are NOT contiguous — they're interspersed in `mixer_types`. So we need to identify per-position sensitivity by layer-index-within-the-8, not by absolute layer id.

**Open optimization vector beyond M2.0**: even within the 8 layers, KV cache **per-channel / per-head FP4 with FP8 outliers**. Not all attention heads are equally sensitive. A "tiered" KV (FP4 for low-entropy heads, FP8 for high-entropy) could hit ~95 % of FP4 bandwidth gain at 99 % of FP8 accuracy. This is more sophisticated than the M2.0 ablation but lives in the same code path.

---

## 3. User query #3 — Linear-attention layer optimization (the deep one)

Yes, lightning ≈ Mamba/GLA family — **linear** in N, not quadratic. Algorithmic structure:

```
S_t = g_t · S_{t-1} + k_tᵀ v_t      # O(d²) state update
o_t = q_t · S_t                       # O(d²) output read
```

Here `S ∈ R^(d×d)` is fixed-size. **There is no seq-len axis to compress.**

### 3a. What this means for the two regimes

| Regime | Bottleneck | What helps lightning |
|---|---|---|
| **Prefill (compute-bound)** | The chunked algorithm `chunk_simple_gla` — Triton kernel doing block-sparse matmuls; cost ≈ `O(N · d²)` GEMM-equivalent | (i) larger chunk size for better tensor-core utilization (we already have `SGLANG_FLA_CHUNK_SIZE` per CHANGE_0080), (ii) FP8/FP4 GEMM in the chunk kernel itself, (iii) fusing q/k norm + RoPE + GLA prelude (we have `fused_qk_norm_rope`) |
| **Decode (memory-bound on weights, NOT on state)** | `fused_recurrent_simple_gla` — for each token: load `S` (d² words), do 2 small matmuls, store `S` back. Total per layer = O(d²) reads/writes. Across 24 layers + per-request batching, **state load/store traffic + weight load** dominates | (i) keep `S` in registers/SMEM across the whole layer (already via `fused_recurrent`), (ii) **quantize the `S` state itself** (BF16→FP8 → halves state BW), (iii) batch-merged kernel that does Q/K/V proj + recurrent update + O proj in one launch (kernel-fusion) |

### 3b. Algorithmic optimization vectors specific to linear attention

These are levers that DO NOT exist for standard attention:

1. **State quantization (BF16 state → FP8 state)** — biggest, untouched lever.
   - Each layer keeps `S ∈ R^(num_kv_heads × d × d)` in BF16/FP32. With `d=128` (typical), that's ~64 KB / layer / req. 24 layers × bs=24 ≈ 36 MB of state — fits in L2 (112 MB), but state load/store dominates the recurrent kernel inner loop.
   - FP8 state would halve that traffic. Accuracy impact is unknown but plausibly small because the state is heavily averaged via geometric decay `g`.
   - Implementation: modify `fused_recurrent_simple_gla` to accept a quantization scale per head. Probably a 1-2 day project on the kernel side.

2. **State re-materialization across requests / streaming-SSM tricks** — for very long-context decode, reset state cadence vs recompute trade-off. Our context is bounded by 128k so probably moot here.

3. **Chunk-size auto-tuning** — `chunk_simple_gla` performance is highly sensitive to chunk size relative to head_dim and SM count (96 SMs on SM120). CHANGE_0080 made chunk size a knob; we should sweep it on the official-style long-context speed dataset (NOT our local short-context one). Likely a 5-15 % prefill win at no accuracy cost.

4. **Chunk vs. recurrent crossover threshold tuning** — `_select_mode` in `SimpleGLAAttnBackend` chooses chunk mode for prefill, fused_recurrent for decode. There's a sweet spot for chunked extend with small new tokens (e.g., chunked-prefill where the new chunk is 64-256 tokens). Currently SGLang picks chunk mode any time `is_extend`. For very small extend chunks the overhead of chunk-launch may exceed recurrent cost — a runtime threshold could pick the cheaper path.

5. **Kernel-fusion: QKV-proj + GLA + O-proj in one Triton kernel** — for small-batch decode, the launch + register-fill overhead dominates. A single fused kernel `compute_lightning_layer(x_in, W_qkv, W_o, S_inout) -> x_out` would cut 3 kernel launches to 1. Significant rewrite (~1-2 weeks) but worth it given lightning is 24/32 of the layers.

6. **Matrix-form parallel decode (speculative-decoding's free friend)** — when decoding `g` tokens via speculative draft, lightning layers can absorb all `g` tokens with a single `chunk_simple_gla` call instead of `g` recurrent calls. Linear attention is the **ideal speculative-decoding target** — verification cost is amortized for free. This lifts speculative decoding from "hard win" to "easy win" specifically for our architecture.

7. **Output-gate / RMSNorm fusion** — we already have `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE`. Verify it's actually picked up on the v18 path; another small constant win.

### 3c. What does NOT help lightning (don't waste time)

- ❌ Sparse attention / topk attention — only meaningful for quadratic softmax attention.
- ❌ KV cache compression — there's no token-indexed KV.
- ❌ Flash-attention v2/v3 — those are for softmax attention.
- ❌ Sliding window — irrelevant since seq-len is already collapsed in `S`.

---

## 4. Concrete priority ranking for our codebase

Sorted by `(expected_gain × probability) / effort`, all on top of the v18 baseline:

| # | Optimization | Layers affected | Expected gain | Effort | Risk | Why now |
|---|---|---|---|---|---|---|
| **1** | **W4A8 FP8 activations (mxfp8 QMMA)** | All linear layers | **+30-50 % GEMM throughput** | 1-2 weeks (new Marlin kernel path) | Medium | SM120 has the hardware; current 296 TFLOPS FP8 unused |
| **2** | **M2.0 → M2.1 mixed FP8/FP4 KV** | 8 std-attn | +10-15 % decode TPS, less KV memory | Low (already proposed; ~50 LOC) | Low (per-layer ablation finds safe layers) | Already drafted |
| **3** | **Lightning state FP8 quantization** | 24 lightning | +5-10 % decode TPS (whole model) | Medium (kernel mod) | Medium-low | Untouched lever; biggest win on linear path |
| **4** | **Fused lightning layer (QKV + GLA + O) Triton kernel** | 24 lightning | +5-15 % latency at small bs | High (~2 weeks) | Medium | Reduces launch overhead at low bs |
| **5** | **Chunk-size sweep + crossover threshold** | 24 lightning | +3-7 % prefill | Low (env-var sweep) | Low | Cheap exploration |
| **6** | **Speculative decoding (n=2-3) leveraging linear-attn cheapness** | All | +20-40 % decode TPS at low bs | Medium (draft model + integration) | Medium | Long-tail mcq issue overlaps with this |
| **7** | **Per-head/per-channel tiered FP4 KV** | 8 std-attn | +5-10 % over uniform FP4 | Medium | Medium | Only after M2.0 lands |

**Recommendation**: stay on M2.0 (#2) since it's already drafted, then go to **#1 (W4A8 FP8) and #3 (lightning state FP8)** as the next two parallel work streams. Combined they could realistically bring us to mid-tier of the leaderboard. **#6 (speculative decoding)** is the only path to top-5 without major architectural changes, and our architecture makes it cheaper than usual.

---

## 5. Constraints and risks to track

- **Local vs official accuracy gap**: aim for ≥ 80 % local accuracy as safety margin (private set unknown).
- **Local vs official speed gap**: official speed dataset has more long-context samples; W4A8 / lightning-state-FP8 / speculative-decoding all help long-context decode the most.
- **Submission constraints**: ≤ 2 GB total, on-site quantization, ≤ 5 h total. New W4A8 kernel adds wheel size — re-measure after each implementation.
- **Accuracy-coefficient cliff**: keep `C ≥ 0.96` always; never sacrifice > 1 pt local accuracy for < 5 % speed gain.

---

## 6. Open questions (to resolve before next proposal)

1. Of #1 / #3 / #4 / #6, which does the user want me to draft as a formal proposal next?
2. Should #1 (W4A8) target the existing GPTQ Marlin path, or use the upstream `mxfp8` / `nvfp4` path (`sgl-kernel/`)?
3. For #3 (lightning state FP8), do we want a per-head static scale or a dynamic per-token scale? Static is much faster but accuracy-risky.
4. For #6 (spec dec): use a small MiniCPM-SALA-itself draft (n-gram or small layer subset) or a separate draft model? The latter eats submission budget.
