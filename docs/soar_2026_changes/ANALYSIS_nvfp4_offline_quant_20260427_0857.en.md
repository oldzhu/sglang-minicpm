# Analysis — Offline NVFP4 weight quantization as a future optimization vector

**Date**: 2026-04-27 08:57
**Context**: Follow-up to `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md`. User asked: "If we do offline NVFP4 quantization instead of offline INT4, will we get better performance? Could it be one of the improving choices after the recommended ones?"
**Status**: Analysis only. No code change. Outcome: rank NVFP4 as a **deferred stretch goal (#5)**, not a primary next iteration.

---

## 1. The theoretical upside

SM120 (RTX PRO 6000 Blackwell) tensor-core peaks:

| Path | Peak TFLOPS |
|---|---|
| BF16 HMMA (today's GPTQ + Marlin) | 148 |
| FP8 QMMA (W4A8 / mxfp8) | 296 |
| **FP4 QMMA (NVFP4 / MXFP4)** | **593** |

If weights and activations could go through native FP4 QMMA, GEMM peak throughput would be **4× the current path**. On long-context, prefill-bound workloads (which is what the official speed dataset is), GEMM is a real bottleneck — so the upside is genuine on paper.

## 2. The empirical downside (we already paid for the lesson)

- **Test 21 already attempted offline NVFP4 quantization on this exact model.**
- Result: accuracy collapsed to ~12 % (catastrophic; far below the 97 % cliff that gives C = 0).
- This is empirical evidence that **naive post-training NVFP4 breaks MiniCPM-SALA**.

## 3. Why NVFP4 broke where GPTQ INT4 worked

| Factor | GPTQ INT4 (today) | NVFP4 (Test 21) |
|---|---|---|
| Encoding | Uniform integer, 16 levels | Floating-point (1 sign + 2 exp + 1 mantissa), 16 levels with logarithmic spacing |
| Scale granularity | Per-group BF16 scale, group_size=128 | Single shared block-scale per ~16-element block |
| Suitability for LLM weights | Mean-zero, Laplacian-like → uniform grid + small group fits well | Logarithmic spacing wastes precision near zero where most weight mass lives |
| Calibration sensitivity | Tolerant; off-the-shelf GPTQ works | Requires rotation transforms (QuaRot / SpinQuant) + outlier handling, otherwise sub-7B models lose >5 pts on MMLU-class tasks |
| Recurrence-friendliness | Errors don't compound across decode steps | 24 lightning layers' state evolves multiplicatively → tiny per-layer FP4 errors compound across 1000+ decode steps |

## 4. Other practical obstacles

1. **Lightning kernels don't natively consume FP4.** Even if weights become FP4, the SimpleGLA Triton kernels need an FP4-aware rewrite. Marlin upstream has W4A4 prototypes, but they target FP4-act / INT4-weight, not what we want.
2. **Submission-size win is marginal.**
   - GPTQ INT4 with group_size=128 + BF16 scale → ~4.125 bits/element.
   - NVFP4 with shared block-scale per 16 elements → ~4.5 bits/element (FP8 block-scale even higher).
   - So NVFP4 is **NOT smaller** — actually slightly larger. No gain on the 2 GB cap.
3. **Calibration cost.** Real recovery requires QuaRot / SpinQuant rotation pre-quant + per-layer NVFP4 calibration. That's a multi-day cycle, comparable to building the GPTQ + sparse_qkv pipeline.
4. **Risk to C coefficient.** Anything below 97 % accuracy zeroes the score. Past Test 21 result is direct evidence that this risk is real and large for NVFP4 on this model.

## 5. Updated optimization priority list

| Rank | Vector | Expected gain | Risk | Effort |
|---|---|---|---|---|
| 1 | W4A8 (FP8 activations + INT4 weights via Marlin W4A8 path) | ~1.5-2× on prefill GEMM, captures FP8 QMMA at 296 TF | Medium | High |
| 2 | Lightning state FP8 (proposal #3) | 10-20 % on long-context decode | Medium | Medium |
| 3 | Fused lightning kernel (proposal #4) | 5-15 % on decode | Low | Medium |
| 4 | Speculative decoding (proposal #6) | 1.3-2× on S₁ | Low | Medium-High |
| **5** | **NVFP4 weights (deferred)** | **~4× GEMM peak in theory, but Test 21 failed at 12 % accuracy** | **Very high** | **High (QuaRot/SpinQuant + multi-day calibration)** |

## 6. Recommendation

Treat NVFP4 as a **stretch goal AFTER W4A8 succeeds**, not as a primary next iteration:

- **Step 1 (proposal #1)**: Get W4A8 working. Captures ~2/3 of the achievable headroom (148 → 296 TFLOPS) at significantly lower accuracy risk, because activations stay quantized but weights remain proven INT4.
- **Step 2 (only if W4A8 succeeds AND time/budget remains)**: Reconsider NVFP4 with proper machinery — QuaRot rotation, careful per-layer calibration, lightning-kernel FP4 retrofitting. Budget separately as a **multi-day calibration cycle** with full accuracy regression tracking.
- **Do NOT** swap GPTQ INT4 → NVFP4 as the *primary* next iteration. Test 21 paid for that lesson once already.

## 7. Decision summary

| Question | Answer |
|---|---|
| Could NVFP4 give better performance than INT4? | Theoretically yes (~4× GEMM peak vs current). |
| Is it a viable next-iteration choice? | No. Test 21 already showed catastrophic accuracy collapse with naive PTQ NVFP4 on this model. |
| Where does it sit in the priority list? | Vector #5, deferred until after W4A8 (#1) is validated. |
| What would have to change for it to become viable? | (a) Add rotation transforms (QuaRot / SpinQuant); (b) FP4-aware lightning kernels; (c) multi-day calibration with regression tracking; (d) accept material risk to C coefficient. |

---

## 8. Cross-reference

- Prior research note: `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md`
- Quantization-stack clarification: `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md`
- Failed NVFP4 attempt: Test 21 in `TEST_RESULTS_TRACKING.md`
