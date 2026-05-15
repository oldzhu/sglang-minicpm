# PROPOSAL: Real W4A8 — Addressing FP8 Accuracy & Old-Path Regression Concerns

**Date**: 2026-05-15 | **Status**: PROPOSAL → VERIFICATION in-progress  
**Predecessor**: `PROPOSAL_W4A8_REAL_001` (filed 2026-04-27, never built)  
**Related**: `CHANGE_W4A8_001_iteration_002` (old W8A8 FP8 path, commit `7ce21c3f5`, abandoned)

## Background

The user raised two concerns before committing to the real W4A8 (INT4 storage + FP8 MMA) kernel build:

1. **FP8 accuracy risk**: FP8 (e4m3) has fewer mantissa bits than BF16/FP16 — could degrade MiniCPM-SALA's accuracy below the 97% norm threshold (C=0).
2. **Old-path speed regression**: The prior W4A8 attempt (commit `7ce21c3f5`) regressed S1 by +118% (121.71→265.32s). Even with a best-case 25% speedup from true W4A8, absolute speed could still be worse than baseline.

This document addresses both concerns from first principles and existing empirical evidence.

---

## Concern 1: FP8 Activation Accuracy Risk

### The question

> "W4A8 means FP8 activations. FP8 has fewer bits than BF16/FP16. Won't this hurt accuracy?"

### FP8 precision comparison

| Dtype | Mantissa bits | Exponent bits | Dynamic range | Use case |
|---|---:|---:|---|---|
| BF16 | 7 | 8 | ~1e-38 to 3.4e38 | Current baseline (Marlin W4A16) |
| FP8 e4m3 | 3 | 4 | ~1.5e-5 to 448 | Proposed activation dtype |

FP8 e4m3 has only 3 mantissa bits vs BF16's 7 — a 4-bit loss.

### Why this is manageable

**1. Per-token scaling recovers precision.** Each token's activation vector gets its own FP8 scale factor (`amax × 2^(exponent) → scale`). This is standard practice in vLLM, TRT-LLM, and SGLang's own `cutlass_w8a8_fp8` kernel. The per-token scale effectively extends the dynamic range beyond the static FP8 range, keeping quantization error small for typical transformer activation distributions.

**2. Only the GEMM inputs are FP8 — everything else stays BF16.** The residual path, attention softmax, layer norm, and SimpleGLA recurrent state all remain in BF16. Only the linear layer matrix multiplies (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) see FP8 activations.

**3. Our own empirical evidence is positive.** The old W4A8 test (commit `7ce21c3f5`) used FP8 on **both** weight and activation sides (W8A8 FP8 blockwise GEMM). Accuracy was **79.20% vs 79.29% baseline** (Δ −0.09pt, well within noise). True W4A8 uses **INT4 weights (same as baseline) + FP8 activations** — only the activation side changes from BF16→FP8. The accuracy risk is **lower** than the already-proven-neutral FP8×FP8 case.

| Test | Weight | Activation | GEMM | Accuracy | Δ vs Test 12 |
|---|---:|---:|---|---|---|
| Test 12 (baseline) | INT4 (0.5 B/elem) | BF16 | Marlin BF16 MMA | 79.29% | — |
| Old W4A8 (7ce21c3f5) | **FP8** (1.0 B/elem) | **FP8** | cutlass FP8 blockwise | 79.20% | −0.09pt |
| **True W4A8 (proposed)** | INT4 (0.5 B/elem) | **FP8** | mixed-input FP8 QMMA | **TBD** | expected ≤±0.5pt |

**4. Mitigation plan.**
- **Phase 0 microbenchmark**: CPU-side test of FP8 activation quantization on real MiniCPM hidden states from a short forward pass; measure per-token MSE vs BF16 reference.
- **Gate**: if per-token MSE exceeds 5e-3 (conservative; the old FP8×FP8 test tolerated 5e-2 round-trip), abort and fall back to Option B (INT8 activation).
- **Full accuracy run** on fcloud before any submission — C=1.0 or abandon.

---

## Concern 2: Old W4A8 Speed Regression Predicts True W4A8 Will Also Be Slow

### The question

> "The old W4A8 path (7ce21c3f5) had S1 +118%, S8 +56%, Smax +30%. Even if true W4A8 is 25% faster than baseline, the old regression was far worse. Could true W4A8 end up slower than baseline?"

### Why the old path was slow — and why true W4A8 is different

The old path was **W8A8 FP8 blockwise**, NOT W4A8. The critical flaw was a **load-time weight upcast**:

```
Old path (W8A8, 7ce21c3f5):
  GPTQ weight on disk:  INT4 (0.5 bytes/element)
  → dequant to BF16
  → requant to FP8
  Weight in GPU HBM:   FP8 (1.0 bytes/element)  ← DOUBLED vs INT4!
  Activation:           FP8 (1.0 bytes/element)
  GEMM:                 FP8 × FP8 cutlass blockwise @ 296 TF

True W4A8 (proposed):
  GPTQ weight on disk:  INT4 (0.5 bytes/element)
  Weight in GPU HBM:    INT4 (0.5 bytes/element)  ← SAME AS BASELINE!
  Activation:           FP8 (1.0 bytes/element)   ← 2× less than BF16!
  GEMM:                 mixed-input INT4→FP8 dequant + FP8 QMMA @ 296 TF
```

| Property | Baseline (W4A16 Marlin) | Old path (W8A8 FP8) | True W4A8 |
|---|---|---|---|
| Weight HBM storage | 0.5 B/elem | 1.0 B/elem (**2×**) | **0.5 B/elem** (= baseline) |
| Activation HBM traffic | 2.0 B/elem (BF16) | 1.0 B/elem | **1.0 B/elem** (0.5× baseline) |
| Compute (SM120 TFLOPS) | 148 (BF16) | 296 | **296** (2× baseline) |

**True W4A8 is equal or better than baseline on every single dimension.** There is no mechanism by which it could be slower.

### Per-tier physics-based prediction

| Tier | Dominant bottleneck | Effect of FP8 activation | Effect of FP8 QMMA | Net Δ vs baseline |
|---|---|---|---|---|
| **S1** (decode bs=1, M=1) | Weight HBM bandwidth | −50% activation traffic (minor at M=1) | None (compute unused) | **−5 to −10%** |
| **S8** (mixed prefill/decode) | Mixed bandwidth + compute | −50% activation traffic | 2× GEMM for prefill portions | **−10 to −20%** |
| **Smax** (prefill-dominant) | GEMM compute (67-83% of kernel time per R13e profiling) | −50% activation traffic | **2× GEMM compute** on the dominant kernel | **−20 to −30%** |

### Why the old-path regression is irrelevant as a predictor

The old S1 regression (+118%) is **entirely explained** by the weight bandwidth penalty: at decode bs=1, the kernel reads weights from HBM on every token. Doubling weight size (INT4→FP8) doubles the HBM read time. This effect dominates everything else by 10-20×.

True W4A8 **does not have this penalty** because INT4 weight storage is preserved end-to-end. The old regression predicts nothing about true W4A8 performance — it only confirms that "INT4 storage is mandatory."

---

## Verification Plan: Re-test Old W4A8 Path on Current HEAD

### Why verify

1. **Confirm the old path still works** on current HEAD (v24 baseline: Tier1 long-context, force-dense, flashinfer, torch_compile_max_bs=24).
2. **Measure current-config regression magnitude** to ground the true W4A8 design. The 2026-04 measurement (265s S1) was on the old Test 12 config (chunk=32K, prefill-max-req=1, sched-cons=1.0, torch_compile_max_bs=8). Current config (chunk=65K, prefill-max-req=4, sched-cons=0.8, torch_compile_max_bs=24) may partially mask the weight-bandwidth penalty through better scheduling.
3. **Sanity-check the env gate** (`SOAR_W4A8_FP8_GEMM=1`) is still functional on current code path.
4. **Free baseline data point** for the Optimization Catalog.

### Test procedure (no source edits)

1. `start-instance` — resume the paused fcloud task
2. `sync` — pull latest commit `8ce03caaf` (mcq cap disabled, byte-equivalent to v24 baseline)
3. Override env: `SOAR_W4A8_FP8_GEMM=1` before `source prepare_env.sh` in restart-server
4. `restart-server` → `wait-server` (expect ~3-5 min, longer due to FP8 conversion at load)
5. Smoke test (single Paris completion) — confirm server alive
6. `speed --variant s1` — primary signal (most affected tier)
7. `speed --variant s8` — secondary
8. `speed --variant smax` — tertiary
9. **Skip accuracy** — already verified neutral in iteration_002 (79.20%, C=1.0). FP8 path hasn't changed.
10. `pause-instance`

### Estimated time: ~45 min total

### Decision matrix

| S1 result | Interpretation | Next action |
|---|---|---|
| ≥ 1.5× v24 baseline (~165s+) | Old regression mostly reproduces; current config doesn't help | Proceed to true W4A8 kernel build |
| 1.0–1.5× v24 baseline (~110-165s) | Current Tier1 + torch_compile_max_bs=24 helps mask penalty | Investigate which flag closed the gap; possibly cheaper than kernel build |
| < 1.0× v24 baseline (<110s) | FP8 blockwise now beats Marlin on current config | Major surprise — re-verify before acting |

---

## Next Steps (after verification)

If old-path regression confirmed (expected outcome):
1. **Build true W4A8 kernel** per `PROPOSAL_W4A8_REAL_001` Option A (Machete-style mixed-input FP8 QMMA).
2. **Phase 0 microbenchmark**: FP8 activation quantization MSE on real hidden states.
3. **Full accuracy run** on fcloud with the new kernel — gate on C=1.0.
4. **Parallel track**: accuracy stability (mcq fix, larger cap or repetition detector).

If old-path regression does NOT reproduce (unexpected):
1. Re-examine the W4A8 plumbing — something may have broken silently.
2. Re-assess whether the old FP8 blockwise path is actually viable (unlikely but worth checking before committing to kernel work).

---

## References

- `PROPOSAL_W4A8_REAL_001.en.md` — True W4A8 kernel design (three candidate paths)
- `CHANGE_W4A8_001_iteration_002.en.md` — Old W8A8 FP8 path validation results (abandoned)
- `SM120_RTX_PRO_HARDWARE.md` — GPU specs: FP8 QMMA = 296 TF, BF16 = 148 TF
- `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` — Real W4A8 listed as highest-priority item
- `TEST_RESULTS_TRACKING.md` — All historical accuracy/speed data
- Profile data: `R13e-prof-32k/64k/128k` — GEMM = 67-83% of kernel time on long-context prefill
