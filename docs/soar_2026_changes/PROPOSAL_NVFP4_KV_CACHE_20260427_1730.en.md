# Proposal — NVFP4 KV cache as next optimization iteration

**Date**: 2026-04-27 17:30
**Status**: Proposal only. **No code changes yet.** Awaiting user approval per workflow rule.
**Predecessor**: `PHASE0_INT8_vs_FP8_SM120_20260427_1630.{en,zh}.md` killed W4-INT8, parked W4-FP8.

## 1. Objective and expected gain

Replace the current FP8 KV cache with **NVFP4-quantized KV cache** (storage-only compression of the K/V tensors in HBM).

- KV cache memory: **−50%** vs FP8 KV (4× smaller vs BF16 KV).
- KV bandwidth (decode = bandwidth-bound on KV reads): **−50%** vs FP8 KV.
- Expected end-to-end official-eval gain: **+8–15%** performance score (priority impact at long context / high concurrency where KV traffic dominates).
- **No compute change.** K/V are dequantized to BF16 before the attention matmul; matmul stays on BF16 tensor cores at 148 TF. The 593 TF FP4 tensor cores are NOT used (would require BOTH GEMM inputs in FP4).

## 2. Rule-compliance check

| Rule | Status |
|---|---|
| Submission size ≤ 2 GB | ✅ no impact (runtime quant) |
| On-site quantization (no pre-quantized weights) | ✅ KV is runtime |
| Total runtime ≤ 5 h | ✅ adds μs/token KV quant overhead |
| Forbidden tricks (private prefix cache during eval) | ✅ none |
| Concurrency flag respect | ✅ orthogonal |
| Accuracy ≥ 97% normalized for C ≠ 0 | ⚠️ MUST validate; KV quant has known sensitivity |

## 3. Risk to accuracy / stability

- **Primary risk**: NVFP4 KV at sub-7B-class models can lose 1–3 normalized accuracy points if outliers in K (especially high-magnitude rotary-encoded keys) are clipped. Mitigation:
  - Per-block scale (16-element block-fp4 with bf16 scale) — already standard NVFP4 layout.
  - Optional: keep first/last N tokens or first attention layer in FP8 KV (hybrid) — matches the latest weekly champion combo described as "mixed NVFP4 + FP8 KV".
  - **Validate against `perf_public_set.jsonl` before any official submission.** Target normalized > 99% (C=1.0). If drops to 97–99%, decide whether the speed gain still wins (×0.92 / ×0.96 vs ×1.0).
- **Stability**: KV cache layout change is local to attention backend; no cudagraph implications if the backend supports it cleanly.

## 4. Implementation plan (before change)

### 4.1 Reference implementations to study
- vllm `csrc/quantization/fp4/` (offline NVFP4 weight quant; not directly applicable but shares the e2m1 + block-scale data layout).
- `sglang.srt.layers.attention.flashattention` → KV dtype config: today supports `auto / fp8_e5m2 / fp8_e4m3 / nvfp4`. Verify the NVFP4 path exists end-to-end (it may be partial).
- Check `sgl-kernel/csrc/attention/` for existing NVFP4 KV dequant in attention backend (Phase 0 of this proposal).

### 4.2 Phase split
| Phase | Goal | Effort |
|---|---|---|
| **P1 — survey** | Read sglang FlashAttention KV-dtype path; confirm NVFP4 already plumbed or identify the gap | 1 d |
| **P2 — plumb** | Wire `--kv-cache-dtype nvfp4` end-to-end through sglang launcher → attention backend → cudagraph capture | 2–3 d |
| **P3 — accuracy validate** | Run accuracy eval; iterate on outlier handling (per-block scale dtype, optional first-layer FP8 fallback) | 2–3 d |
| **P4 — speed validate** | S1/S8/Smax on fcloud; confirm gain | 1 d |
| **P5 — hybrid (optional)** | First-layer / sink-token FP8 + rest NVFP4 if pure NVFP4 falls short of 99% | 2 d |
| | **Total** | **6–10 days** |

### 4.3 Files likely to touch
- `python/sglang/srt/configs/model_config.py` — add `nvfp4` to `kv_cache_dtype` validation
- `python/sglang/srt/layers/attention/flashattention_backend.py` — KV dequant on read
- `python/sglang/srt/layers/radix_attention.py` — KV write quantization
- `sgl-kernel/csrc/attention/` — possibly extend FlashAttention kernel for NVFP4 K/V dequant
- `benchmark/soar/demo_sala/prepare_env.sh` — switch `--kv-cache-dtype fp8_e5m2` → `nvfp4`

## 5. Validation commands

### Accuracy
```bash
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
```
**Pass criterion**: normalized accuracy > 99% (C=1.0). 97–99% acceptable only if speed gain compensates ×0.92/×0.96 hit.

### Speed
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```
**Pass criterion** (vs v18 baseline S1=121.71 / S8=44.09 / Smax=35.86): improvement on S8 and Smax expected; S1 small or neutral.

## 6. Result summary table (template — fill after test)

| Run | Config | S1 (s) | S8 (s) | Smax (s) | ori_acc | norm_acc | C |
|---|---|---|---|---|---|---|---|
| Baseline (v18) | W4A16 + FP8 KV | 121.71 | 44.09 | 35.86 | 79.29 | 99.11% | 1.0 |
| NVFP4 KV (P2) | W4A16 + NVFP4 KV | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid (P5) | W4A16 + NVFP4 KV + FP8 first-layer | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. Rollback instructions

```bash
# revert prepare_env.sh
git checkout HEAD -- benchmark/soar/demo_sala/prepare_env.sh
# revert sglang source if changed
git checkout HEAD -- python/sglang/srt/layers/attention/ python/sglang/srt/configs/
# resync to fcloud
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

## 8. Next-step suggestions (after this iteration)

If NVFP4 KV succeeds (normalized > 99%, +8–15% speed):
- Revisit the parked **W4-FP8 dense GEMM** (Option A from `PROPOSAL_W4A8_REAL_001`).
- Consider sparse attention pattern tuning (next item in `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`).

If NVFP4 KV fails accuracy (drops below 97%):
- Try hybrid (P5) before abandoning.
- If hybrid still under 97%, fall back to FP8 KV and pursue W4-FP8 spike (`PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730`).

## 9. Cross-references

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.en.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md)
- [ANALYSIS_nvfp4_offline_quant_20260427_0857.en.md](ANALYSIS_nvfp4_offline_quant_20260427_0857.en.md) — note: this prior analysis was about NVFP4 *weight* quant (which broke accuracy). KV quant is a different, milder application.
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- WeChat reference: latest weekly champion combo = "W4A16 GPTQ + mixed NVFP4/FP8 KV cache + others".
