# PROPOSAL — NVFP4 KV Cache Dense-Mode Smoke Test

**Date**: 2026-04-30
**Status**: PROPOSAL — awaiting user approval
**Predecessors**:
- `SURVEY_NVFP4_KV_P1_20260428_1130.en.md` (P1 plumbing survey, completed)
- `CHANGE_0131_nvfp4_kv_p2_plumbing.{en,zh}.md` (env-flag plumbing already in tree)
- `CHANGE_0132_nvfp4_kv_force_dense_compat.{en,zh}.md` (dense-compat fixes already drafted)
- `PROPOSAL_iteration_M20_kv_fp4_ablation.{en,zh}.md` (per-layer ablation — DEFERRED to a follow-up)
**Sibling iteration**: `PROPOSAL_tier1_long_context_retest_20260430.{en,zh}.md` (run BEFORE this one)

## 1. Background

NVFP4 (MXFP4, `fp4_e2m1`) is natively supported by SM120 (Blackwell, RTX 6000D). Survey P1 found:

- Plumbing already 80 % in tree: `SOAR_FP4_KV_CACHE=1` env flag exists in `prepare_env.sh` (CHANGE_0131); `MHATokenToKVPoolFP4` is upstream sglang.
- 4 gates inside our custom MiniCPM attention backend currently switch on `kv_cache_dtype_str.startswith("fp8")` and need to extend to `fp4_e2m1` — surveyed in detail.
- `--force-dense-minicpm` bypasses 2 of those gates (the sparse-only ones). Dense-only smoke is the cleanest first signal.

KV cache memory savings: **FP8 → FP4 ≈ 44 % reduction** of KV cache bytes per token. On the long-context speed set (68 % inputs 32K–512K) this directly attacks the Smax tier where competitors are running away from us, by allowing larger concurrent batches.

## 2. Objective

Run the simplest possible NVFP4-KV configuration end-to-end on fcloud:

- `SOAR_FP4_KV_CACHE=1` (env flag already plumbed in `prepare_env.sh:155`)
- `--force-dense-minicpm` (already default in v20)
- All other Tier 1 gates inherited from whichever baseline #1 ships (v20 or v21).

Three questions to answer in this iteration:

1. **Boot**: does the server start cleanly with FP4 KV?
2. **Accuracy**: does end-to-end accuracy stay above the safety threshold?
3. **Speed**: is there a measurable Smax improvement (S1 / S8 may be flat)?

## 3. Rule compliance

- No model weight changes. No on-site quantization changes (same GPTQ pipeline).
- Submission package format unchanged; opt-in via env var.
- KV is computed at runtime — fully on-site, fully reproducible, fully Apache-2.0.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Boot failure due to `MiniCPMAttentionBackend` gates not extended for FP4 | Survey identified 4 gates; CHANGE_0132 already drafted the dense-compat fixes — apply them BEFORE the smoke run. Verify by `grep -n 'fp4_e2m1' python/sglang/srt/layers/attention/minicpm_backend.py`. |
| Accuracy regression > 5 pt vs FP8 KV baseline (FP4 has only 1 mantissa + 2 exp + sign) | Hard threshold: **acc_ori ≥ 75 %** and normalized ≥ 97 % (catalog C-tier floor). If below, kill the iteration immediately. We currently have ~5 pt of accuracy margin (80 %). |
| `KVFP4QuantizeUtil` uses `@torch.compile` — cudagraph capture may break | Survey flagged this; if cudagraph capture fails, drop `--enable-torch-compile` for this iteration only and re-test (slower but isolates the issue). |
| Smax OOM from larger admitted batch sizes | Same `mem-fraction-static 0.84`; KV is *smaller* with FP4, so OOM risk is actually lower. |
| `set_kv_buffer` double-scaling K via `layer.k_scale` (Survey Gap D) | Verify `layer.k_scale` is None when `kv_cache_dtype == "fp4_e2m1"`; else null it out for the smoke test. |

## 5. Files to change

| File | Change |
|---|---|
| `python/sglang/srt/layers/attention/minicpm_backend.py` | Extend the 4 gates surveyed in P1 to also fire for `fp4_e2m1` (Survey Gaps A, C, D). Reference CHANGE_0132 for the exact diff. |
| `benchmark/soar/demo_sala/prepare_env.sh` | No code change — `SOAR_FP4_KV_CACHE` already plumbed. The runner exports it. |
| (no model preprocessing change) | KV is per-step; no weight re-quantization needed. |

## 6. Validation commands

```bash
# Pre-flight: verify FP4 gate fixes are present
ssh fcloud "grep -nE 'fp4_e2m1|kv_cache_dtype_str' /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py | head -20"

# Sync code
python3 scripts/fcloud/fcloud_workflow.py sync

# Baseline guard (FP8 KV — same baseline you ran for #1's step A on the same day)
# (skip if just ran in #1)

# FP4 KV candidate
ssh fcloud "cd /root/submission_sim && export SOAR_FP4_KV_CACHE=1 && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 1) Boot check — server log clean? (no exceptions, no NaN)
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 200

# 2) Accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 3) Speed
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

Total fcloud time estimate: ~30 min (1 × accuracy + 1 × speed).

## 7. Success / failure criteria

| Outcome | Acc | S1 | S8 | Smax | Decision |
|---|---|---|---|---|---|
| **WIN** | ≥ 78 % (≤ 2 pt drop) | tie ± 3 % | tie ± 3 % | ≤ 0.95 × baseline | Ship as next version. Open the per-layer ablation iteration M2.0. |
| **PARTIAL ACCURACY HOLD** | 76–78 % | tie ± 3 % | tie ± 3 % | ≤ 0.95 × baseline | Hold; investigate per-layer ablation (M2.0) to recover 1–2 pt acc before shipping. |
| **ACCURACY REGRESSION** | < 76 % normalized < 97 % | — | — | — | Park NVFP4 KV until per-layer ablation done. Document. |
| **SPEED REGRESSION** | ≥ 78 % | — | — | ≥ 1.05 × baseline | Surprising; check for FP4 dequant overhead in attention prefill kernel; abandon if root cause non-trivial. |
| **BOOT / CUDAGRAPH FAIL** | — | — | — | — | Document gate that broke; if cudagraph-related, retest without `--enable-torch-compile`. |

(Baseline = same-day FP8-KV run from the v20-or-v21 chosen baseline.)

## 8. Rollback

`unset SOAR_FP4_KV_CACHE` and re-source `prepare_env.sh`. The MiniCPM-backend gate fixes are no-ops on FP8 paths (they extend a string match), so leaving them in place is safe even after rollback.

## 9. Next-step suggestions

- **WIN** → open `PROPOSAL_iteration_M20_kv_fp4_ablation` for per-layer sensitivity ablation (already drafted) — this is how we recover any residual accuracy loss while keeping the memory savings.
- **WIN + #1 already won** → submit a combined v22 = Tier 1 + NVFP4 KV. Update leaderboard memory.
- **REGRESSION** → return to catalog and pick #3 candidate (NOT FP8 W8A16 — that path is closed).

## 10. Open question for user

There's an existing `PROPOSAL_iteration_M20_kv_fp4_ablation.{en,zh}.md` from a previous session that goes straight to per-layer sensitivity. **This proposal is a strictly simpler, dense-only smoke test that should run FIRST.** If the dense-only smoke reveals catastrophic accuracy loss, the per-layer ablation is the natural follow-up. Confirm this ordering before we proceed.
