# PROPOSAL — Tier 1 Scheduler Config Retest on New Long-Context Speed Set

**Date**: 2026-04-30
**Status**: PROPOSAL — awaiting user approval
**Predecessor**: `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` Tier 1 row "best so far"
**Sibling iteration**: `PROPOSAL_nvfp4_kv_dense_smoke_20260430.{en,zh}.md` (run AFTER this one)

## 1. Background

The catalog records a Tier 1 scheduler configuration that beat the v18 baseline on the OLD speed dataset:

| Config | S1 | S8 | Smax |
|---|---|---|---|
| v20 shipped (`pmr=1`, `sc=1.0`, `chunk=32K`) | 121.71 | 44.09 | 35.86 |
| Tier 1 best (`pmr=4`, `sc=0.8`, `chunk=65K`) | 110.54 | 40.54 | 33.59 |

This was a **−8.2 % S1** win on the old (mostly-short) dataset. After the official speed set was updated on 2026-04-15 to be **prefill-dominant (68 % of inputs 32K–512K)**, the v20 submission rolled BACK to `pmr=1, sc=1.0, chunk=32K`. There is no documented official measurement of the Tier 1 best config against the new long-context dataset.

On a long-context, prefill-dominant workload the Tier 1 levers are *more* leveraged, not less:

- `chunked-prefill-size 65536` doubles per-step prefill throughput when input ≥ 64K (one fewer chunk per request).
- `prefill-max-requests 4` allows 4 concurrent in-flight prefills, helping S8 / Smax tiers where many long inputs queue up.
- `schedule-conservativeness 0.8` admits more requests earlier, raising tier utilization.

## 2. Objective

Measure whether the Tier 1 best config retakes its old-data lead on the new long-context speed set, and (if so) ship it as v21.

**Expected gain (estimate)**: −5 % to −15 % across S1 / S8 / Smax. Bigger swing on Smax than on the old data, because more chunks per request amortize differently.

## 3. Rule compliance

- No model changes, no quantization changes, no kernel changes, no submission-format changes.
- Pure scheduler tuning that the official launcher already honors via `SGLANG_SERVER_ARGS` in `prepare_env.sh`.
- Within the official 5 h run budget. No new dependencies.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Long-context Smax OOM at `pmr=4 / chunk=65K` (more concurrent KV) | `mem-fraction-static 0.84` unchanged; monitor server log; fall back to `pmr=2 / chunk=65K` if OOM |
| Accuracy regression from greedier scheduling | Accuracy path is independent of these flags; expect identical acc to v20 (full eval still run as guard) |
| Local↔official ratio inversion | We report local numbers, but acknowledge official long-context dataset is even more prefill-heavy → the win should be ≥ local |

## 5. Files to change

| File | Change |
|---|---|
| `benchmark/soar/demo_sala/prepare_env.sh` | Add env-flag `SOAR_TIER1_LONG_CONTEXT` (default OFF). When `=1`, override `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --schedule-conservativeness 0.8` in the gptq branch only. |
| (no other file changes) | |

The env flag preserves v20 as the default so a fresh checkout still reproduces the shipped baseline.

## 6. Validation commands

On fcloud, after `python3 scripts/fcloud/fcloud_workflow.py sync`:

```bash
# A) Baseline guard (sanity — should match Test 12 numbers)
ssh fcloud "cd /root/submission_sim && unset SOAR_TIER1_LONG_CONTEXT && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# B) Tier 1 candidate
ssh fcloud "cd /root/submission_sim && export SOAR_TIER1_LONG_CONTEXT=1 && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # safety guard
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

Total fcloud time estimate: ~45 min (2 × accuracy ~10 min + 2 × speed ~15 min).

## 7. Success / failure criteria

| Outcome | S1 | S8 | Smax | Acc | Decision |
|---|---|---|---|---|---|
| **WIN** | ≤ 0.97 × baseline | ≤ 0.97 × baseline | ≤ 1.00 × baseline | ≥ 79 % | Ship as v21. Update catalog row. |
| **PARTIAL** | ≤ 0.97 × baseline | tie/regress < 2 % | tie/regress < 2 % | ≥ 79 % | Hold. Re-evaluate after #2 NVFP4 result. |
| **REGRESSION** | regress ≥ 2 % on any tier | — | — | — | Document and abandon Tier 1; mark catalog row "verified regression on long-context". |
| **OOM / crash** | — | — | — | — | Try `pmr=2 / chunk=65K` fallback once; if still bad, abandon. |

(Baseline = the same fcloud-instance `SOAR_TIER1_LONG_CONTEXT=0` run from step A on the same day, NOT the historical Test 12 numbers — fcloud disk I/O variance is real.)

## 8. Rollback

`unset SOAR_TIER1_LONG_CONTEXT` and re-source `prepare_env.sh`. No source-code state change. The git revert is a single edit to `prepare_env.sh`.

## 9. Next step after this iteration

- If WIN → ship v21 → run #2 NVFP4 KV proposal *on top of* the new baseline.
- If REGRESSION → keep v20 → run #2 NVFP4 KV proposal *on the v20 baseline*.

Either way, #2 is the next test regardless of outcome here.
