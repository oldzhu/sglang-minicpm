# PROPOSAL #2-A — `--torch-compile-max-bs` sweep on top of v21

**Date**: 2026-05-04
**Author**: Agent (awaiting user approval)
**Predecessor**: v21 (`SOAR_TIER1_LONG_CONTEXT=1` default-on, commit `edf97175e`)
**Catalog ref**: `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` Tier-2 entry
"torch.compile graph coverage at higher batch sizes"
**Risk**: low (env-gated, default-off)

## 1. Objective

v20/v21 ship `--torch-compile-max-bs 8`, meaning compiled CUDA graphs cover only
batch sizes ≤ 8. At Smax (max-running-requests=24, no concurrency cap) the
runtime falls back to eager mode for `bs ∈ [9, 24]`. Compiling those higher
batch buckets too should give double-digit % decode-throughput gains in the
Smax tier, where most of the official long-context speed score lives.

Hypothesis: bumping `--torch-compile-max-bs 8 → 16` (and possibly 24) at
v21's exact server-arg footprint extends graph coverage into the bs=9-16
range, which is where Smax decoding actually spends time.

## 2. Rule-compliance

- **Allowed**: `--torch-compile-max-bs N` is an upstream sglang flag; no
  custom kernel changes; no privately re-enabled prefix cache; no submission-
  template tampering.
- **Eval interface**: unchanged. Eval script `/root/data/eval_model_001.py`
  is untouched.
- **Concurrency tiers**: unchanged. We continue to honor `--max-concurrent
  {1,8,inf}` as the official harness sets them.
- **Submission constraints**: tarball stays ≤ 2GB; no extra weights.

## 3. Risk to accuracy / stability

| Risk | Mitigation |
|------|-----------|
| Compile time at server boot grows from ~214s to ~400-600s (more graphs to compile per bucket) | Acceptable: official launcher waits for `/health`. We will measure. |
| Higher static GPU memory (more captured graphs) → OOM at Smax | Default `--mem-fraction-static 0.84` should still fit; if OOM, fall back to bs=12 or revert. User noted bs=16/24 OOM'd previously — fcloud silicon/runtime has changed since. |
| torch.compile bucket bug at certain bs values silently corrupting outputs | Run accuracy eval **before** speed; require ≥ 78% (within Test 29 floor) to ship. |
| Compile cache invalidation on any sglang/torch upgrade | None for the submission packaging line — wheels are pinned. |

## 4. Files to change (tiny)

Only `benchmark/soar/demo_sala/prepare_env.sh` — replace the hardcoded
`--torch-compile-max-bs 8` with an env-gated value.

```bash
# In the gptq branch (~line 187):
SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs ${SOAR_TORCH_COMPILE_MAX_BS}"
```

- Default = 8 (v21 byte-equivalent).
- Set `SOAR_TORCH_COMPILE_MAX_BS=16` → ship as v22 candidate iff fcloud test passes.
- Set `=24` → optional ceiling probe.

(No other code changes; no model/kernel patches.)

## 5. Exact fcloud test plan

After user approval and one-line `prepare_env.sh` edit + push:

| # | Step | Expected |
|---|------|----------|
| 1 | `start-instance` | task started |
| 2 | `sync` | pull patch |
| 3 | `restart-server --env SOAR_TORCH_COMPILE_MAX_BS=16` | server up; expect boot 300-500s; verify cmdline shows `--torch-compile-max-bs 16` |
| 4 | `accuracy` | acc ≥ 78% (no regression vs Tier1-B 78.73%) |
| 5 | `speed --variant all` | record S1/S8/Smax |
| 6 | If OK: also `restart-server --env SOAR_TORCH_COMPILE_MAX_BS=24` + accuracy + speed | optional ceiling |
| 7 | `pause-instance` | done |

Total fcloud time budget: ~80 min for bs=16 alone, ~150 min if bs=24 also tested.

## 6. Success / failure matrix

| Outcome (vs Tier1-B baseline 78.73 / 111.36 / 40.49 / 33.62) | Decision |
|---|---|
| acc ≥ 78% AND **Smax ≤ 32s** (≥4% gain) | Ship as v22; mark #2-A complete. |
| acc ≥ 78% AND Smax in [32, 33.5] | Ship anyway (small but free). |
| acc ≥ 78% AND Smax ≥ 33.5 (no improvement) | Don't ship; document; move on to #3. |
| acc < 78% OR boot-time > 800s OR OOM | Revert env default; document failure mode in TEST_RESULTS_TRACKING. |

Note: local Smax inputs are short (≤1K tokens); decode at Smax is bs ~8-12
typically. So local sweep CAN measure this (unlike Tier 1 chunked-prefill).

## 7. Rollback

`export SOAR_TORCH_COMPILE_MAX_BS=8` (or unset) → v21 byte-equivalent. Single
env knob; trivial.

## 8. Next-step suggestions

- If #2-A wins: file #2-B = combined `bs=16` + raise `--max-running-requests`
  from 24 to 32 (more in-flight requests at Smax).
- If #2-A is neutral: pivot to #3 = prefill GEMM kernel direction (Marlin
  M=2048 tile path; per `R13e-prof-128k` profile data BF16 GEMM is 76.9%
  of 128K-context kernel time).
- If #2-A regresses: park torch-compile entirely as a Tier-2 dead-end and
  move #3 up.

## 9. Cross-references

- v21 base: commit `edf97175e`, [PROPOSAL_tier1_long_context_retest_20260430.en.md](PROPOSAL_tier1_long_context_retest_20260430.en.md), [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) rows `Tier1-A-baseline` and `Tier1-B-candidate`.
- Optimization catalog: [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).
- 128K profile evidence: `profile_data/round13e_analyze.txt` (BF16 GEMM 76.9%).

---

## Awaiting

Reply **approve** to apply the one-line `prepare_env.sh` edit, push, and run
the fcloud sequence in §5. Reply **adjust** with a different bs ceiling (or
to skip the bs=24 probe).
