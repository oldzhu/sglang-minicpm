# PROPOSAL: Revised Iteration A — Runtime / Server-Arg Speed Tuning

**Date**: 2026-04-23
**Supersedes**: Iteration A (Marlin SM120 decode tile specialization) in [STRATEGIC_ROADMAP_TOP5.en.md](STRATEGIC_ROADMAP_TOP5.en.md)
**Status**: PROPOSAL — awaiting user approval before any code change

---

## 1. Why the original Iteration A (Marlin tile specialization) must be abandoned

The deep analysis in [CHANGE_0125_sm120_marlin_tiles_001.en.md](CHANGE_0125_sm120_marlin_tiles_001.en.md) proved that adding more Marlin kernel tile instantiations cannot help MiniCPM-SALA on SM120:

1. **Scorer physics**: The SM120 Marlin scorer's `fill_ratio × 1000` term correctly prefers narrow `thread_n=64` tiles because they produce 72–448 tiles for MiniCPM-SALA's `N ∈ [1024, 28672]` shapes, filling all 96 SMs. Wide `thread_n=256` tiles produce only 18–112 tiles, leaving 78+ SMs idle. The scorer is mathematically correct — this is not a tuning bug.
2. **Wrong instruction set**: Marlin uses SM80-era `mma.sync.aligned.m16n8k16`. Tile reshuffling cannot access SM120 native warp-level MMA, TMA, or QMMA. The 2× hardware advantage is unreachable through this path.
3. **Empirical confirmation**: Test 27 (CHANGE_0125) added tiles and measured zero speed change on all three tiers.

**Implication**: Extracting real gain from SM120 requires replacing Marlin, not reshuffling it. That is a multi-week effort (CUTLASS SM120 integration, or SM120-native MMA rewrite), outside this iteration's scope.

## 2. Revised Iteration A — scope and goal

**Goal**: extract remaining free throughput from the existing kernel stack via server-arg / runtime tuning, **without kernel edits**.

**Expected gain**: S1 2–5%, S8 2–5%, Smax 0–3% (modest, but free, deterministic, rollbackable).

**Expected C impact**: neutral (config-only changes, same math).

**Leverage on final score**: if speed improves 4%, performance score rises ~4% multiplicatively → roughly +1.6 final points on the current team-beta score (39.62 → ~41.2). Modest but real, and it unblocks the path to Iteration B (speculative decoding) by establishing a clean runtime baseline.

## 3. Changes proposed (bundle)

All changes are to `benchmark/soar/demo_sala/prepare_env.sh` (the `SGLANG_SERVER_ARGS` export inside the GPTQ branch). No source-code changes.

### Change A-1: Drop `--enable-torch-compile --torch-compile-max-bs 8`

**Evidence**: Test 33 (commit `e625363a8`) ran the identical config minus torch.compile and measured:
- Server boot: 36s vs 219s with compile — **saves 183s**
- Total eval duration: 3016s vs 3005s with compile — **within 0.4% noise**
- Per-tier speed: torch.compile delivered ≈0% speedup on this workload

Torch.compile's graph breaks on dynamic-shape prefill; sglang's built-in CUDA graph capture already covers decode. The compile pass is pure overhead on SM120 for MiniCPM-SALA.

**Risk**: none observed; confirmed in Test 33.

### Change A-2: Raise `--cuda-graph-max-bs` to 24

**Evidence / rationale**: current implicit default may not cover all decode batch sizes up to `--max-running-requests 24`. If any decode batch misses the graph, it falls to eager path (≈15–20% slower per step). Explicitly aligning `cuda-graph-max-bs = max-running-requests` guarantees full decode coverage.

**Risk**: larger graph capture adds a few hundred MB of GPU memory per captured batch size. `--mem-fraction-static 0.84` already leaves headroom; we will watch for OOM during warmup and reduce to 16 if needed.

### Change A-3: Add `--stream-interval 2`

**Evidence / rationale**: default stream-interval=1 emits a detokenization/send event every decoded token. Setting to 2 halves tokenization-path overhead at the cost of slightly laggier streaming. For this benchmark (completion-latency metric, not token-latency), there is no user-visible downside.

**Risk**: negligible; purely a server-side batching knob.

### Change A-4 (optional, only if A-1..3 pass): reintroduce `prefill-max-req=4, sched-cons=0.8`

**Evidence**: Test 25A-spd showed this combo improved S1 from 120.48s → 110.58s (**−8.2%**) with no accuracy regression on that old-fcloud run. It was reverted in Test 29+ during the accuracy bisect but the bisect proved accuracy noise is independent of scheduling.

**Sequencing**: apply only after A-1..3 land cleanly; keep as a separate test to isolate its impact on this fcloud instance.

**Risk**: minor; reversible.

## 4. Implementation plan (to execute only after approval)

1. Edit `benchmark/soar/demo_sala/prepare_env.sh` GPTQ branch:
   - Remove `--enable-torch-compile --torch-compile-max-bs 8`
   - Add `--cuda-graph-max-bs 24 --stream-interval 2`
2. Commit to `mixed_minicpm_cudagraph`, push to `minicpm-src`.
3. Request user to start fcloud → run full pipeline:
   - `sync` → `restart-server --quant-mode gptq` → `wait-server`
   - `accuracy` (confirms C ≥ 0.92)
   - `speed --variant all`
   - `shutdown`
4. Record as **Test 35** in [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md).
5. Decision gate:
   - Speed ≥ 3% faster on S1 **and** C ≥ 0.96: promote to v20 submission candidate, continue to Change A-4.
   - Speed neutral: keep A-1 only (boot-time win), revert A-2/A-3.
   - Accuracy drops to C=0: revert whole bundle (pure config revert, low-cost).
6. If A-4 is applied later as Test 35b: compare delta vs Test 35; if ≥5% S1 improvement and C preserved, lock in.

## 5. Validation commands

```bash
# Locally
git diff benchmark/soar/demo_sala/prepare_env.sh

# On fcloud (after sync)
grep 'SGLANG_SERVER_ARGS=' /root/submission_sim/prepare_env.sh | head -2
```

Speed measurement reference (current-fcloud Test 34a numbers):
| Tier | Test 34a | Expected after bundle | Target |
|---|---|---|---|
| S1 | 108.91s | ≤ 106s | ≤ 105s |
| S8 | 39.99s | ≤ 39s | ≤ 38.5s |
| Smax | 33.44s | ≤ 33.5s | ≤ 33s |

Accuracy gate: C ≥ 0.92 (norm ≥ 97.01%). Higher is nice-to-have, not required for this iteration.

## 6. Rollback

```bash
git checkout benchmark/soar/demo_sala/prepare_env.sh
git push minicpm-src mixed_minicpm_cudagraph --force-with-lease
```
(pure config revert; no wheel rebuild needed)

## 7. What is NOT in this iteration

- **No kernel changes**. CUTLASS SM120 / SM120 native MMA / QMMA-fp8 GEMM are deferred to a dedicated proposal after speculative decoding (Iteration C) is scoped.
- **No new quantization path**. FP8 blockwise (`OPTION_B-final`) already failed at accuracy; W4A8 Marlin variant also requires kernel work.
- **No draft-model changes**. Speculative decoding (Iteration C1 n-gram, then C2 EAGLE training) remains the next major iteration.

## 8. Why this iteration before Iteration C (speculative decoding)

- A-1..3 are almost free (one prepare_env edit, one commit, one test round).
- They establish a clean runtime baseline by removing the torch.compile confound, which will make speculative-decoding benchmarking more interpretable.
- The boot savings (3 min) matter under the official 5h ceiling.
- They are fully reversible if Iteration C later needs a different runtime shape.

---

## Open questions for user

1. Approve the A-1..3 bundle as described?
2. Prefer to run A-4 together with A-1..3 (one test) or keep them separate (two tests)?
3. Any objection to removing torch.compile? (Test 33 evidence is strong but user may have context about official-eval environment we should respect.)
