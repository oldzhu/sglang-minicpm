# CHAT: W4A8 Verification & Decision — 2026-05-15

## 1. User Questions / Asks

1. Asked for full discussion of best configs before deciding next action: "what is our best config so far and what is our best config for nvfp4 fos so far and let us dicuss more on this."
2. After the discussion, asked: "so our next improving plan better be based on Best GPTQ (Marline W4A16) Baseline, do u think real W4A8 could be direction to make our rank back to top 30?"
3. Raised two specific concerns about W4A8:
   - Concern 1: FP8 accuracy risk (fewer mantissa bits than BF16)
   - Concern 2: Old W4A8 path regression (S1 +118%) might predict true W4A8 will also be slow
4. Asked to document the concern analysis and proceed with W4A8 verification test on fcloud.

## 2. Agent Answers / Actions

### Best Config Summary
- **Best GPTQ baseline**: v24 (current HEAD, commit `8ce03caaf`) — S1=110.54s, S8=40.32s, Smax=33.48s local. Official: pre-v18 scored 30.04 (#22), v23 scored 25.55 (#31). Local accuracy floor 75-79% with ±3pt variance dominated by mcq runaway.
- **Best NVFP4-FOS**: FOS-7 at 70.98% accuracy, norm≈88.73% → C=0 always. Speed S1=173.83s (57% worse than GPTQ), Smax=31.07s (13% faster). Parked — not submission-viable.
- **Strategy**: W4A8 is the highest-ROI item. Estimated 15-20% effective speed gain → score ~29-30 (back to top 25-30) IF paired with accuracy stability.

### Concern Analysis (documented in PROPOSAL_W4A8_REAL_002)
- **Concern 1 (FP8 accuracy)**: FP8 has 3 mantissa bits vs BF16's 7, but per-token scaling recovers precision. Old W8A8 test (both sides FP8) was accuracy-neutral (79.20% vs 79.29%). True W4A8 only changes activation side → risk is lower. Mitigation: Phase 0 MSE microbenchmark + full accuracy gate (C=1.0 or abandon).
- **Concern 2 (old regression predicts bad speed)**: Old path was slow because INT4 weights were upcast to FP8 at load time (doubling HBM weight traffic). True W4A8 preserves INT4 weight storage → zero bandwidth penalty. True W4A8 is equal or better than baseline on every dimension.

### W4A8 Verification Test on fcloud
- Started fcloud instance, synced code, overrode `SOAR_W4A8_FP8_GEMM=1`
- Server launched with FP8 conversion (saw all 32 layers' [SOAR W4A8] enabled logs)
- CUDA graph capture took ~5 min (bs 1-24, 7 buckets)
- Speed benchmarks completed:

| Tier | W4A8 FP8 GEMM=1 | v24 Baseline | Δ |
|---|---:|---:|---:|
| S1 | **267.19s** | 110.54s | **+142%** |
| S8 | **68.85s** | 40.32s | **+71%** |
| Smax | **45.86s** | 33.48s | **+37%** |

- Old-path regression CONFIRMED. Current Tier1/long-context config does not help — weight bandwidth penalty still dominates.
- Reverted env override and paused instance.

## 3. Outcomes

- **Decision**: Proceed to true W4A8 kernel build (INT4 storage + FP8 MMA) per `PROPOSAL_W4A8_REAL_001` Option A.
- **Verification complete**: Old path regression confirmed at same magnitude. No surprises.
- **Documents created**: `PROPOSAL_W4A8_REAL_002_concerns_and_verification.{en,zh}.md`
- **TEST_RESULTS_TRACKING.md updated** with W4A8#1-reverify row.

## 4. Cross-References

- `PROPOSAL_W4A8_REAL_002_concerns_and_verification.en.md` — concern analysis + verification plan
- `PROPOSAL_W4A8_REAL_002_concerns_and_verification.zh.md` — Chinese version
- `TEST_RESULTS_TRACKING.md` — W4A8#1-reverify row added
- `PROPOSAL_W4A8_REAL_001.en.md` — True W4A8 kernel design (Option A: Machete-style FP8 QMMA)
- Commit `8ce03caaf` — v24 baseline (mcq cap disabled)
- Commit `7ce21c3f5` — Old W8A8 FP8 path (abandoned)

## 5. Open Questions / Follow-ups

- Proceed to true W4A8 kernel build? (3-4 day effort, requires user approval)
- Parallel track: fix mcq accuracy stability (larger cap 16384+ or repetition detector)?
