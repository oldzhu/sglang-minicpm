# CHAT — Medusa R1b Stage 2 implementation — 2026-05-11 08:04

## User asks (this round)

> 1. k=1 only predict 1 token? k>=2 will predict more than 2? if yes, prefer >=2
> 2. what difference between ngramverifyinput and medusainput.verify?
> 3. 288M snapshot for what? each step 288M, how many steps? totally we will
>    have 288M*N steps additional memory cost?
> 4. what are difference between eager-only and insist on CUDA graphs.
> 5. how about Land Stage 2 as an intermediate smoke-test commit and test to
>    evaluate before go stage3, what do u prefer?
>
> go

## Agent responses

### Q&A summary

1. **K=1 vs K≥2**: K=1 gives up to 2 accepted tokens/step; K=N gives up to
   N+1. R1 heads are zero-init → expected accept = 0 regardless of K.
   Recommendation: keep K=1 in R1b; raise after R3 trained heads.
2. **NgramVerifyInput vs MedusaInput.verify**: Both produce verify trees with
   the same structure. K=1 Medusa is a 2-token chain, structurally identical
   to ngram_len=2. Reuse saves ~200 LOC. Recommendation: reuse.
3. **288 MiB snapshot**: One snapshot lives between snapshot+restore per
   step. Total constant ~288 MiB (not multiplied by N steps).
4. **Eager vs CUDA graphs**: Eager has higher launch overhead but no shape
   constraint. R1b ships eager; R1c will add graph capture for verify shape.
5. **Stage 2 intermediate commit**: Strongly preferred. Isolates wiring bugs
   from algorithm bugs. ~80 LOC vs ~400 LOC for Stage 3.

### Decisions taken (with user's "go")

- K=1, reuse `NgramVerifyInput` for Stage 3, eager-only in R1b, ship Stage 2
  as the next commit.

### Implementation completed

**Stage 2 turned out to be even simpler than CHANGE_0154 §4 suggested.**
Code survey showed the six `is_ngram()` call sites only matter when MEDUSA
actively triggers TARGET_VERIFY capture. Stage 2 stays in normal DECODE
mode, so 5 of the 6 sites need no change. Final footprint:

| File | Change |
|---|---|
| `python/sglang/srt/speculative/medusa_worker.py` | Full rewrite of the R1a stub → ~150 LOC Stage 2 worker (delegation + heads instantiation) |
| `python/sglang/srt/managers/scheduler.py` L866 | `+ or self.spec_algorithm.is_medusa()` guard for `init_disaggregation` (defensive; not on SOAR path) |
| `docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.{en,zh}.md` | New |
| `docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.{en,zh}.md` | This file |

### Stage 2 worker behavior

- Instantiates `MedusaHeads(K, hidden, lm_head, dtype)` on GPU
  (~32 MiB BF16 at K=1, hidden=4096). W1 zero-init from R1a is preserved.
- Logs `MedusaWorker Stage 2 ready: K=1, hidden=4096, dtype=torch.bfloat16,
  device=cuda:0, approx_weight_MiB=32.0`.
- `forward_batch_generation(batch)` is **pure delegation**: pulls
  `batch.get_model_worker_batch()`, calls
  `target_worker.forward_batch_generation(mwb)`, wraps the result.
- No `capture_hidden_mode` mutation, no `forward_mode` change, no draft
  tokens, no spec_info. Output should be byte-identical to baseline.

### Validation criteria (to run on fcloud)

- `SOAR_SPEC_MEDUSA=1` server boots cleanly.
- Server log contains `MedusaWorker Stage 2 ready`.
- Accuracy = baseline (byte-identical predictions, not just close).
- S1 regression ≤ 3% (heads allocated but unused).

## Outcome

- Stage 2 implementation lands ready for fcloud validation.
- Stage 3 hand-off documented in CHANGE_0155 §8.
- Awaiting user approval to run fcloud workflow (`start-instance` → `sync`
  → `restart-server` → `accuracy` → `speed` → `pause-instance`).

## Cross-references

- [CHANGE_0155_medusa_phase_r1b_stage2.en.md](../CHANGE_0155_medusa_phase_r1b_stage2.en.md)
- [CHANGE_0154_medusa_phase_r1b_design.en.md](../CHANGE_0154_medusa_phase_r1b_design.en.md)
- [CHANGE_0153_medusa_phase_r1_design.en.md](../CHANGE_0153_medusa_phase_r1_design.en.md)
- Commit: TBD (to be filled in after push)
