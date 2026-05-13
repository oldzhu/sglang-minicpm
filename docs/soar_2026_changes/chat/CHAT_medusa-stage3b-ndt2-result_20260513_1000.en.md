# CHAT — Medusa Stage 3b ndt=2 result (post-validation)

Date: 2026-05-13 ~10:00 UTC
Topic: validation of CHANGE_0164 (ndt=2 refactor) on fcloud
Branch: `mixed_minicpm_cudagraph`
Related docs:
- [CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md](../CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md)
- [PROPOSAL_medusa_stage3b_k1_draft_token_num_2.en.md](../PROPOSAL_medusa_stage3b_k1_draft_token_num_2.en.md)
- [CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.en.md](CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.en.md)
- [TEST_RESULTS_TRACKING row `Stage3b-ndt2-CATASTROPHIC`](../TEST_RESULTS_TRACKING.md)

## User asks (verbatim / condensed)
1. "pls go" — execute the ndt=2 fcloud validation cycle.
2. (Implicit) report results; decide next step.

## Agent actions
1. Pushed commit `4b442f421` (CHANGE_0164 ndt=2 refactor + docs) to `minicpm-src`.
2. `start-instance` (resumed paused fcloud).
3. `sync` — fast-forwarded `c7da86d4..4b442f4`.
4. `restart-server`; `wait-server` ready after 217 s (CUDA graph capture 826 s).
5. Server log confirmed `MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1_gptq.pt`, `trained=True, K=1, num_heads=1, hidden=4096`.
6. Started accuracy eval at 07:43 UTC; finished at 09:55 UTC (7911 s).
7. `pause-instance` (after first 504 retry succeeded).

## Outcome
**CATASTROPHIC accuracy regression** — see CHANGE_0164 Result section.
- ori_accuracy = **15.13 %**, normalized = **18.92 %**, **C = 0 (eliminated)**.
- mcq = **0.00 %**, avg_out = 64460 (full max_out_len) — every MCQ runs away.
- niah=3.33 %, cwe=15.67 %, fwe=20.0 %, qa=36.67 %.
- Eval duration 7911 s (~2.6× baseline).
- Kernel reported `accept_len=1.46, accept_rate=0.73` throughout — speculative pipeline structurally alive but committing wrong tokens.

This is **worse than the v23 ndt=1 silent corruption** (78.71 %) and worse than any prior medusa attempt. The bonus-position output token itself appears to be corrupted (no stop token ever emitted on MCQ).

## Decision
**Revert CHANGE_0164.** Restore Stage 3a (`ndt=1`) Medusa worker from commit `3a15a6de3` (Stage3a-force-dense baseline: 78.40 % acc, S1=204.86 s) as the best-known-good Medusa state. Open a follow-up investigation to bisect the ndt=2 verify-layout bug before re-attempting Stage 3b.

## Open questions / follow-ups
- Does `NgramVerifyInput.prepare_for_verify` mutate `batch.seq_lens` before the forward pass uses our externally-computed `positions`? Need to instrument and compare with a working NGRAM single-step.
- Is the bonus token's KV expected at `seq_lens` or `seq_lens-1` in the NGRAM convention? The kernel structurally accepts but commits garbage, suggesting an off-by-one in either positions or `req_to_token`.
- Should we abandon the ndt=2 refactor entirely and instead investigate why v23 ndt=1 silent-corrupts (–0.58 pt)? That regression is dramatically smaller and may be cheaper to fix.

## Cross-references
- Commit `4b442f421` (ndt=2 refactor) — to be reverted.
- Commit `3a15a6de3` (Stage3a-force-dense) — rollback target.
- TEST_RESULTS_TRACKING row `Stage3b-ndt2-CATASTROPHIC`.
