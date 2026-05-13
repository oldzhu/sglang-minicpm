# CHAT — Medusa Stage 3b: trained head loads but draft never accepted

Date: 2026-05-13 03:37 UTC
Topic: CHANGE_0164 Medusa Stage 3b — root-cause of `accept_len=1.00` after retraining

## User asks
- "go" / "pls go" — execute the full Medusa Stage 3b pipeline on fcloud (collect GPTQ hidden states, retrain head, restart server, measure speedup). Expect `accept_len > 1.0`.

## Agent actions / outcomes

### 1. Three full pipeline iterations executed
| Iter | Head version             | Train acc | S1 (s) | S8 (s) | Smax (s) | Avg accept_len |
|------|--------------------------|-----------|--------|--------|----------|----------------|
| v0   | Stage 3a (zero-init ref) | n/a       | 202.70 | 61.60  | 43.29    | 1.00           |
| v1   | Trained, WRONG labels (current-token, no shift) | 99.76% | 215.13 | 64.43 | 45.21 | 1.00 |
| v2   | Trained, CORRECT labels (next-token, shift by 1) | 99.89% | 216.82 | 64.59 | 45.26 | 1.00 |

### 2. Verified offline that v2 head IS correct
Ran `verify_head.py` on the dumped hidden states with the saved checkpoint:
- `W1 norm = 24.125` (non-zero, real weights)
- `match_shifted (y_pred[i] == y_next[i+1]) = 99.91%` — head correctly predicts the NEXT token
- `match_same (y_pred[i] == y_next[i]) = 2.27%` — confirms labels are next-token, not current-token

Also reproduced via the `MedusaHeads` module API itself (not just F.linear):
- `MedusaHeads.load_trained_weights()` succeeds in the same fcloud env → `W1 norm` jumps from 0 → 24.125.
- `heads(h)` forward returns 99.91% next-token match.

### 3. Verified server-side load
Server log on the retrained head shows:
```
MedusaHeads: loaded 1 head(s) from /root/medusa_head_k1_gptq.pt
MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1_gptq.pt
MedusaWorker ready: K=1, num_heads=1, hidden=4096, dtype=torch.bfloat16, device=cuda:0, trained=True, approx_weight_MiB=32.0
```
So at runtime `_use_trained_heads=True` and the head weights are loaded.

### 4. So why does inference give 0% accept rate?
Looking at `NgramVerifyInput.prepare_for_verify` + `_greedy_verify` + `verify_tree_greedy` in `python/sglang/srt/speculative/ngram_info.py`:

With our K=1 worker setting `draft_token_num=1`:
- The verify forward processes exactly **1 position per request**, with `input_ids = self.draft_token` (just the draft).
- `target_predict = argmax(next_token_logits)` at that single position — this is the model's prediction of what comes **after** the draft (i.e., position p+1 token, not p).
- `candidates[0] = draft_token` is checked against `target_predict[0]` for acceptance.

This means we're asking: **"Does the model think the draft is what comes AFTER the draft?"** — which is structurally unable to verify the draft. To check whether the draft is correct, the bonus position (state before consuming the draft) needs its `argmax(lm_head)` to match the draft, not the post-draft state.

This is consistent with the observation that **Stage 3a also gave accept_len=1.00 on fcloud** (the medusa K=1 path has never produced any acceptance, even with the n-gram draft = `output_ids[-1]`).

### 5. Hypothesis (to verify next session)
- `draft_token_num` in our NgramVerifyInput should likely be **2** (1 bonus position + 1 draft), with `batch.input_ids = [last_accepted_token, draft]` per request. The bonus position is where the model predicts the next token from the previous accepted state — and that prediction is what we compare the draft to.
- Or: the K=1 medusa path needs a custom verify that, instead of `argmax(post_draft_logits) == draft`, uses `argmax(pre_draft_logits) == draft`. In sglang's spec design, "pre-draft logits" only exist as the bonus position output.

### 6. Cost saving
fcloud instance paused at 03:37 UTC (`pause-instance` returned `{"status":"success"}`).

## Cross-references
- Source: `python/sglang/srt/speculative/medusa_worker.py` (`_forward_verify_k1` lines ~283-450)
- Source: `python/sglang/srt/speculative/ngram_info.py` (`prepare_for_verify`, `_greedy_verify`, `verify`)
- Trainer: `benchmark/soar/demo_sala/train_medusa_head.py` (`_train_from_gptq_dump`, label-shift fix applied)
- Trained head: `/root/medusa_head_k1_gptq.pt` on fcloud (v2, 32 MiB, next-token labels, 99.89% train top-1)
- Verification script: `/tmp/verify_head.py` (local), `/tmp/test_load.py` (local) — reproduced on fcloud, both confirm head is correct
- Commit: `c7da86d42` on `minicpm-src/mixed_minicpm_cudagraph` (CHANGE_0164 dump mode + trainer)
- TEST_RESULTS_TRACKING.md row to add: Test #N, 2026-05-13, c7da86d42, GPTQ+FP8 KV dense Tier1 + MEDUSA K=1 + trained head, S1=216.82s S8=64.59s Smax=45.26s accept=1.00 → no speedup

## Conclusion / Next steps
The head training pipeline (dump → train → load) is end-to-end correct. The remaining bug is in the **verify acceptance setup** in `MedusaWorker._forward_verify_k1`: `draft_token_num=1` with a single draft position does not give the verify logic a position to compare against. The fix requires reshaping the verify input so that there are 2 positions per request (bonus + draft) and the verify compares draft against `argmax(logits[bonus_position])`. This is a NgramVerifyInput-layout change, not a head problem.

**Recommendation for next iteration**:
1. Study how sglang's stock EAGLE/Medusa verify lays out its bonus + draft positions (it works there, so the pattern exists).
2. Refactor `_forward_verify_k1` to: `draft_token = [last_accepted, head_predicted_draft]` per request, `draft_token_num=2`, with retrive structures that mark position 0 as bonus and position 1 as draft-of-0.
3. Capture hidden states only at the bonus position (where the next step's draft will be predicted from).
4. Re-run dump → re-train head with hidden states from the bonus position → re-test.

This session did NOT produce a working accept_len > 1.0. The next iteration must fix the verify layout before any speedup can be measured.
