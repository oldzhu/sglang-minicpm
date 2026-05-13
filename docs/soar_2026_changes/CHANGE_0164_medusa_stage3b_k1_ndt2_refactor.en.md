# CHANGE_0164 — Medusa Stage 3b K=1 refactor to canonical `draft_token_num=2` layout

Date: 2026-05-13
Branch: `mixed_minicpm_cudagraph`
Related: `PROPOSAL_medusa_stage3b_k1_draft_token_num_2.{en,zh}.md`,
         `CHANGE_0163_medusa_stage3b_trained_heads.{en,zh}.md`,
         `CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.{en,zh}.md`

## Background & motivation

CHANGE_0163 introduced the trained-head Medusa Stage 3b worker using a
`draft_token_num=1` verify layout. Three full fcloud iterations
(202–217 s on S1, accept_len = 1.00 always) showed that the draft can
**never** be accepted regardless of head quality. Reading the
`sgl_kernel::VerifyTreeGreedy` CUDA kernel (`sgl-kernel/csrc/speculative/eagle_utils.cu`)
revealed two coupled bugs:

1. **Speed bug** — the kernel treats `retrive_index[0]` as the "root,
   always accepted" and walks children at indices 1..ndt-1 against
   `target_predict[root]`. With `ndt=1` there are *zero* children, so
   no draft can ever be validated.
2. **Correctness bug** — the kernel's final line
   `predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx]`
   commits the model's prediction conditioned on `(prefix + draft)`.
   With our ndt=1 Stage 3a fallback (`draft = output_ids[-1]`), the
   committed token is sampled from `prefix + T_N + T_N` (duplicated),
   not `prefix + T_N` — a small but structural distribution bias on
   every Medusa decode step. This is consistent with the
   v23-medusa-passthrough submission showing 78.71 % vs the v22
   baseline 79.29 %.

Canonical EAGLE / NGRAM layout uses `ndt = num_drafts + 1`:
* position 0 = **bonus** = last committed token (`output_ids[-1]`)
* positions 1..ndt-1 = speculative drafts

`prepare_env.sh` already exports `NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))`
(= 2 for K=1), so the server arg side was already correct; only the
worker code disagreed.

## Rule-compliance statement

* No change to the official eval harness or scoring path.
* No change to the model weights, quantization, or KV-cache dtype.
* No change to baseline `--force-dense-minicpm` + FP8 KV cache + Tier1
  long-context server args.
* Server-arg surface unchanged — both before and after, sglang sees
  `--speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2`.
* Only the internal MedusaWorker verify construction changes.

## Implementation plan (before change)

1. `__init__`: bump `self.draft_token_num` from `num_heads` (=1) to
   `num_heads + 1` (=2).
2. `_forward_verify_k1`:
   * Build flat `draft_token` tensor of shape `(bs*2,)` with
     `[output_ids[-1], head_pred_or_fallback]` interleaved per request.
   * `retrive_index = arange(bs*2).view(bs, 2)`.
   * `retrive_next_token = [[1, -1], ...]` (bonus → draft linear chain).
   * `retrive_next_sibling = [[-1, -1], ...]`.
   * `positions = seq_lens[i] + arange(2)` flattened.
   * `tree_mask` per request follows NGRAM `USE_FULL_MASK` convention:
     `(ndt, seq_len_i - 1 + ndt)` shape, prefix all ones, trailing
     `(ndt, ndt)` lower triangular. Flatten + concat.
   * `CaptureHiddenMode.FULL` (need bonus hidden, not last hidden).
   * After forward, reshape `hidden_states` to `(bs, ndt, h)` and take
     position 0 — the bonus hidden, which represents the model's state
     after consuming `prefix + T_N`, byte-equivalent to the
     `CaptureHiddenMode.LAST` view our v1 trainer was fit on.
   * Use this bonus hidden for both the head forward (next draft) and
     the offline dump.
3. Remove the dead `CHANGE_0160` bonus-zeroing loop (it only fired
   when `accept_length >= 1`, which never happened with ndt=1; with
   ndt=2 the standard `_free_cache` path in `NgramVerifyInput.verify`
   handles `req_to_token` correctly).

## Actual code changes (after change)

* `python/sglang/srt/speculative/medusa_worker.py`
* `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py`
  (verbatim copy — kept in sync via `cp` to ensure the demo_sala
  submission tarball ships the same worker code as the upstream tree)

Both files now:
* `self.draft_token_num = self.num_heads + 1` (= 2 for K=1).
* `_forward_verify_k1` builds the 2-node verify input as described
  above.
* `CaptureHiddenMode.FULL` is used when either trained heads are
  active or hidden-state dump mode is collecting samples.
* Position 0 of the reshaped hidden states feeds both the head and
  the dump.

## Expected behaviour

* **Correctness restored on every Medusa decode step**: the bonus
  position always commits `target_predict[0]` = argmax of logits
  computed conditioned on `prefix + T_N`. This is byte-equivalent to
  what non-spec decode would produce. Worst case (draft rejected) we
  get exactly 1 correct token, matching standard decode.
* **Speed**: if the trained head's draft equals the model's true
  next-token (offline match: 99.91 % on shifted labels), the kernel
  walks the child, accepts, and commits 2 tokens per step
  (`accept_len ≈ 2.0`). If the head misses, falls back to 1 token
  (= standard decode speed, no regression).
* **Stage 3a fallback** (no trained head, `SOAR_MEDUSA_HEAD_PATH=""`):
  draft = duplicate of last token → always rejected → exactly 1
  correct token committed per step. No more silent corruption, but no
  speedup either (acc baseline restored).

## Validation commands

```bash
# fcloud one-shot cycle (gated on user approval):
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# (a) Correctness: Stage 3a fallback should now match v22 baseline acc.
SOAR_SPEC_MEDUSA=1 SOAR_MEDUSA_HEAD_PATH="" \
  python3 scripts/fcloud/fcloud_workflow.py accuracy
# Expected: acc ≈ 79.3 % (vs current v23 77.87 %)

# (b) Speed + accept_len with trained head v2:
SOAR_SPEC_MEDUSA=1 \
SOAR_MEDUSA_HEAD_PATH=/root/models/medusa_head_v2.pt \
  python3 scripts/fcloud/fcloud_workflow.py speed --variant all
# Expected: S1 drops from 216 s → 120–140 s if head transfers,
#           accept_len ≈ 1.8–2.0; else no regression vs S1 = 202 s.

python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

## Result summary table (to be filled after fcloud cycle)

| Config | S1 (s) | S8 (s) | Smax (s) | accept_len | acc (%) | Notes |
|---|---|---|---|---|---|---|
| v22 baseline (no Medusa)        | 121.71 | 44.09 | 35.86 | n/a  | 79.29 | reference |
| CHANGE_0163 + ndt=1 (broken)    | 202.70 | 61.60 | 43.29 | 1.00 | 77.87 | corruption bug |
| CHANGE_0164 + ndt=2 Stage 3a    |  TBD   |  TBD  |  TBD  | 1.00 |  TBD  | correctness fix only |
| CHANGE_0164 + ndt=2 Stage 3b v2 |  TBD   |  TBD  |  TBD  |  TBD |  TBD  | with trained head |

## Rollback instructions

To revert to CHANGE_0163 (ndt=1) behavior:

```bash
git revert <this-commit>
# or, manually:
#   self.draft_token_num = self.num_heads          # was: num_heads + 1
#   _forward_verify_k1 reverts to single-position layout
```

Submission rollback: set `SOAR_SPEC_MEDUSA=0` in `prepare_env.sh` (or
override at launch) — disables the worker entirely, pure v22
behavior.

## Next-step suggestions

1. If accept_len ≥ 1.8 and S1 ≤ 140 s: package as v24 submission.
2. If accept_len ≈ 1.0–1.3: investigate train-vs-serve distribution
   mismatch (sample dump position 0 hidden during run, compare to
   offline trainer's hidden distribution).
3. If accuracy regresses below 79 %: audit `_free_cache` interaction
   with `req_to_token` pool when ndt=2 (the `CHANGE_0160` workaround
   was for ndt=1; the ndt=2 path uses the standard NGRAM eviction
   walker and should be correct, but worth a smoke test on a 10-req
   subset first).

---

## Result (2026-05-13) — CATASTROPHIC, REVERT REQUIRED

**Test ID**: `Stage3b-ndt2-CATASTROPHIC` (commit `4b442f421`, fcloud `ai-e7e98a7c52`).

| Metric | Value | vs Baseline (Test 12 = 79.29%) |
|--------|-------|-------------------------------|
| ori_accuracy | **15.13 %** | −64.16 pt |
| normalized   | **18.92 %** | well below 97 % → **C = 0 (eliminated)** |
| mcq          | **0.00 %**  | runaway: avg_out = 64460 (full max_out_len) |
| niah         | 3.33 %      | −96.67 pt |
| cwe          | 15.67 %     | −56.33 pt |
| fwe          | 20.00 %     | −78.89 pt |
| qa           | 36.67 %     | −26.66 pt |
| eval duration| 7911 s (2 h 12 m) | ~2.6× baseline |

Kernel reported `accept_len = 1.46, accept_rate = 0.73` throughout the run —
i.e. the speculative pipeline was *structurally* alive, but the committed
tokens are wrong. Every MCQ sample exhausts the 65536 max_out_len budget,
strongly suggesting the bonus-position output token is corrupted (no stop
token ever emitted).

### Hypotheses (not yet bisected)

1. **Position-of-bonus mismatch.** We compute
   `positions = seq_lens + arange(ndt)`, expecting the bonus to occupy
   slot `seq_lens` and the draft to occupy `seq_lens+1`. NGRAM's own
   `_prepare_for_speculative_decoding` uses
   `reconstruct_indices_from_tree_mask(..., batch.seq_lens, positions, ...)`
   which derives positions from the tree mask. If the NGRAM convention
   expects the *root* of the verify tree to be at position `seq_lens-1`
   (overwriting the last-committed-token slot) rather than `seq_lens`,
   our bonus KV is written at the wrong row and every subsequent step
   reads stale state.
2. **Hidden distribution mismatch for the trained head.** The head was
   retrained on hidden states captured from the ndt=1 path
   (`hidden[:,0,:]` of a 1-token TARGET_VERIFY). In ndt=2 the position-0
   hidden has different attention context (it now sees one extra "future"
   token via tree-mask routing if the mask is wrong, or is identical if
   correct). If the captured-vs-served distributions diverge, the head
   produces nonsense drafts; but that alone would just hurt accept rate,
   not destroy the bonus output. So this can't be the *sole* cause.
3. **`prepare_for_verify` side-effects on `seq_lens`.** The shared
   `NgramVerifyInput.prepare_for_verify` may bump `seq_lens` by `ndt`
   internally; if our explicit `positions` was already computed before
   that call, the actual forward writes KV at offset `+ndt` past where
   we intended.

### Decision: REVERT

- Revert `medusa_worker.py` in both `python/sglang/srt/speculative/` and
  `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/` to
  the Stage 3a `ndt=1` path (commit `3a15a6de3` /
  `Stage3a-force-dense` baseline: 78.40 % acc, S1=204.86 s).
- Keep CHANGE_0164 documents in place as a record of the failure.
- File follow-up investigation: instrument `_forward_verify_k1` to dump
  one micro-batch's `(positions, seq_lens_in, seq_lens_out, draft_tokens,
  input_ids_used_by_attention, committed_token)` quadruple and compare
  byte-for-byte against a single-token NgramWorker decode for the same
  prompt, to localize the off-by-one before re-attempting ndt=2.

### Rollback commands

```
git revert 4b442f421       # or
git checkout 3a15a6de3 -- python/sglang/srt/speculative/medusa_worker.py \
                          benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py
```
