# PROPOSAL — Medusa Stage 3b refactor: switch to draft_token_num=2 (bonus + draft layout)

Date: 2026-05-13
Author: Copilot agent (SOAR 2026 / MiniCPM-SALA)
Status: PROPOSED — needs user approval before implementation

## 1. Background

CHANGE_0164 added GPTQ-aligned head training and Stage 3b inference for Medusa K=1. Three iterations on fcloud produced:

| Iter | Head           | Train top-1 | S1 (s) | S8 (s) | Smax (s) | Accept |
|------|----------------|-------------|--------|--------|----------|--------|
| 3a   | zero-init      | —           | 202.70 | 61.60  | 43.29    | 1.00   |
| 3b-v1| wrong labels   | 99.76%      | 215.13 | 64.43  | 45.21    | 1.00   |
| 3b-v2| correct labels | 99.89%      | 216.82 | 64.59  | 45.26    | 1.00   |

Offline verification proved the trained head IS correct: on the dumped hidden states, the inference-side forward (`SiLU(W1·h) + h` → `F.linear(., lm_head.weight)`) reproduces 99.91% next-token accuracy. The bug is **not** the head.

## 2. Root cause (verified from sgl-kernel/csrc/speculative/eagle_utils.cu::VerifyTreeGreedy)

The K=1 verify path in `MedusaWorker._forward_verify_k1` uses `NgramVerifyInput(draft_token_num=1)`. In the kernel:

```cpp
last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];     // root
accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx; // always accept root
for (j = 1; j < num_speculative_tokens; ++j) { ... compare children ... }
// num_speculative_tokens == draft_token_num here ⇒ loop runs 0 times when ndt=1
predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
```

The kernel treats `retrive_index[0]` as the **root** (last accepted from previous step, never verified). Children (positions 1..ndt-1) are compared against `target_predict[root_position]`. With `draft_token_num=1`, there are no children → no draft can ever be accepted.

Even worse, our setup passes `input_ids = [head_predicted_draft]` (a single token at position seq_len). The model produces `logits[draft_pos]` = "next-token-after-the-draft", which gets written as `predicts[root_pos]` and appended to `req.output_ids`. So:
- **Correctness bug**: We append `next-token-after-an-unverified-draft` instead of `next-token-after-last-accepted`. Whenever the draft is wrong, we corrupt the sequence.
- **Speed bug**: Zero acceptance regardless of head quality.

This affects **both Stage 3a and Stage 3b** — they share the same `_forward_verify_k1` layout.

## 3. Proposed fix

### 3a. Verify layout — draft_token_num = 2

Each request occupies 2 verify positions per step:
- Position 0 (root): `last_output` = `req.output_ids[-1]` (token we know is correct)
- Position 1 (child / draft): trained-head prediction `req._medusa_draft_token` (or `last_output` for the very first step before any head prediction has been cached)

```python
# Each request contributes 2 tokens to verify input
for req in batch.reqs:
    last_out = req.output_ids[-1]
    cached = getattr(req, "_medusa_draft_token", None)
    draft = cached if (cached is not None and self._use_trained_heads) else last_out
    draft_token_list.extend([last_out, draft])
draft_tokens = torch.tensor(draft_token_list, dtype=torch.int64, device=self.device)  # (bs*2,)

# Tree: 2-node linear chain.
#   retrive_index[bx]   = [0, 1]   (positions in flattened batch row)
#   retrive_next_token[bx] = [1, -1]  (root's first child = idx 1; child has no child)
#   retrive_next_sibling[bx] = [-1, -1]  (no siblings)
retrive_index = (
    torch.arange(bs * 2, device=self.device, dtype=torch.int64)
    .reshape(bs, 2)
)
retrive_next_token = torch.tensor([[1, -1]] * bs, ...)
retrive_next_sibling = torch.tensor([[-1, -1]] * bs, ...)

# Positions: per-req absolute positions [seq_len, seq_len+1]
positions = torch.repeat_interleave(batch.seq_lens, 2) + torch.tile(torch.arange(2), (bs,))

# Tree mask: each verify position attends to:
#   - all KV-cached prefix tokens of that req
#   - position 0 attends to itself
#   - position 1 attends to position 0 + itself
# In the flat layout used by NgramVerifyInput, tree_mask is per-token over (seq_len + ndt) tokens.
# For a K=1 chain both new tokens see the full prefix; position 1 additionally sees position 0.

spec_info = NgramVerifyInput(
    draft_token=draft_tokens,           # (bs*2,)
    tree_mask=...,                       # see below
    positions=positions,                 # (bs*2,)
    retrive_index=retrive_index,         # (bs, 2)
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    draft_token_num=2,
)
spec_info.capture_hidden_mode = CaptureHiddenMode.FULL  # capture all 2 positions per req
```

### 3b. Hidden state capture — position 0 (root), not last

We need the hidden state where the **NEXT step's** draft will be predicted. That state is `h(token_added_this_step)`. Two scenarios:
- Draft accepted: appended tokens are `[predicts[root]=draft, predicts[draft]=new_bonus]`. Next step's last_output = `new_bonus`. We need `h(new_bonus_position) = h(position 1)`.
- Draft rejected: appended token is `predicts[root] = correction` (one token only). Next step's last_output = correction. But the correction was inserted at root_position — we don't have its hidden state directly. However the model's hidden state at position 0 of the next forward pass = h(correction). We just need to use that on the next forward.

A simpler design: **always capture hidden state at position 0** (root). That's `h(last_output)` — the state from which the model predicts what comes next. The trained head's job is exactly: given `h(last_output)`, predict the draft.

Implementation:
- Use `CaptureHiddenMode.FULL` (captures all `bs*2` positions).
- In post-processing, take `hidden_states.view(bs, 2, hidden_size)[:, 0, :]` = the root position hidden state per request.
- Pass that through the trained head to produce the **next** step's draft.

This is also what training expects: we trained `h → next-token`, where `h` is the hidden state at the token whose next we want to predict.

### 3c. Training data — re-dump after the layout fix

The hidden states in `/root/gptq_hidden_collect.pt` were captured under the old `draft_token_num=1` layout, where `input_ids = [last_output]`. The hidden state at that single position represents h(last_output) too, BUT under a different KV/attention context (only 1 added token, not 2). The numerical distribution differs subtly:
- Old: req sees prefix + 1 new token (= last_output). `h_old(last_output)` attends to itself + prefix.
- New: req sees prefix + 2 new tokens (= [last_output, draft]). `h_new(last_output)` at position 0 attends only to itself + prefix (the same context as old). Position 1 attends to position 0 + prefix.

So `h_new[:, 0, :]` should be **byte-identical** to `h_old[:]`! The training data CAN be reused without re-dumping, provided we capture position 0 (not position 1).

Verification plan (cheap): after implementing fix, capture both old (ndt=1) and new (ndt=2 pos 0) hidden states on a small batch and compare norms / cosine similarity. If they match, reuse trained head v2.

### 3d. Dump mode adjustment

`_dump_hidden_buffer` should store position 0 of the FULL capture, not the LAST. One-line change in `_forward_verify_k1`:

```python
# raw_hidden_states from CaptureHiddenMode.FULL has shape (bs*2, hidden_size).
# Reshape and take position 0 per req for both head input AND dump.
raw_per_req = raw_hidden_states.view(bs, 2, -1)
root_hidden = raw_per_req[:, 0, :]   # (bs, hidden_size) — for head, dump, training
```

## 4. Risk / impact analysis

### Correctness risk
- Stage 3a (current shipped behaviour) is **also affected by the correctness bug** described in §2. The current submission may be appending wrong tokens whenever Medusa is enabled. Need to check accuracy impact: prior fcloud accuracy runs with MEDUSA enabled scored 77.87% — within noise of baseline 79.29% — suggesting the bug rarely fires (probably because `draft = output_ids[-1]` happens to make `h(last_output)` produce `argmax = last_output` rarely, OR because the failure mode is masked by short generation lengths).
- After the fix, the verify path is structurally identical to standard 2-token spec: cannot corrupt sequences.

### Speed risk
- 2-token verify processes 2x the tokens at the verify forward (still eager, no cuda-graph). This adds a small constant overhead per decode step. We need accept_rate > 0 to break even.
- Expected accept rate from offline head match: 99.91% → if it holds online, ~1.99 tokens/step effective rate → S1 from ~200s → ~100-130s (theoretical 2x; realistic 1.4-1.7x after overheads).

### Submission size risk
- No change to packaged artifacts beyond `medusa_worker.py` (already in submission). Head file is 32 MiB; loads only if SOAR_MEDUSA_HEAD_PATH is set in `prepare_env.sh`. For competition submission we'd add the head file to the tarball (still well under 2GB).

### Rollback
- `SOAR_SPEC_MEDUSA=0` in `prepare_env.sh` disables the entire path → byte-equivalent to v22 baseline.

## 5. Implementation plan (one feature iteration)

1. Refactor `_forward_verify_k1` in both `python/sglang/srt/speculative/medusa_worker.py` and `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py`:
   - Build `draft_token` as `(bs*2,)` flattened `[last_output, draft]` pairs
   - Build `retrive_index`, `retrive_next_token`, `retrive_next_sibling` for 2-node linear chain
   - Build `tree_mask` for 2 added tokens per request (size `sum(seq_len_i+2)` bits/booleans)
   - Build `positions` as `[seq_len_i, seq_len_i+1]` per request
   - Set `draft_token_num=2`
   - Change `capture_hidden_mode` to `CaptureHiddenMode.FULL`
   - After forward, reshape captured hidden states to `(bs, 2, hidden)` and use `[:, 0, :]` for head input AND dump
2. Verify trained head v2 still applies (since position-0 hidden equals old-layout single-position hidden). If similarity < 99%, re-dump and re-train.
3. Run end-to-end on fcloud: restart server with v2 head → speed S1 (small) → if accept_len > 1.0, run S8/Smax and accuracy.
4. Document as `CHANGE_0164_medusa_stage3b_k1_001.{en,zh}.md` (continuation of CHANGE_0164).
5. Update `TEST_RESULTS_TRACKING.md` with new results.

## 6. Estimated fcloud cost

- 1 server restart: ~15 min (cuda graph capture)
- 1 small smoke test (S1 first 5 samples): ~2 min
- If working, full S1/S8/Smax + accuracy: ~10 min
- 1 more server restart only if head re-training needed: ~15 min
- Total: ~30-45 min of compute

## 7. Decision request

Approve the refactor outlined in §3-§5? Specifically:
- [ ] Approve changing `draft_token_num` from 1 to 2 in `_forward_verify_k1`
- [ ] Approve switching `capture_hidden_mode` from `LAST` to `FULL` and using position 0
- [ ] Approve reusing the existing trained head v2 if position-0 hidden similarity verifies ≥99% to the old dump
- [ ] Approve burning ~30-45 min of fcloud cycles for one more iteration
