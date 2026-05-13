# PROPOSAL — Medusa K=1 positional off-by-one fix (corrected ndt=2 design)

Date: 2026-05-13
Status: **PROPOSAL — awaiting review, no code changes**
Branch (target): `mixed_minicpm_cudagraph`
Predecessor: [CHANGE_0164 post-revert analysis](CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#post-revert-root-cause-analysis-2026-05-13)
References:
- [python/sglang/srt/speculative/ngram_info.py](../../python/sglang/srt/speculative/ngram_info.py)
- [python/sglang/srt/speculative/ngram_worker.py](../../python/sglang/srt/speculative/ngram_worker.py)
- [sgl-kernel/tests/speculative/test_ngram_utils.py](../../sgl-kernel/tests/speculative/test_ngram_utils.py)
- [python/sglang/srt/speculative/medusa_worker.py](../../python/sglang/srt/speculative/medusa_worker.py) (current Stage 3a)

## Objective

Restore the spec-decode positional invariant in `MedusaWorker._forward_verify_k1` so that:

1. **Stage 3a (ndt=1, no head)** is byte-equivalent to dense decode and recovers the ~0.6 pt regression observed in v23 (78.71 % → 79.29 %).
2. **Stage 3b (ndt=2, trained head)** runs without the catastrophic drift seen in CHANGE_0164 (15.13 % acc). If the head's prediction matches the model's true next-token, two tokens commit per step (S1 speedup); otherwise we fall back to a single correct token per step (no regression).

## Background

The invariant violated by both Stage 3a and the reverted Stage 3b is:

> KV slot `k` contains the KV (token-id + positional embedding `k`) of the token at conceptual position `k` in `origin_input_ids ++ output_ids`.

Current code feeds `output_ids[-1]` (a token at conceptual position `seq_lens - 1`) at position `seq_lens`. The model writes KV at slot `seq_lens` with positional embedding `seq_lens` but token-id of the wrong-position token. Every Medusa decode step is off by 1.

NGRAM doesn't have this bug because its `input_ids[0]` is a **fresh n-gram prediction** for slot `seq_lens`, not a re-feed of the previous bonus.

For Medusa we cannot drop the bonus re-feed (the LM head's prediction is the **only** source of a "valid prediction for slot seq_lens"; the trained head produces a draft for slot `seq_lens + 1`, not for `seq_lens`). The fix is to allocate slots at the **correct** position.

## Rule-compliance statement

- No change to the official eval harness or scoring path.
- No change to model weights, quantization, or KV-cache dtype.
- No change to baseline `--force-dense-minicpm` + FP8 KV cache + Tier1 long-context server args.
- Server-arg surface unchanged.
- Only `MedusaWorker._forward_verify_k1` changes; `NgramVerifyInput` and the sgl-kernel verify path are reused unmodified (and unchanged from upstream).

## Proposed design

### Step 1 — fix the positional invariant for ndt=1 (Stage 3a first; **independent and lower-risk**)

In `_forward_verify_k1`, BEFORE constructing `NgramVerifyInput`:

```python
# Decrement seq_lens by 1 so slot allocation starts at the bonus's actual
# conceptual position. After NgramVerifyInput.verify() the standard
# `batch.seq_lens.add_(accept_length + 1)` brings seq_lens back to its
# correct post-step value (since accept_length is now 0 for ndt=1, the
# +1 advance matches the single committed token).
batch.seq_lens = batch.seq_lens - 1
batch.seq_lens_cpu = batch.seq_lens_cpu - 1
```

Then build `positions = batch.seq_lens.clone()` (now equal to `seq_lens_original - 1`, the correct conceptual position of `output_ids[-1]`).

**Required complementary change inside `prepare_for_verify`**: the call writes `req_to_token[idx, seq_lens : seq_lens + ndt]`. With the decrement, this overwrites the slot at `seq_lens_original - 1` — exactly where the bonus token's KV *should* live. This is correct (and actually *repairs* any stale KV from previous steps where the invariant was violated).

**Sanity check** on `_free_cache` / `out_cache_loc`: `prepare_for_verify` calls `alloc_token_slots(... len(batch.input_ids))` for `page_size=1` — this allocates `ndt` fresh slots from the page pool, independent of `seq_lens`. After verify, `_free_cache` (called inside `verify()`) frees unaccepted slots based on `accept_length`. We need to confirm that freeing the slot at `seq_lens_original - 1` (which we overwrote) doesn't double-free the previous step's allocation for that slot. **This is the main risk and needs a careful read of `get_src_tgt_cache_loc` / `_free_cache` before implementation.**

If `_free_cache` does double-free, an alternative is to (a) skip the seq_lens decrement, (b) keep `positions = seq_lens` (the current convention), but (c) feed `input_ids[0] = output_ids[-2]` instead of `output_ids[-1]` — wait, no, that's nonsense. The correct alternative is sketched in §"Fallback design" below.

### Step 2 — extend to ndt=2 with trained head (Stage 3b)

After Step 1 lands and Stage 3a is verified at ~79.3 % acc:

```python
self.draft_token_num = self.num_heads + 1  # = 2 for K=1

# input_ids per request: [bonus = output_ids[-1], draft = head_pred_or_fallback]
# positions per request: [seq_lens_original - 1, seq_lens_original]
#   (after the same seq_lens decrement from Step 1)
```

`positions`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling` are computed by **calling the canonical kernel** instead of being hand-rolled:

```python
# Build compact (bs, ndt, ndt) tree mask: lower-triangular for the K=1
# linear chain [root, child]. For bs=1 ndt=2 this is [[1,0],[1,1]]
# (flattened to length bs*ndt*ndt).
compact_mask = torch.tensor(
    [[1, 0, 1, 1]] * bs, dtype=torch.bool, device=self.device
).reshape(bs * self.draft_token_num * self.draft_token_num)

reconstruct_indices_from_tree_mask(
    compact_mask,
    batch.seq_lens,           # already decremented in Step 1
    positions,                # output, shape (bs*ndt,)
    retrive_index,            # output, shape (bs, ndt)
    retrive_next_token,       # output, shape (bs, ndt)
    retrive_next_sibling,     # output, shape (bs, ndt)
    bs,
    self.draft_token_num,
)
```

This mirrors `ngram_worker._prepare_for_speculative_decoding` byte-for-byte and ensures we don't reintroduce subtle off-by-ones in the metadata.

For `USE_FULL_MASK=True` (flashinfer requirement), expand `compact_mask` to per-request `(ndt, seq_len_i - 1 + ndt)` shape exactly as NGRAM does (`req_mask = torch.cat([ones(ndt, seq_len-1), compact[i].view(ndt, ndt)], dim=1)`).

`CaptureHiddenMode.FULL`, head forward on `hidden[:, 0, :]` (bonus position), draft caching on `req._medusa_draft_token` — unchanged from the reverted CHANGE_0164 attempt.

### Fallback design (if `_free_cache` doesn't tolerate the seq_lens decrement)

If reading `_free_cache` / `get_src_tgt_cache_loc` reveals that overwriting an already-allocated slot causes a double-free or pool corruption, the alternative is to **not re-feed the bonus** and instead:

- ndt = `num_heads` (= 1 for K=1, = 2 for K=2, ...).
- `input_ids[0..ndt-1]` = head predictions for slots `seq_lens .. seq_lens + ndt - 1`.
- For K=1, this is exactly one head prediction. There's no "root always-accepted" trick — every committed token must match the verify accept rule.
- This requires re-deriving the head training target (was: predict bonus from current hidden; needs to become: predict bonus + draft from previous hidden). Significant retraining cost, deferred unless Step 1 is blocked.

## Detailed implementation plan (pre-code; review checklist)

1. **Read `spec_utils.get_src_tgt_cache_loc` and `NgramVerifyInput._free_cache`** to confirm the seq_lens decrement is safe.
2. **Write a 30-line standalone Python script** that runs one MedusaWorker step in eager mode on a single short prompt, dumps `(positions, seq_lens_in, seq_lens_out, req_to_token[idx, seq_lens-2:seq_lens+ndt+1], input_ids, predicts, accept_length)` to JSON.
3. **Run the same script with the corresponding NgramWorker single-step** (force `ngram_cache.batch_get` to return `[output_ids[-1], output_ids[-1]]`) to confirm both workers produce byte-identical `req_to_token` updates and `predicts[0]`.
4. **Implement Step 1** (ndt=1 with seq_lens decrement). Local pre-flight: server starts, decodes 10 short prompts without crash, accuracy on a 10-request subset ≥ 78 %.
5. **fcloud full accuracy eval** for Step 1.
6. **Implement Step 2** (ndt=2 with `reconstruct_indices_from_tree_mask`). Same pre-flight gates.
7. **fcloud full accuracy + S1/S8/Smax eval** for Step 2.

## Validation criteria

| Test | Pass criterion |
|---|---|
| Step 1 (Stage 3a, ndt=1, fixed positions, no head) | ori_accuracy ≥ 79.0 % (recover the 0.6 pt regression vs v23); S1 ≥ baseline (no spec, no speedup expected) |
| Step 2 (Stage 3b, ndt=2, fixed positions, head v2) | ori_accuracy ≥ 79.0 %; accept_len ≥ 1.5; S1 ≤ 140 s |
| Both | mcq avg_out_len < 4096 (no runaway) on the public 90-set |

## Rollback

Each step is a single commit; revert is a single `git revert`. The Stage 3a (commit `3a15a6de3`) state remains the fallback floor.

## Risks

- **Slot pool corruption** from the seq_lens decrement is the headline risk. Mitigated by the step-2 read of `_free_cache` and the standalone script.
- **`reconstruct_indices_from_tree_mask` may have undocumented constraints** on `draft_token_num` (the kernel test only exercises ndt=4). Mitigated by running the existing test under our build first.
- **Head v2 distribution mismatch** is *separate* from this fix. If Step 1 passes but Step 2 has low accept_rate, we know the position bug is fixed and the head needs more training. This is the desired bisection outcome.

## Next-step suggestions

- Do not start any fcloud work for this proposal until Step 1 of the implementation plan (the standalone single-step verification script) passes locally.
- Step 1 of the proposal alone (ndt=1 fix, no head) is worth landing on its own to recover the 0.6 pt v23 regression, even if Step 2 is deferred indefinitely.
