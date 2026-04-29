# CHANGE_0137 — Sparse Prefill Off-by-One in `sparse_page_table` Slice (Discovery + Proposed Fix)

**Status**: discovered while validating CHANGE_0136 sanity; proposed fix not yet applied.
**Discovered**: 2026-04-29 (third boot attempt of `SOAR_SPARSE_DENSE_LEN=524288`).
**Repro commit**: `7a7a568eb` on `minicpm-src/mixed_minicpm_cudagraph`.

## Symptom

First eval prefill request (103 tokens, single sequence, no prefix) crashes at
`python/sglang/srt/layers/attention/minicpm_backend.py:1087`:

```python
metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = \
    page_table[dense_bs, : kv_len] * 2
```

```
RuntimeError: The expanded size of the tensor (103) must match the existing
size (104) at non-singleton dimension 0.  Target sizes: [103].  Tensor sizes: [104]
```

Full backtrace ends at `MiniCPMSparseBackend.forward_extend` after the
`if forward_batch.sparse_batch_size < bs:` branch (i.e., when at least one
batch element is to be processed via the **dense fallback** inside the sparse
backend).

## Root cause

Two different sources of length disagree across the build/write boundary:

| Site | Value used | File / Line |
|------|------------|-------------|
| Allocator (sizes `sparse_page_table.shape[1]`) | `forward_batch.extend_seq_lens_cpu[i]` (else branch) | `minicpm_sparse_utils.py:1421` |
| Writer (slice index `:kv_len`) | `forward_batch.seq_lens_cpu[dense_bs]` | `minicpm_backend.py:1086` |

Under **overlap-mode scheduling** (`event_loop_overlap`, the default for
`disable_overlap_schedule=False`), `seq_lens_cpu[i]` reflects the
**post-step** sequence length — i.e., it has already been incremented by 1 for
the next decode token before the prefill batch's metadata is rebuilt. Therefore
for a request that contributes 103 tokens to this prefill chunk:

```
extend_seq_lens_cpu[i] = 103   (chunk length, used by allocator)
seq_lens_cpu[i]        = 104   (prefix=0 + 103, then +1 from overlap pre-update)
```

`max_sparse_cache_len = max(prev, 103) = 103` → `sparse_page_table.shape[1] = 103`.
But the writer's slice is `[..., :104]`, which on the LHS gets capped at 103
(the row width) and on the RHS reads 104 entries from `page_table[dense_bs, :104]`.
Slice sizes differ → broadcast assignment fails.

## Why this only surfaces *after* CHANGE_0136

Both pre-existing sparse-mode runs and the released submission baseline had at
least one of the following properties that **prevented** the dense-fallback
write at line 1087 from ever firing on a "narrow" `sparse_page_table` row:

1. **Test 12 / submission baseline** — `--force-dense-minicpm` is on, which the
   `server_args` post-init at `server_args.py:1525` rewrites
   `minicpm_flashinfer → flashinfer`. The full-attention backend becomes the
   stock `FlashInferAttnBackend`. `MiniCPMSparseBackend` is **never instantiated**,
   so neither the allocator nor the writer at issue runs. Bug invisible.

2. **`--dense-as-sparse` (used in many baselines)** — forces `self.dense_len = 0`
   inside `MiniCPMSparseBackend.__init__`. The build loop at
   `minicpm_sparse_utils.py:1404` then takes the `seq_lens_cpu[i] >= dense_len`
   branch for **every** request, sizing `max_sparse_cache_len = sparse_topk *
   block_size` (a large number) instead of `extend_seq_lens_cpu[i]`. Row width
   is large enough to absorb the +1 from overlap mode. `sparse_batch_size == bs`
   so the dense_bs branch at `minicpm_backend.py:1057` is skipped entirely.
   Bug invisible.

3. **`SOAR_SPARSE_MODE=1` (default `dense_len = sparse_dense_len = 8192`)** —
   for the public eval set's ~150 samples, prompt lengths range from
   ~100 (mcq) to ~128k (cwe/niah). Many requests have `seq_lens < 8192`, so
   they DO take the else branch and DO end up in `dense_bs_list`. In principle
   the bug is reachable here. In practice Round 13d crashed at boot
   (`current_seed during cudagraph capture`), so we never reached
   forward_extend on a real request to observe it. The bug would have
   triggered if Round 13d had been allowed to start eval.

4. **CHANGE_0136 with `SOAR_SPARSE_DENSE_LEN=524288`** — `dense_len` is larger
   than every realistic prompt, so **every** request takes the else branch in
   the build loop AND every request lands in `dense_bs_list` (because
   `sparse_batch_size = 0 < bs`). The dense_bs writer at line 1087 is now on
   the hot path, and `max_sparse_cache_len = max(prev, extend_seq_lens_cpu[i])`
   is exactly 1 short of the writer's `seq_lens_cpu[i]`. Crash on first sample.

So CHANGE_0136 did not introduce the bug; it merely **uncovered a latent
defect** that the legacy server-args masked. This is consistent with the
broader CHANGE_0133 finding (decode-time `compress_k1/k2` over-fill) — the
sparse path on HEAD has multiple latent off-by-ones that only surface when the
dense routing fraction or prefix-cache pattern shifts.

## Proposed fix (not applied yet)

Make the allocator size the row by the same length the writer will use:

```diff
--- a/python/sglang/srt/layers/attention/minicpm_sparse_utils.py
@@ build_sparse_prefill_metadata
             else:
-                max_sparse_cache_len = max(
-                    max_sparse_cache_len, forward_batch.extend_seq_lens_cpu[i]
-                )
+                # Writer uses forward_batch.seq_lens_cpu[i] (post-overlap-update),
+                # which can exceed extend_seq_lens_cpu[i] by 1. Size by the larger
+                # of the two so dense_bs page-table copy at minicpm_backend.py:1087
+                # never overruns the row.
+                max_sparse_cache_len = max(
+                    max_sparse_cache_len,
+                    int(forward_batch.seq_lens_cpu[i]),
+                )
```

Notes:
- Use `seq_lens_cpu` (full kv length) — not `extend_seq_lens_cpu` (chunk
  length). The writer copies the **entire** page table for the dense_bs row,
  not just the new chunk.
- Cast to `int` so the `max(...)` reduction stays in Python ints (avoids
  spurious 0-d tensor → Python int conversion churn).
- The `seq_lens_cpu[i] >= dense_len` branch at line 1404 already sizes by
  `sparse_topk * block_size` (independent of the request length), so it is
  unaffected by this change.

### Alternative defensive fix (in writer)

If we want belt-and-braces, also clamp the writer slice:

```diff
--- a/python/sglang/srt/layers/attention/minicpm_backend.py
@@ forward_extend
-                metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = page_table[dense_bs, : kv_len] * 2
-                metadata.sparse_page_table[sparse_page_table_idx_start + 1, : kv_len] = page_table[dense_bs, : kv_len] * 2 + 1
+                copy_len = min(int(kv_len), metadata.sparse_page_table.shape[1])
+                metadata.sparse_page_table[sparse_page_table_idx_start, :copy_len] = page_table[dense_bs, :copy_len] * 2
+                metadata.sparse_page_table[sparse_page_table_idx_start + 1, :copy_len] = page_table[dense_bs, :copy_len] * 2 + 1
```

But this hides the size mismatch from downstream consumers (e.g.,
`sparse_cache_seqlens_int32 = (metadata.sparse_page_table != 0).sum(dim=1)`
will then under-count by 1). The allocator-side fix is the correct one.

## Validation plan (for when CHANGE_0137 lands)

1. Apply the allocator-side patch.
2. Re-run CHANGE_0136 sanity (`SOAR_SPARSE_DENSE_LEN=524288`):
   - Server boots OK (already verified).
   - First prefill batch must complete without size error.
   - Observed S₁ wall-clock should be **within ±5%** of Test 12 (since at
     `dense_len=524288` no request is routed sparse; behavior must equal pure
     dense apart from the sparse-allocator overhead).
3. Then proceed with `=65536` conservative and `=16384` aggressive.

## Out of scope for this fix

- The CHANGE_0133 decode-time `compress_k1/k2` over-fill (separate symptom).
- Whether `seq_lens_cpu`-vs-`extend_seq_lens_cpu` divergence is a recent
  regression in overlap-mode scheduling or has always been there. (No
  archeology needed for the fix; the allocator should size by the value the
  writer uses.)

## Cross-references

- Discovery: [chat/CHAT_round13f_change0136_validation_20260429_1410.en.md](chat/CHAT_round13f_change0136_validation_20260429_1410.en.md)
- Test row: TEST_RESULTS_TRACKING.md `R13f-CHANGE_0136-sanity`.
- Related latent bug: [CHANGE_0133_sparse_compress_buffer_oversize.en.md](CHANGE_0133_sparse_compress_buffer_oversize.en.md).
- Trigger doc: [CHANGE_0136_minicpm_sparse_dense_len_flag.en.md](CHANGE_0136_minicpm_sparse_dense_len_flag.en.md).
