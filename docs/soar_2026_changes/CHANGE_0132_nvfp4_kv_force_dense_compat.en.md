# CHANGE 0132 — NVFP4 KV + `--force-dense-minicpm` compatibility

**Status**: **Option A INFEASIBLE** after Round 13b code review — see §8.
**Predecessor**: CHANGE_0131 (P2 plumbing — RED on Round 13 smoke due to architectural blocker)
**Branch**: `mixed_minicpm_cudagraph` on `minicpm-src`

## 1. Background and motivation

CHANGE_0131 added MXFP4 KV cache plumbing inside `MiniCPMAttentionBackend`
(`python/sglang/srt/layers/attention/minicpm_backend.py`) and a
`SOAR_FP4_KV_CACHE` opt-in toggle in `prepare_env.sh`. The Round 13
fcloud smoke uncovered three bugs; the first two were fixed in commits
`fd7e797ea`, `252cc4d64`, `8a0976593`. The third bug is architectural
and is the subject of this proposal.

When the user passes `--force-dense-minicpm` (which the production
GPTQ baseline always does), `_handle_model_specific_adjustments` in
`server_args.py` unconditionally rewrites
`attention_backend = "minicpm_flashinfer"` → `"flashinfer"`. This
bypasses our custom backend entirely and lands the request on stock
FlashInfer's `BatchDecode`. Stock FlashInfer has **no compiled FP4 KV
decode kernel**, so cudagraph capture fails with:

```
File ".../flashinfer/jit/attention/modules.py", line 77, in get_batch_decode_uri
    f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
KeyError: torch.float4_e2m1fn_x2
```

Because CHANGE_0131 plumbing lives only in `MiniCPMAttentionBackend`,
the FP4 path is unreachable on the production submission config.

## 2. Rule-compliance statement

- Only edits open-source code we already maintain (`server_args.py`,
  optionally `MiniCPMAttentionBackend`).
- No KV layout / quantization-recipe rule violation — we already passed
  the FP4 KV format review under CHANGE_0131.
- No change to evaluation harness or chat template.
- Submission package shape unaffected (toggle is opt-in via env var;
  default behavior remains FP8 e5m2 KV).

## 3. Detailed implementation plan (before change)

### Option A (originally proposed) — **INFEASIBLE** (see §8 for evidence)

The proposal was: in `_handle_model_specific_adjustments`, skip the
`minicpm_flashinfer → flashinfer` rewrite when
`kv_cache_dtype == "fp4_e2m1"`, on the assumption that the MiniCPM
custom backend has a dense-only codepath we could fall through to.

**The Round 13b code review (§8) showed this assumption is wrong.**
The custom backend is `MiniCPMSparseBackend` and it `raise ValueError`s
when `has_sparse_attention=False` — which is exactly what
`force_dense_minicpm=True` sets. There is no dense codepath. Even
`forward_decode` unconditionally calls `get_topk_for_sparse`. Skipping
the rewrite would not work.

### Option A′ (revised) — heavy fork of MiniCPMSparseBackend

Add a real dense codepath inside `MiniCPMSparseBackend`:

1. Loosen the `has_sparse_attention` guard so the backend can init
   under `force_dense_minicpm`.
2. In `forward_decode`/`forward_extend`, branch on a new
   `is_dense_run` flag: when set, skip `get_topk_for_sparse`,
   skip `sparse_kernel_extension` calls, use the full `page_table`
   directly through the already-imported
   `BatchDecodeWithPagedKVCacheWrapper` (FP4-aware via CHANGE_0131).
3. Update `init_cuda_graph_state` to allocate dense (full-page-table)
   buffers in addition to or instead of sparse ones.
4. Update metadata builders accordingly.

Effort estimate: several hundred lines + careful CUDA graph testing.
Not a one-day patch.

### Option B — heavy alternative

Add FP4 KV support to stock FlashInfer's decode wrapper. This requires
upstream FlashInfer kernel templates for `dtype_kv = float4_e2m1fn_x2`.
Out of scope for the competition timeline.

### Option C (recommended) — park CHANGE_0131/0132

Keep the four boot-fix commits as general hardening (they cost nothing
at runtime when `SOAR_FP4_KV_CACHE=0`, which is the default). Park the
FP4 KV experiment until either:

- We need the KV memory savings for a different code path (e.g. very
  long-context official speed dataset that hits the OOM ceiling).
- Stock FlashInfer adds FP4 KV support natively (Option B).

Pivot to higher-ROI optimizations from
`OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`.

### Validation pipeline (only relevant if Option A′ is undertaken)

```bash
# 1. Sync + restart
git push minicpm-src mixed_minicpm_cudagraph
python3 scripts/fcloud/fcloud_workflow.py sync
SOAR_FP4_KV_CACHE=1 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 2. Smoke S1 (1 sample)
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1

# 3. Accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 4. If accuracy ≥ 75%, full speed
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 4. Actual code changes (after change)

To be filled after the patch is applied.

## 5. Result summary table

| Variant | Baseline (FP8 KV) | New (FP4 KV) | Δ |
|---|---|---|---|
| S1 (s) | 121.71 | TBD | TBD |
| S8 (s) | 44.09 | TBD | TBD |
| Smax (s) | 35.86 | TBD | TBD |
| ori_accuracy | 79.29% | TBD | TBD |
| KV memory / token / layer | 2048 B | 1152 B | −44% |

## 6. Rollback instructions

```bash
git revert <commit-hash>
git push minicpm-src mixed_minicpm_cudagraph
```

Or simply leave `SOAR_FP4_KV_CACHE` unset / `0` (default). Then
`KV_CACHE_DTYPE_ARG` resolves to `fp8_e5m2` and Option A's gate is
never taken.

## 7. Next-step suggestions

Given §8, the recommendation is **Option C — park FP4 KV** for the
competition timeline:

1. Keep the four R13 hardening commits (`d4608f170`, `fd7e797ea`,
   `252cc4d64`, `8a0976593`). They are inert when `SOAR_FP4_KV_CACHE=0`.
2. Pivot to other catalog items (Marlin tile tuning, scheduling,
   speculative decoding variants).
3. Revisit FP4 KV only if:
   - Stock FlashInfer adds an FP4 KV decode kernel (Option B obsoletes
     itself).
   - We obtain evidence the official long-context speed dataset is OOM
     bound on FP8 KV (memory pressure outweighs the kernel-fork cost
     of Option A′).

## 8. Round 13b code review — evidence Option A is infeasible

Verified against `python/sglang/srt/layers/attention/minicpm_backend.py`
on commit `49a7ed5f4`.

### Finding 1 — backend hard-asserts `has_sparse_attention=True`

[L221-230](../../python/sglang/srt/layers/attention/minicpm_backend.py#L221-L230):

```python
self.has_sparse_attention = hf_config is not None and getattr(
    hf_config, "has_sparse_attention", False
)
if not self.has_sparse_attention:
    raise ValueError(
        "MiniCPM model must have sparse attention enabled. "
        "Please ensure the model config has 'has_sparse_attention=True'."
    )
```

Meanwhile, [`model_config.py` L238 / L248](../../python/sglang/srt/configs/model_config.py#L238)
overrides `has_sparse_attention → False` whenever
`force_dense_minicpm=True`. Therefore the custom backend **cannot
instantiate** under the production config.

### Finding 2 — `forward_decode` has no dense branch

[L1130-1300](../../python/sglang/srt/layers/attention/minicpm_backend.py#L1130-L1300)
unconditionally executes:

```python
topk_idx = self.get_topk_for_sparse(
    q_reshaped.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
    1, layer, forward_batch, False,
)
sparse_page_table = sparse_kernel_extension.get_block_table_v3(...)
```

There is no `if not is_sparse_layer: ...` branch and no
`force_dense_minicpm` short-circuit. Every decode call goes through
the sparse top-k path.

### Finding 3 — `forward_extend` `else` branch is also sparse-shaped

[L989-1056](../../python/sglang/srt/layers/attention/minicpm_backend.py#L989-L1056):
the `max(seq_lens) >= self.dense_len` branch runs full sparse top-k,
and the `else` branch still routes through `metadata.sparse_page_table`,
`sparse_cache_seqlens_int32`, and the compressed-key allocator. Even
short sequences traverse sparse plumbing.

### Finding 4 — config-level effect of `force_dense_minicpm`

[`model_config.py` L237-238](../../python/sglang/srt/configs/model_config.py#L238)
and [L247-248](../../python/sglang/srt/configs/model_config.py#L248)
clamp `has_sparse_attention → False` and `sparse_layer_ids → []`.
Both Finding 1's `raise` and the empty `sparse_layer_ids` make
implicit Option A unworkable.

### Conclusion

The combination required by the production submission
(`GPTQ + --force-dense-minicpm + --kv-cache-dtype fp4_e2m1`) has **no
cheap route** to the existing CHANGE_0131 plumbing. The only viable
implementations are:

- Option A′: real dense-codepath fork inside `MiniCPMSparseBackend`
  (heavy, multiple hundreds of LoC + cudagraph re-validation).
- Option B: stock FlashInfer FP4 KV decode kernel (out of scope).

Neither fits the competition timeline. Recommend **Option C: park**.

