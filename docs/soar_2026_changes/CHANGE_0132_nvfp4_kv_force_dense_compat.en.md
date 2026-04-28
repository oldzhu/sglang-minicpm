# CHANGE 0132 — NVFP4 KV + `--force-dense-minicpm` compatibility

**Status**: Proposal (awaiting approval)
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

### Option A — preferred (small, low-risk)

In `python/sglang/srt/server_args.py`, inside
`_handle_model_specific_adjustments`, locate the block that rewrites
`minicpm_flashinfer` → `flashinfer` when `force_dense_minicpm` is set,
and **skip the rewrite when `kv_cache_dtype == "fp4_e2m1"`**:

```python
# server_args.py (sketch)
if self.force_dense_minicpm:
    if self.kv_cache_dtype == "fp4_e2m1":
        # Keep MiniCPM custom backend so FP4 KV plumbing in
        # MiniCPMAttentionBackend is reached. The custom backend
        # already supports dense-only batches via its existing
        # force_dense_minicpm code path.
        pass
    else:
        if self.attention_backend == "minicpm_flashinfer":
            self.attention_backend = "flashinfer"
        # ... existing rewrites
```

Why this works: `MiniCPMAttentionBackend` already has a dense-only
code path (it inspects `force_dense_minicpm` internally), and
CHANGE_0131 already gates the FP4-aware logic by
`self.use_fp4_kv_cache`. Keeping `attention_backend == "minicpm_flashinfer"`
under FP4 simply lets that plumbing run.

### Option B — fallback (heavy)

Add FP4 KV support to stock FlashInfer's decode wrapper. This requires
upstream FlashInfer kernel templates for `dtype_kv = float4_e2m1fn_x2`.
Out of scope for the competition timeline.

### Validation pipeline (Option A)

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

1. After Option A is green on dense + accuracy ≥ 75%:
   - Measure actual throughput / max-running-requests gain from the
     freed KV memory.
   - If gain < 5%, deprioritize and pivot to Marlin tile work.
   - If gain ≥ 5%, file CHANGE_0133 to re-enable sparse path under
     FP4 (CHANGE_0131 §3 Gap A + B).
2. If Option A fails accuracy: suspect MiniCPMAttentionBackend dense
   codepath divergence between `minicpm_flashinfer` and stock
   `flashinfer` — investigate per-layer outputs.
