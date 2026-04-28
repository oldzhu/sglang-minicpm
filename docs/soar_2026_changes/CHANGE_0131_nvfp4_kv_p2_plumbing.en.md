# CHANGE 0131 — NVFP4 KV cache P2 plumbing (dense-only)

**Date**: 2026-04-28
**Status**: Code change committed; awaiting fcloud smoke test.
**Predecessor**: [SURVEY_NVFP4_KV_P1_20260428_1130.en.md](SURVEY_NVFP4_KV_P1_20260428_1130.en.md)
**Branch**: `mixed_minicpm_cudagraph` on `minicpm-src`

## 1. Background and motivation

P1 survey found that ~80% of MXFP4 KV-cache plumbing is already in upstream sglang (server arg, dtype resolution, `MHATokenToKVPoolFP4`, `KVFP4QuantizeUtil`). The remaining gaps live entirely inside our custom `python/sglang/srt/layers/attention/minicpm_backend.py`. P2 closes the must-fix gates so a `--kv-cache-dtype fp4_e2m1 --force-dense-minicpm` smoke test can boot.

Why MXFP4 KV is worth the work:

- Saves **~44% KV memory vs FP8** (1152 vs 2048 bytes per token per layer at MiniCPM-SALA shapes).
- Helps S∞ directly: more concurrent requests at long context.
- ~3 days plumbing vs 3–4 weeks for a tuned W4-FP8 kernel — far better ROI per the [W4-FP8 spike RED verdict](RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md).

## 2. Rule-compliance statement

- **Stays inside SOAR rules**: KV-cache compression is an inference-time optimization, no model retraining, weight quantization unchanged (still GPTQ W4A16).
- **Keeps baseline config invariants**: dense mode (`--force-dense-minicpm`) for first smoke; FP8 path remains the production fallback. No removal of FP8 branches.
- **Accuracy gate**: ≥75% normalized accuracy (user-set tolerance, R11). Current baseline is 79.29%, so max acceptable regression is ~4.3pp.
- **No eval-script edits** — purely server-side plumbing.

## 3. Detailed implementation plan (before change)

### Gates to fix in `python/sglang/srt/layers/attention/minicpm_backend.py`

| # | Site | Current logic | Problem with `fp4_e2m1` | Fix |
|---|---|---|---|---|
| A | L834 sparse-bridge | `if self.kv_cache_dtype_str.startswith("fp8")` casts query/compressed_k to bf16 | FP4 path also returns non-bf16 (or BF16 already from dequant); dense smoke skips this branch entirely (no sparse), but a future sparse re-enable needs this | Generalize gate to "any compressed KV dtype" |
| B | L189 `use_fp8_sparse_scratch` | `kv_cache_dtype_str.startswith("fp8")` | Falls False for FP4 → sparse scratch buffers stay BF16. Dense smoke doesn't hit sparse path; safe for P2 | **No change in P2** (revisit when re-enabling sparse) |
| C | L932, L1144 `set_kv_buffer` | passes `layer.k_scale, layer.v_scale` (FP8 per-tensor scales) | `MHATokenToKVPoolFP4.set_kv_buffer` does `cache_k.div_(k_scale)` BEFORE MXFP4 quant — mis-scales K when FP4 is the storage dtype | Pass `None, None` when `kv_cache_dtype_str == "fp4_e2m1"` |
| D | L952, L1173 `k_descale` build | builds `k_descale = layer.k_scale.expand(...)` whenever `kv_cache_dtype_str != "auto"` | FP4 pool's `_get_key_buffer` already returns dequanted BF16; passing k_descale to FA would double-scale | Skip block when `kv_cache_dtype_str == "fp4_e2m1"` (keep `None, None`) |

### Files touched

- **`python/sglang/srt/layers/attention/minicpm_backend.py`** — only file with source edits.

### Files NOT touched (verified safe)

- `prepare_env.sh` — server arg already supports `fp4_e2m1`; user will toggle for smoke test by editing `SGLANG_SERVER_ARGS` locally.
- `memory_pool.py` — `MHATokenToKVPoolFP4` is upstream-as-is.
- `kvfp4_tensor.py` — `KVFP4QuantizeUtil` is upstream-as-is.
- `model_runner_kv_cache_mixin.py` — auto-routes to FP4 pool when dtype matches.
- Eval script — never edit (per copilot instructions).

### Validation commands

User will run on fcloud (after agent's go-ahead and explicit fcloud start):

```bash
# 1. Edit prepare_env.sh: set SGLANG_SERVER_ARGS to include
#    --kv-cache-dtype fp4_e2m1 --force-dense-minicpm
# 2. Sync and restart server
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
# 3. Quick smoke (small concurrency)
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
# 4. If smoke passes, run accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy
# 5. If accuracy ≥ 75%, run S8/Smax
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### Success / failure criteria

- **Smoke pass**: server boots with `fp4_e2m1` dtype, serves at least one request without exception/garbage.
- **Accuracy pass**: normalized ≥ 75% (≥ 73% absolute, since 75% normalized of best could be higher).
- **Speed pass (target)**: S1/S8/Smax not worse than FP8 baseline by more than 5%; ideally Smax improves (more concurrency from 44% memory savings).
- **Failure modes anticipated**:
  - cudagraph capture incompatibility with `@torch.compile` in `KVFP4QuantizeUtil` → fall back to eager or rewrite as Triton (P2.5).
  - sparse re-enable needed early → handle in CHANGE_0132 (Gap A + Gap B).

## 4. Actual code changes (after change)

All edits in `python/sglang/srt/layers/attention/minicpm_backend.py`. No other source file touched. Both forward_extend and forward_decode code paths covered.

### Edit 1 (L188-193) — add `self.use_fp4_kv_cache` flag in `__init__`

```python
self.kv_cache_dtype = model_runner.kv_cache_dtype
self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
self.use_fp8_sparse_scratch = self.kv_cache_dtype_str.startswith("fp8")
# MXFP4 KV cache flag (kv_cache_dtype = fp4_e2m1). Used to disable
# FP8-style per-tensor scaling (k_scale/v_scale) which would corrupt
# MXFP4's per-16-element block-scaled quantization.
self.use_fp4_kv_cache = self.kv_cache_dtype_str == "fp4_e2m1"
```

### Edit 2 (L839) — Gap A: extend sparse-bridge gate to also catch FP4

```python
# was: if self.kv_cache_dtype_str.startswith("fp8"):
if self.kv_cache_dtype_str.startswith("fp8") or self.use_fp4_kv_cache:
    # cast query / compressed_k / compressed_k2 to bf16
```

### Edit 3 (L940-941, L1160-1161) — Gap C: pass None,None to set_kv_buffer when FP4

Applied at both sites where `set_kv_buffer` is called (forward_extend and forward_decode):

```python
k_scale = None if self.use_fp4_kv_cache else layer.k_scale
v_scale = None if self.use_fp4_kv_cache else layer.v_scale
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, k, v, k_scale, v_scale
)
```

Why: `MHATokenToKVPoolFP4.set_kv_buffer` does `cache_k.div_(k_scale)` *before* MXFP4 quant. Passing the FP8 per-tensor scale would corrupt K/V data (incompatible scaling semantics).

### Edit 4 (L964-965, L1194-1195) — Gap D: skip k_descale build when FP4

Applied at both sites:

```python
if (
    self.kv_cache_dtype_str != "auto"
    and not self.use_fp4_kv_cache       # NEW
    and layer.head_dim <= 256
    # ...
):
    if layer.k_scale is not None:
        # build k_descale, v_descale
```

Why: `MHATokenToKVPoolFP4._get_key_buffer/_get_value_buffer` already returns dequanted BF16 (via `KVFP4QuantizeUtil.batched_dequantize`). Passing a non-None k_descale to FlashAttention would double-scale.

### What we deliberately did NOT change (and why)

- **L189 `use_fp8_sparse_scratch`** — left as-is. With `--force-dense-minicpm` the sparse scorer path is not exercised, so `scratch_only=False` (BF16 fallback) is fine for the smoke test. Will revisit when re-enabling sparse (CHANGE_0132).
- **`memory_pool.py` / `kvfp4_tensor.py`** — upstream-as-is; survey confirmed they already handle MXFP4 layout correctly.
- **`prepare_env.sh`** — user toggles `--kv-cache-dtype fp4_e2m1 --force-dense-minicpm` locally for the smoke test.
- **Eval script** — never modified per project rule.

### Self-consistency checks performed

- `python3 -c "import ast; ast.parse(...)"` → AST parse OK
- `get_errors` → no compile/lint errors
- All 4 patched sites grep-verified to reference `self.use_fp4_kv_cache`
- `kv_cache_dtype_str.startswith("fp8")` references audited: only L189 (`use_fp8_sparse_scratch`, deliberately untouched) and L839 (now `or self.use_fp4_kv_cache`)

## 5. Result summary table

Smoke test (Round 13) — RED. The server cannot boot under the baseline
config (`--force-dense-minicpm` + `--kv-cache-dtype fp4_e2m1`).

| Variant | Baseline (FP8 KV) | New (FP4 KV) | Δ |
|---|---|---|---|
| S1 (s) | 121.71 | n/a — boot blocked | — |
| S8 (s) | 44.09 | n/a — boot blocked | — |
| Smax (s) | 35.86 | n/a — boot blocked | — |
| ori_accuracy | 79.29% | n/a — boot blocked | — |

### 5.1 Boot-time bug chain found and fixed (committed)

| # | Symptom | Root cause | Fix commit |
|---|---|---|---|
| 1 | `AssertionError: KV4 MHA expects attention_backend ['triton','torch_native','flex_attention','trtllm_mha'], got flashinfer` | Stock `_handle_kv4_compatibility()` whitelist does not match MiniCPM custom backend (`minicpm_flashinfer`) and does not match the post-`force_dense_minicpm` rewrite (`flashinfer`). | `fd7e797ea` (minicpm prefix bypass) + `252cc4d64` (force_dense_minicpm bypass) |
| 2 | `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'` from `torch.zeros(..., dtype=fp4_e2m1fn_x2)` | `HybridLinearKVPool` always picks `MHATokenToKVPool`, ignoring the existing `MHATokenToKVPoolFP4` class which allocates `uint8`-packed K/V plus an e8m0 shared-exponent buffer. | `8a0976593` (route fp4 to FP4 pool) |
| 3 | `KeyError: torch.float4_e2m1fn_x2` from `flashinfer.decode.get_batch_decode_uri` | **Architectural blocker.** `--force-dense-minicpm` rewrites `attention_backend = "minicpm_flashinfer"` → `"flashinfer"` in `_handle_model_specific_adjustments`. Standard FlashInfer has no FP4 KV support. | **NOT FIXED** — see §7 below |

### 5.2 Architectural blocker (bug #3)

The CHANGE_0131 plumbing lives entirely in `MiniCPMAttentionBackend`
(`python/sglang/srt/layers/attention/minicpm_backend.py`). When users
pass `--force-dense-minicpm`, `_handle_model_specific_adjustments`
unconditionally rewrites `minicpm_flashinfer` → `flashinfer`, which
bypasses our backend and lands on stock FlashInfer's `BatchDecode` —
which has no compiled FP4 KV decode kernel.

The current production submission **always** passes
`--force-dense-minicpm` (it is part of `SGLANG_SERVER_ARGS` for the
GPTQ baseline), so we cannot reach the FP4 codepath without further
work.

## 6. Rollback instructions

```bash
git revert <commit-hash>
git push minicpm-src mixed_minicpm_cudagraph
```

Or simply set `SOAR_FP4_KV_CACHE=0` in environment (default) — the
opt-in toggle keeps the FP8 baseline path live. The three bug-fix
commits (kv4-compat bypass + memory-pool routing) are independent
hardenings and can stay landed.

## 7. Next-step suggestions

The boot-time fixes (commits `fd7e797ea`, `252cc4d64`, `8a0976593`)
are correct and should remain landed — they are general hardening
for any future MiniCPM + FP4 work.

To actually exercise FP4 KV on the production config, **CHANGE_0132**
must do one of:

- **Option A (preferred — small change):** Skip the
  `minicpm_flashinfer` → `flashinfer` rewrite when
  `kv_cache_dtype == "fp4_e2m1"`. The MiniCPM custom backend
  already handles dense-only batches when `--force-dense-minicpm`
  is set; we just need to keep using it (instead of stock
  flashinfer) so the FP4 plumbing in `minicpm_backend.py` is
  reached. Then re-run the §3 validation pipeline.
- **Option B (heavier — weeks of work):** Add FP4 KV support to
  stock `flashinfer_backend` (requires flashinfer-side kernel
  templates for `dtype_kv = float4_e2m1fn_x2`). Out of scope for
  competition timeline.
- **Option C (sparse re-enable, after Option A passes):**
  CHANGE_0133 re-enables the sparse path under FP4 (Gaps A + B in
  CHANGE_0131 §3).

Recommend proceeding with Option A in the next iteration.
