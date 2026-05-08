# CHANGE_0152 — Apply OpenBMB PR #10 `_init_rope` fix for transformers ≥ 4.43 / 5.x compatibility

## Background and motivation

OpenBMB published an official patch (HF discussion
[`openbmb/MiniCPM-SALA/discussions/10`](https://huggingface.co/openbmb/MiniCPM-SALA/discussions/10),
commit
[`f28de5e4`](https://huggingface.co/openbmb/MiniCPM-SALA/commit/f28de5e488b065a06bec8526d9683c14fe83bf7b))
to `modeling_minicpm_sala.py` that fixes a hard load failure on newer
transformers releases:

```
ValueError: Unknown RoPE scaling type default
```

Root cause: `transformers>=4.43` standardizes the `rope_scaling` config field
and auto-fills a missing/None value with `{"rope_type": "default", "factor":
1.0}` at config-load time. The shipped `_init_rope()` only handled the
original `None`, `"linear"`, `"dynamic"`, and `"longrope"` cases and raises on
the new `"default"` marker. With `gptqmodel` 7.0.0 pulling in `transformers`
5.8.0 the model can no longer be loaded for either quantization or serving.

We were already working around this in `preprocess_model.py` via an in-memory
patch (`_install_gptqmodel_minicpm_rope_patch`) that clears
`config.rope_scaling` to `None` *before* `_init_rope` runs in the GPTQ flow.
That hook only covers the offline quant path — sglang serving has no such
hook, so a quantized model loaded with `trust_remote_code=True` would hit the
same `ValueError` after transformers 5.x re-fills the default marker on every
config load.

The official fix is purely a modification to the trust-remote-code modeling
file (no behavioural change, since our config has no `rope_scaling`). We need
to apply it everywhere the model is loaded.

## Rule-compliance statement

- The change is to a **trust-remote-code Python file shipped with the model**.
  It originates from the upstream model author (OpenBMB) and is publicly
  published.
- No change to the eval harness, no per-task generation overrides, no
  forbidden tricks. The patch is byte-equivalent to the upstream fix.
- Behaviour is unchanged when `config.rope_scaling` is missing (our case):
  `MiniCPMRotaryEmbedding` is selected on both old and new code paths.
- Submission constraints respected: zero extra runtime cost, no extra
  weights, no impact on the 2 GB submission cap.

## Detailed implementation plan (before change)

Two `_init_rope()` methods exist in `modeling_minicpm_sala.py`:
- `MiniCPMAttention._init_rope` (around line 879)
- `LightningAttention._init_rope` (around line 2144)

Both share the same buggy structure. The upstream fix (PR #10):

1. Extracts `rope_scaling = self.config.rope_scaling` to a local variable.
2. Treats `None` and `scaling_type in (None, "default")` as **no scaling**
   (falls through to `MiniCPMRotaryEmbedding`).
3. Accepts both legacy `"type"` and new `"rope_type"` keys.
4. Replaces all `self.config.rope_scaling[...]` subscript reads inside the
   `LongRoPE` branch with the local `rope_scaling[...]`.

The model file is downloaded by HuggingFace `trust_remote_code=True` and lives
in the model directory (not in this repo). We therefore apply the patch
**in-place** to the model directory in our `preprocess_model.py`, which is
the single SOAR submission entry-point that touches the model files.

Apply locations:
- `dst` (output) directory after every supported mode (`copy`, `gptq`,
  `nvfp4`) — covers sglang serving.
- The existing `_install_gptqmodel_minicpm_rope_patch` already protects the
  GPTQ load path on `src`, so we do not modify the user-supplied input dir.

Idempotency: a marker comment string
`transformers>=4.43 standardizes rope_scaling` is added by the patch; if
present, the patcher exits early.

## Actual code changes (after change)

- `benchmark/soar/demo_sala/preprocess_model.py`
  - Added `_patch_modeling_init_rope_inplace(model_dir, label)` plus
    `_INIT_ROPE_PATCH_MARKER`, `_INIT_ROPE_OLD_HEADER`,
    `_INIT_ROPE_NEW_HEADER`, `_INIT_ROPE_OLD_ELSE`, `_INIT_ROPE_NEW_ELSE`
    constants matching upstream PR #10 verbatim.
  - Wired calls to `_patch_modeling_init_rope_inplace(dst, ...)` after each
    of the three modes finalizes (`gptq`, `nvfp4`, `copy`).

## Validation

Local test (already executed — see commit log):

```python
# Round-trip test: synthesize a pre-fix file by reversing upstream PR #10 on
# the post-fix file, run the patcher, compare to upstream post-fix byte-for-
# byte. Also verify idempotency by running the patcher twice.
```

Result:

```
[preprocess][init-rope-patch] post-fix idempotency: ... already patched; skip
OK idempotent on already-patched file
[preprocess][init-rope-patch] synth pre-fix: patched ... (replaced 2 _init_rope headers, 2 else-branches)
EXACT MATCH after patch on synthetic pre-fix
```

Quant-time validation (next fcloud round, NVFP4-FOS iter-4):

```bash
# After preprocess runs
grep -n "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# Expect: 2 matches (one per _init_rope method)
```

Runtime validation: server starts cleanly under `transformers==5.8.0` without
the `Unknown RoPE scaling type default` traceback.

## Result summary

| Item                                 | Before                  | After                       |
|--------------------------------------|-------------------------|-----------------------------|
| Load with `transformers>=4.43`        | `ValueError` on quant   | Loads cleanly                |
| Load via sglang `trust_remote_code`  | Risk of `ValueError`    | Patched modeling file        |
| Patch source                          | Local in-memory hack    | Upstream PR #10 verbatim     |
| Behavioural change (rope_scaling=None) | n/a                     | None — same `MiniCPMRotaryEmbedding` |
| Idempotency                           | n/a                     | Marker check, safe re-run    |

## Rollback

```bash
git revert <commit>
```

The patcher is purely additive and gated on a marker; reverting the commit
removes both the function and its call sites. The next preprocess run will
leave `modeling_minicpm_sala.py` untouched.

## Next-step suggestions

1. NVFP4-FOS iter-4: re-quantize at `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`
   keeping iter-2's stratified 90 (qa, mcq, cwe + FOS=1) calibration set,
   under iter-1 Tier-1 scheduling, with abort gate < 70 %.
2. After verifying iter-4 accuracy, consider upstreaming the same patch into
   the bundled HF cache copy (`~/.cache/huggingface/modules/...`) on fcloud
   if any code path bypasses our preprocess flow.
