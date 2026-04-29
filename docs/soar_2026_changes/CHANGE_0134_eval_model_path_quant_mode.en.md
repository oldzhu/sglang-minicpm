# CHANGE_0134 — Fix eval `--model_path` to honor `quant_mode`; clarify sparse-path activation rules

## Status: APPLIED (no fcloud test yet — instance is shut down)

Commit (pending push): `scripts/fcloud/fcloud_workflow.py` — adds
`_resolve_model_path()` + `--quant-mode` / `--model-path` CLI flags to
`accuracy`, `quick-accuracy`, `full`.

## Background

Round 13e Test 1 (BF16 + sparse + FP8 KV + CHANGE_0133 fix,
concurrency=32) was aborted at 85/150 with many 3000 s read-timeouts.
Decode throughput was steady at ~295–330 tok/s for 8 long-context
requests (~37 tok/s/req). User raised two questions:

1. Default toolkit args use `--attention-backend flashinfer` and no
   `--force-dense-minicpm` — does that mean sparse routing is on by
   default? And: is sparse activated by either
   `--attention-backend minicpm_flashinfer` or `--dense-as-sparse`?
2. The fcloud automation passed `--model_path GPTQ_MODEL` to the eval
   harness even when the *server* served the BF16 (non-quanted) model.
   Could that mismatch cause runaway generation?

This document records the answers and the workflow fix for #2.

## Sparse-path activation rules (clarified)

Sparse routing on the MiniCPM-SALA model requires **both**:

1. `--attention-backend minicpm_flashinfer` (or `minicpm_flashattn`).
   Only the custom `MiniCPMAttentionBackend` knows about
   `mixer_types == ["sparse_attention", "lightning_attn"]`. Stock
   `flashinfer` / `triton` / `flash_attn` backends do not branch on
   `is_sparse_layer`, so the 8 sparse layers are dispatched as plain
   dense attention. Reference:
   [`python/sglang/srt/layers/attention/minicpm_backend.py`](../../python/sglang/srt/layers/attention/minicpm_backend.py)
   L335 `elif attention_backend == "minicpm_flashinfer":` validates this.
2. NOT `--force-dense-minicpm`. This flag does two things:
   - rewrites `attention_backend` `minicpm_flashinfer` → `flashinfer`
     (per CHANGE_0070 / CHANGE_0131 docs), and
   - overrides `has_sparse_attention=False` and clears `sparse_layer_ids`
     at config level (per CHANGE_0132 §"Finding 4").

`--dense-as-sparse` is **not** an activator. It only takes effect when
sparse is already active (rules 1 + 2 satisfied). Inside the sparse
backend it sets `self.dense_len = 0` instead of the default
`hf_config.sparse_dense_len`, which means short batches that would
normally short-circuit to dense compute also go through the sparse
compute path. It increases sparse compute *coverage* on already-sparse
runs; it does not turn sparse on.

### Implication for the official toolkit baseline

Toolkit page lists default `SGLANG_SERVER_ARGS = --disable-radix-cache
--attention-backend flashinfer --chunked-prefill-size 32768`. With
`--attention-backend flashinfer` (rule 1 not met), **the default
toolkit run is effectively dense for sparse layers**. So the
"original-model + default args runs fine on fcloud" observation does
*not* contradict our Round 13e timeout; the official baseline is also
not running the custom sparse backend.

## Bug fix #2: eval `--model_path` mismatch

### Bug

`eval_model_001.py --model_path X` is used **client-side** by the eval
harness to:

- load the tokenizer for prompt formatting and chat template, and
- load `GenerationConfig` to derive `eos_token_id` → `stop` words.

It is **not** used to tell the server which checkpoint to serve.
`fcloud_workflow.py.step_accuracy()` previously hardcoded
`--model_path {MODEL_PATH}` (the GPTQ path) regardless of what the
server actually served. Round 13e served the non-quanted model
(`/root/models/openbmb/MiniCPM-SALA`) but the harness loaded
tokenizer + GenerationConfig from the GPTQ dir
(`/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8`).

If `tokenizer_config.json`, `chat_template.jinja`, or
`generation_config.json` differ between the two dirs (very likely,
because `preprocess_model.py` may rewrite chat template / generation
config for the GPTQ build), this can cause:

- wrong chat template → server sees a malformed prompt → model never
  emits proper EOS → run-on generation hits `max_tokens=65536` → 3000 s
  read timeout. (Exactly the symptom observed in Round 13e Test 1.)
- wrong stop tokens → same effect.
- wrong special tokens / BOS handling → silent accuracy drop.

This is a real bug regardless of whether sparse-attn is also slow.

### Fix

`scripts/fcloud/fcloud_workflow.py`:

1. New helper `_resolve_model_path(quant_mode, model_path)` that
   picks the matching `*_MODEL_PATH` constant just like
   `step_restart_server()` does.
2. `step_accuracy()` and `step_quick_accuracy()` now accept
   `quant_mode` / `model_path` and compute `eval_model_path` via the
   helper before passing to `--model_path`.
3. `workflow_full()` accepts `quant_mode` / `model_path` and forwards
   to both `step_restart_server()` and `step_accuracy()`.
4. CLI: `accuracy`, `quick-accuracy`, and `full` now accept
   `--quant-mode {gptq,fp8_blockwise,noquant}` and
   `--model-path PATH`. Default remains `gptq` for backward
   compatibility with the dense submission baseline (Test 12).

### Validation

Logical:
- `python3 scripts/fcloud/fcloud_workflow.py accuracy --help` shows the
  new flags.
- `python3 scripts/fcloud/fcloud_workflow.py full --help` shows the
  new flags.

Operational (next fcloud session, before any retest, do a quick diff):

```bash
diff -u /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/tokenizer_config.json
diff -u /root/models/openbmb/MiniCPM-SALA/generation_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/generation_config.json
diff -u /root/models/openbmb/MiniCPM-SALA/chat_template.jinja \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/chat_template.jinja 2>/dev/null || true
```

If any of those differ → the prior Round 13e timeouts are at least
partially explained by the mismatch and a re-test is justified. If all
identical → tokenizer mismatch is ruled out and the timeouts are
genuinely sparse-attn slowness.

## Next steps (proposal)

1. (this commit) Push the workflow fix.
2. User starts fcloud.
3. Run the three diffs above to confirm/rule out tokenizer mismatch.
4. If diffs are non-trivial → re-run Round 13e Test 1 with the fix
   (`accuracy --quant-mode noquant`) and CHANGE_0133 already applied.
   - Pass criterion: completes inside 1 h with accuracy ≥ 78 % and
     no read-timeouts.
5. If diffs are empty → close the sparse line, document final result,
   continue dense-path work.

## Risks

- **None to dense path** — the workflow change is automation-side only;
  default `--quant-mode gptq` preserves prior behaviour for all
  existing dense-path tests.
- **None to submission package** — `eval_model_001.py` is the local
  copy of the official harness, not part of the submission tarball.
  We are NOT modifying the eval script (per the eval-script-integrity
  rule).

## Rollback

```bash
git checkout scripts/fcloud/fcloud_workflow.py
```

## Cross-references

- CHANGE_0070 — `--force-dense-minicpm` rewrites `minicpm_flashinfer`
  → `flashinfer`.
- CHANGE_0131 — KV4 backend whitelist & `force_dense_minicpm` bypass.
- CHANGE_0132 — `force_dense_minicpm` config-level effect on
  `has_sparse_attention`.
- CHANGE_0133 — sparse decode `compress_k1/k2` over-fill (already
  applied this round; necessary but not sufficient to make sparse path
  viable at concurrency=32 long-context).
