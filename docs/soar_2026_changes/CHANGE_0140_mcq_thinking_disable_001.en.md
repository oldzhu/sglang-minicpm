# CHANGE_0140 (continuation 001) — mcq thinking disable v2

This document continues `CHANGE_0140_mcq_thinking_disable.en.md`, which described
the v1 attempt. v1 turned out to be a no-op on MiniCPM-SALA-90 because the
model's `chat_template.jinja` never reads `enable_thinking`. v2 replaces the
strategy entirely.

## Background and motivation

- v18/v20 evals show mcq is the only task where reasoning hurts: the long
  thinking phase fills the `max_tokens` budget and the final letter answer is
  truncated → 0 score.
- v1 (CHANGE_0140 original) prepended a Jinja preamble that set
  `enable_thinking=false` for mcq prompts. Confirmed via `grep "enable_thinking"
  chat_template.jinja` that the upstream template has no guards keyed on that
  variable, so v1 changed nothing. fcloud A1 with v1 produced
  `mcq=56.67%`, `avg_out_len=10438` — thinking was still active.
- The model is trained to begin every assistant turn with `<think>...`. We
  cannot disable that with a flag; we must change the actual token sequence the
  model receives so it skips reasoning.

## Rule-compliance statement

- **Submission-side change only.** The patched `chat_template.jinja` lives
  inside the model directory packaged in the submission tarball. The local eval
  script (`eval_model_001.py`) is unchanged — official evaluator runs the
  identical patched template via HF `apply_chat_template()`.
- No private prefix-cache re-enable. No accuracy-leaking shortcut.
- Detection signal is a literal substring (`"LETTER is one of ABCD"`) present
  in 30/30 public mcq prompts and 0 of qa/niah/cwe/fwe — verified locally
  before deployment.

## Strategy v2 — pre-seed a closed `<think>` block (Qwen3 standard trick)

The model was trained on assistant turns shaped like:

```
<|im_start|>assistant
<think>
  ...reasoning...
</think>

<final answer>
<|im_end|>
```

Three structural facts about this training:

1. After `<|im_start|>assistant\n`, the next token is always `<think>`.
2. After `</think>\n\n`, the next tokens are the **final answer**, concise
   and without further reasoning.
3. The training distribution contains **no** assistant turn that re-opens
   `<think>` after `</think>`.

v2 exploits fact (2): for mcq prompts only, after emitting the assistant header
we **pre-seed an empty closed `<think>\n\n</think>\n\n`**. The model is now in
its trained "post-thinking" state and samples the answer directly. We did not
disable thinking; we made the model believe it has already finished.

This is the documented Qwen3 chat-template "disable thinking" idiom and
generalises to any Qwen3-derived chat model that lacks an `enable_thinking`
guard (which MiniCPM-SALA-90 happens to be).

## Where v2 runs

v2 is a **prompt-shaping** patch, executed on the **request ingress** path:

```
client → POST /v1/chat/completions → sglang tokenizer_manager
       → HF tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            └── Jinja template runs HERE; v2 patched block fires
       → returns prompt token ids → scheduler → forward → sampler
       → output streamed back (untouched)
```

It is NOT in sglang Python code, NOT in the eval script, NOT on the output
path. It only changes the input prefix the model sees.

## Detection — why "LETTER is one of ABCD" is reliable

| Task | Public samples containing the substring | Public total | Notes |
|------|-----------------------------------------|--------------|-------|
| mcq  | 30 | 30 | trigger phrase appears verbatim in every mcq prompt |
| qa   | 0  | —  | no false positive |
| niah | 0  | —  | no false positive |
| cwe  | 0  | —  | no false positive |
| fwe  | 0  | —  | no false positive |

Even adversarial-looking strings like "Pick A or B" do **not** contain the
exact phrase, so they don't trigger v2.

## Implementation

### Files changed (commit `0c7767aa8`)

- `benchmark/soar/demo_sala/preprocess_model.py`
  - Bumped marker constant to `{# SOAR_MCQ_THINKING_DISABLE_v2 #}`.
  - Kept `CHAT_TEMPLATE_MCQ_PATCH_MARKER_V1` so revert can clean up legacy v1
    preambles from any older patched model directory.
  - Defined `CHAT_TEMPLATE_MCQ_PATCH_OLD_TRAILING` (the exact unpatched block
    we replace) and `CHAT_TEMPLATE_MCQ_PATCH_NEW_TRAILING` (the v2 replacement).
  - Added shared helpers `_apply_mcq_patch_to_template(template)` and
    `_revert_mcq_patch_from_template(template)` so the runtime A/B toggle and
    the packaging-time preprocess use identical logic.
  - `_patch_chat_template_for_mcq(dst)` now calls the helper and supports both
    HF storage layouts (external `chat_template.jinja` file and embedded
    `tokenizer_config.json["chat_template"]`).
  - Gated by env `SOAR_DISABLE_MCQ_THINKING` (default true).

- `benchmark/soar/demo_sala/toggle_mcq_thinking_patch.py`
  - Rewritten to import constants and helpers from `preprocess_model.py` (no
    duplication).
  - `--mode status` now reports ON(v2) / LEGACY-v1-only / OFF.
  - `--mode off` cleans both v2 and legacy v1 in one pass.

### Replacement target (the unpatched trailing block)

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

### v2 replacement block

```jinja
{# SOAR_MCQ_THINKING_DISABLE_v2 #}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- set _soar_mcq_ns = namespace(disable_think=false) -%}
    {%- if messages is defined and messages -%}
        {%- for _m in messages -%}
            {%- if _m['content'] is defined and _m['content'] is string and 'LETTER is one of ABCD' in _m['content'] -%}
                {%- set _soar_mcq_ns.disable_think = true -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if _soar_mcq_ns.disable_think -%}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif -%}
{%- endif %}
```

Jinja note: `{% set %}` inside `{% if %}/{% for %}` is block-local. We use
`namespace(...)` because mutation of namespace attributes IS visible across
scopes, which is the standard Jinja workaround.

## Local validation (before any fcloud run)

End-to-end Jinja2 render on a stub template that contains the exact
`OLD_TRAILING_BLOCK`:

| Test | Input | Expected tail | Result |
|------|-------|---------------|--------|
| mcq render | message with `LETTER is one of ABCD` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | PASS |
| qa render  | "What is the capital of France?" | `<\|im_start\|>assistant\n` | PASS |
| false-positive guard | "Pick A or B from the options." | `<\|im_start\|>assistant\n` (no think) | PASS |
| apply idempotent | already-patched template | helper returns None | PASS |
| revert | patched → original | byte-equal | PASS |
| revert legacy-v1 | template with v1 preamble | preamble stripped | PASS |

## fcloud A/B testing protocol

Using `toggle_mcq_thinking_patch.py` on the already-quantized model directory:

```bash
# Apply v2
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode on

# Revert (cleans both v2 and legacy v1 if present)
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode off

# Inspect
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode status
```

Restart sglang server after every toggle (chat template is loaded once at
startup).

## Rollback instructions

- **In a deployed submission tarball**: set `SOAR_DISABLE_MCQ_THINKING=false`
  in `prepare_env.sh`. Next `prepare_model.sh` run will skip the patch.
- **In an already-quantized model directory**: run the toggle helper with
  `--mode off`. It removes both v2 and legacy v1 markers idempotently.
- **In source code**: revert commit `0c7767aa8`.

## fcloud regression test (v2 ON, 2026-04-30)

After local Jinja2 render tests passed, we applied v2 to the actual fcloud
model (`MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8`), restarted sglang, and ran
the full public accuracy eval at concurrency=32. Render-test against the live
model tokenizer confirmed the prefix correctly ended with
`<|im_start|>assistant\n<think>\n\n</think>\n\n` for mcq prompts.

| Task | v1 (no-op, A1 result) | **v2 ON (this round)** | v20 baseline (clean) |
|------|----------------------:|------------------------:|---------------------:|
| mcq  | 56.67% (avg_out 10438) | **46.67% (avg_out 12094)** | (high; not separately broken out) |
| qa   | ~50%                   | 53.33%                 | — |
| niah | 100%                   | 96.67%                 | — |
| cwe  | 82.67%                 | 82.00%                 | — |
| fwe  | 100%                   | 98.89%                 | — |
| **Average** | **77.87%**       | **75.51%**             | (acc_ori 80.87, normalized 100) |
| Wall  | (not captured)        | 2880.74 s              | — |

**v2 is a regression**, not a fix:

- mcq accuracy DROPPED from v1's 56.67% → 46.67%.
- mcq `avg_out_len` went UP from 10438 → 12094 — i.e. the model produced MORE
  reasoning tokens despite the pre-seeded closed `<think>`.
- Render-test confirms the input prefix is correct, so the failure is in the
  model's response, not in template plumbing.

**Root cause (hypothesis, confirmed by behaviour):** MiniCPM-SALA-90 does NOT
treat `<|im_start|>assistant\n<think>\n\n</think>\n\n` as a sink state. The
Qwen3 closed-empty-think idiom assumes the model has been distilled / SFT'd on
data where empty `<think>...</think>` blocks are valid "already finished
thinking" turns. MiniCPM-SALA-90 instead **re-opens reasoning** after our
closed `</think>\n\n` (either by emitting another `<think>` or by writing
free-form reasoning text), so we end up with EXTRA tokens, not fewer. The
standard Qwen3 trick does not transfer to this model.

**Why v1 looked similar but "better":** v1 was a pure no-op — it just left
thinking on. v2 actively perturbs the prefix in a way the model wasn't trained
on, which appears to slightly destabilise mcq behaviour relative to the
untouched template (v20 clean).

## Disposition

- v2 was reverted on the fcloud model immediately after this run
  (`toggle_mcq_thinking_patch.py --mode off`); the chat_template now matches
  the v20 submission byte-for-byte.
- fcloud paused (`pause-instance` succeeded after one 504 retry).
- Code stays in the repo as a **documented dead-end**. The toggle helper +
  preprocess wiring + tests are still useful infrastructure if we ever revisit
  thinking-control.
- We will **not ship v2**. The submitted v20 already has `C=1.0` officially
  (acc=100 normalized, acc_ori=80.87) — accuracy is not the bottleneck.

## Status (as of v20 submission window)

- v20 returned `acc=100.0` (`acc_ori=80.87`), `C=1.0`, but `final_score=32.84`
  — accuracy is no longer the bottleneck; speed is.
- v2 was tested and rejected (this document); CHANGE_0140 closes here.
- Pivoting subsequent work to speed (S1/S8/Smax) per
  `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`.

## Next-step suggestions (post-pivot)

1. v2 is closed; do NOT re-test or re-apply it on any future iteration without
   first solving the "model re-opens think after `</think>`" behaviour.
2. Move directly to speed work: open
   `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` and pick
   the top-priority candidates focused on prefill throughput, sparse
   attention, KV cache efficiency (which dominate the hidden long-context
   official speed set).
3. If a future iteration ever DROPS below `C=1.0` and we need an mcq accuracy
   lever, the next idea worth trying is **server-side stop-token** suppression
   (force `</think>` to be sampled early) rather than prefix shaping —
   different mechanism that does not depend on the model's training pattern.
