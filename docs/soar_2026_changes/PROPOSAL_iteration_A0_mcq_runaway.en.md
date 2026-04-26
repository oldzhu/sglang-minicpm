# PROPOSAL: Iteration A (NEW priority) — Fix mcq/qa runaway generation

**Date**: 2026-04-23
**Trigger**: Official v18 result → `acc_ori=76.64%, C=0 (eliminated), S1=586s, S8=1089s, Smax=2864s`
**Supersedes priority of**: [PROPOSAL_iteration_A_revised_runtime_tuning.en.md](PROPOSAL_iteration_A_revised_runtime_tuning.en.md) (kept as lower-priority follow-up)
**Status**: PROPOSAL — awaiting user approval

---

## 1. Smoking gun from code inspection

### Eval script (`benchmark/soar/demo_sala/eval_model_001.py`)

| Line | Code | Meaning |
|---|---|---|
| 352 | `outputs = model.generate(inputs, max_out_len=65536)` | **All 5 tasks use max_tokens=65536**, including mcq (should be ≤ 50) |
| 353 | `chat_template_kwargs={"enable_thinking": True}` | **Thinking mode ON for all tasks**, including mcq |
| 79–92 | `_get_potential_stop_words` | Stop list = tokenizer EOS only; no `</think>` terminator, no per-task stops |
| 178 | `extract_final_answer(pred)` splits on `</think>` | If `</think>` never emitted (because thinking overflows `max_tokens`), extractor returns the entire thinking blob → mcq extraction fails |

### Observed behavior (Test 34a, 150 samples, concurrency=32)

| Task | Accuracy | avg_in tokens | avg_out tokens | Expected avg_out |
|---|---|---|---|---|
| **mcq** | **56.67%** | 270 | **10,946** ❌ | 1–50 |
| qa | 46.67% | 71,569 | 104 | ✓ |
| niah | 100% | 73,983 | 360 | ✓ |
| cwe | 85.33% | 74,163 | 11,350 | a few K |
| fwe | 98.89% | 68,154 | 13,806 | a few K |

Every mcq generates 200–1000× the tokens it should. At concurrency=32 with unbounded max_tokens this alone explains:
- **Official Smax=2864s**: queue stalls under long-running mcq chains blocking new prefills
- **Official S8=1089s**: same effect at concurrency 8
- **Official acc 76.64% → C=0**: mcq chains overflow `max_tokens` and get truncated mid-thinking, never producing a final letter answer; scorer returns 0

## 2. Strategic pivot from prior roadmap

### What the v18 result tells us

| Previously believed | Actual evidence from v18 |
|---|---|
| Local S1/S8/Smax ≈ 110/40/33s reflects official | **False**: official is 586/1089/2864s — 5–86× slower |
| Accuracy floor ~77–79% is noise-dominated and safe | **False**: 76.64% official shows we're on the C=0 side of the knife edge |
| Runtime tuning (torch.compile, cuda-graph) is the speed lever | **False**: kernel-speed is not the bottleneck — runaway generation is |
| Tests 29–33 per-task variance (mcq 40–96%) was eval noise | **Partially false**: the LOW-end is structural (thinking overflow), the HIGH-end is lucky early `</think>` emission |

### Implications for priority ordering

Every prior speed knob (torch.compile, Marlin tiles, scheduling) affects **per-step latency**. None of them reduce the **number of steps**. When the model emits 10,946 steps for mcq instead of 10–50, a 10% per-step speedup saves only 10% — but capping mcq output to 256 tokens saves 97.6%. The multiplicative gap dwarfs every other optimization currently on the table.

**Corollary**: the right first question is no longer "how do we reach top 5" but "how do we get C ≠ 0". Only after C ≥ 0.92 is speed work even meaningful.

## 3. Proposed fix (two layers)

### Layer 1 — Eval-side fix (local benchmarking only; does NOT affect official eval)

These changes are limited to `benchmark/soar/demo_sala/eval_model_001.py`. Their role is to give us a **fast local proxy** for whether a server/model fix is working. They do not ship to official.

**Fix 1.1 — Per-task `max_out_len`**

| Task | Proposed max_out_len | Rationale |
|---|---|---|
| mcq | 1024 | Enough for brief reasoning + `ANSWER: X`; official may cap differently but caps the test cost |
| qa | 512 | Short factual answers only |
| niah | 1024 | Retrieval strings; current avg=360 |
| cwe | 16384 | Keep current tail |
| fwe | 16384 | Keep current tail |
| (default) | 16384 | Prevent untyped tasks from runaway |

Implement as task→`max_out_len` mapping, then group inputs by task and call `generate()` per-group (or pass per-request `max_tokens` in `sampling_kwargs`).

**Fix 1.2 — Add thinking-terminator stop sequences**

Add to stop list: `</think>`, `<|endoftext|>`. If the model fails to emit `</think>` but emits `<|endoftext|>` (a known alternative), we still terminate. Low risk — these are end-markers.

**Expected local effect**: mcq avg_out drops from 11K → ≤1K; overall eval duration drops from 2817s to ≈800–1200s; accuracy rises if the extraction begins seeing `ANSWER: X` text before truncation.

### Layer 2 — Model/server-side fix (affects OFFICIAL eval)

This is the fix that actually matters for our submission score. Two candidate paths, to be validated in sequence:

**Fix 2.1 — Disable default `enable_thinking` in model's chat template** (HIGH reliability, LOW risk)

The MiniCPM-SALA chat template likely has `enable_thinking` controlled by a Jinja variable. We modify the tokenizer's `tokenizer_config.json` (chat_template string) via `preprocess_model.py` so that:

- `mcq` tasks (short prompts without long context) skip thinking
- OR: thinking is always skipped, which officially may hurt accuracy on long-context tasks (cwe/fwe)

Actual behavior must be confirmed by inspecting the model's chat_template string on fcloud.

**Fix 2.2 — Server `--reasoning-parser qwen3`** ❌ **VERIFIED INEFFECTIVE (2026-04-26)**

Review of `benchmark/soar/demo_sala/eval_model_001.py` confirms:

1. The eval reads only `choices[0].message.content` (line 40); it never reads `reasoning_content`.
2. The eval performs the `</think>` split client-side itself in `extract_final_answer` (lines 176-178):
   ```python
   def extract_final_answer(pred):
       parts = pred.split('</think>')
       return parts[-1].strip() if len(parts) > 1 else pred
   ```
3. All scorers (`score_mcq`, `score_exact_match`, …) already strip thinking via that helper.

**Therefore enabling `--reasoning-parser qwen3` would have these effects**:

| Case | Without (current) | With reasoning-parser |
|---|---|---|
| Healthy output (contains `</think>`) | content has full text → split → answer extracted ✅ | content already stripped by parser → answer extracted ✅ (equivalent) |
| **mcq runaway** (max_tokens hit mid-thinking, no `</think>`) | content is thinking blob; regex still occasionally catches a stray `ANSWER: X` mention inside reasoning → partial credit | content becomes **empty string** (everything routed to `reasoning_content`); regex matches nothing → **guaranteed 0** |

Official scoring uses the same harness (the copilot-instructions ban on modifying the eval script exists precisely to preserve this signal alignment), so the conclusion holds online too.

**Net expected accuracy change ≤ 0. Fix 2.2 is removed from the action list.**

**Fix 2.3 — Server-side max-tokens clamp with per-request sampler override** (HIGH risk)

SGLang doesn't have a per-task max_tokens override; any clamp would affect all tasks. **Not proposed** — would hurt cwe/fwe.

**Fix 2.4 — Add `</think>` as a model-level stop token** (MEDIUM reliability, MEDIUM risk)

Modify `generation_config.json` in the preprocessed model to add `</think>` to `eos_token_id`. If model emits `</think>` correctly in most cases, this force-stops thinking and lets the eval extract the post-`</think>` part. If the model doesn't emit `</think>`, no effect.

**Fix 2.5 — Reduce `max_position_embeddings`-bounded thinking via server `--max-tokens` default** (LOW risk, LOW impact)

Uncertain — SGLang may not expose a server-side default max_tokens cap. Skip unless others fail.

## 4. Proposed execution sequence

### Phase 1 — Local evidence gathering (no fcloud)
1. **Inspect** Test 34a `predictions.jsonl` locally (I'll ask user to share a 3-sample excerpt) OR ask fcloud for 5 mcq predictions to confirm runaway pattern.
2. **Inspect** model's chat_template (on fcloud) to determine whether thinking is always-on or conditional.

### Phase 2 — Apply Layer 1 (eval-side) — validates hypothesis
3. Edit `eval_model_001.py` with Fix 1.1 (per-task max_out_len) and Fix 1.2 (extra stops).
4. Commit + sync to fcloud + run accuracy.
5. **Success criterion**: mcq avg_out ≤ 2000, overall acc ≥ 78%, eval duration ≤ 1500s.

### Phase 3 — Apply Layer 2 (model/server-side) — fixes official eval
6. Based on chat_template inspection, pick Fix 2.1 or Fix 2.2 (or 2.4).
7. Edit `preprocess_model.py` or `prepare_env.sh`, re-quantize model, test accuracy + speed.
8. **Success criterion**: local mcq avg_out ≤ 2000 (matching Layer 1 result) — confirms the fix works at model/server level, not just eval level.

### Phase 4 — Resubmit as v19
9. Package new submission with fixes.
10. Expected official result: `acc_ori ≥ 78%` → `C ≥ 0.96`, `Smax ≤ 1500s` (roughly halved).

## 5. Risk table

| Risk | Mitigation |
|---|---|
| Disabling thinking hurts cwe/fwe/qa accuracy | Apply conditionally (only when short prompt) OR keep thinking but add hard `max_tokens` cap |
| Modified chat_template breaks official eval loader | Keep original chat_template as a backup in `preprocess_model.py`; feature-flag with env var |
| ~~`--reasoning-parser qwen3` doesn't match MiniCPM token format~~ | ~~Verify token strings locally against reasoning_parser.py before enabling~~ — Fix 2.2 withdrawn (see above) |
| Eval-side fix (Layer 1) makes local results look great but official still fails | Layer 2 is explicitly designed to transfer; we won't resubmit until Layer 2 passes local reproduction |
| Official eval script is totally different from ours | Low probability given the ~1pt local↔official accuracy alignment |

## 6. Rollback

Each layer is commit-scoped and flag-gated:
```bash
# Layer 1 revert
git checkout benchmark/soar/demo_sala/eval_model_001.py
# Layer 2 revert (config changes)
git checkout benchmark/soar/demo_sala/preprocess_model.py benchmark/soar/demo_sala/prepare_env.sh
```
No kernel rebuild needed for any layer. No wheel rebuild needed.

## 7. What this changes in the strategic roadmap

[STRATEGIC_ROADMAP_TOP5.en.md](STRATEGIC_ROADMAP_TOP5.en.md) needs an update after this iteration lands:
- Iteration A (runtime tuning) demoted to "free cleanup after C is positive"
- New Iteration A-0 (this proposal) = mcq/qa runaway fix, **blocker for everything else**
- Speculative decoding (Iteration C) only meaningful once generation lengths are under control
- Build official-representative speed dataset `speed_full_{s1,s8,smax}.jsonl` (from `perf_public_set.jsonl` at different concurrency caps) as part of Phase 4 prep

## 8. Open questions for user

1. **Approve Phase 1 investigation** (inspect a few Test 34a predictions + chat_template — can be done offline from local file share, no fcloud required)?
2. Prefer Layer 1 first (isolated hypothesis test) then Layer 2, OR combined single test?
3. Any knowledge of MiniCPM-SALA's chat_template format that would pre-decide Fix 2.1 vs 2.4?
4. Should we wait for user to share an mcq prediction sample before writing code, or start implementing Fix 1.1 (per-task max_out_len) now speculatively?
