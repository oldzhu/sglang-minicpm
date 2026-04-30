# PROPOSAL: Round 14 — Local Accuracy Improvement Plan

**Status**: Discussion / pre-implementation menu (no code yet)
**Scope**: v20 baseline (GPTQ sparse_qkv_w8 + FP8_e5m2 KV + stock flashinfer + torch.compile bs=8 + fused_qk_norm_rope + mixed-chunk)
**Goal**: Move local mean accuracy meaningfully above the new-fcloud noise floor (currently 74.87–78.73% on the new instance, ~75–76% mean) so we maintain a comfortable margin above the official 77% C=0.92 cutoff and have a real shot at the 99% normalized-acc tier.

---

## 0. Why we are doing this now (and not earlier)

1. **v20 unblocks speed**: 13f-1 confirmed +9% / +8% / +6% over v18 with no acc regression (gap +0.30pt within ±1pt noise band). Speed work can continue, but it now has a clear ceiling per the SM120 hardware unless we attack model/quantization.
2. **Accuracy is the multiplier**: official score = `Performance × C`, where C drops from 1.0 → 0.96 at 99%, → 0.92 at 98%, → 0 below 97% (normalized). Each tier crossed = ~4% of total score, which dominates any single speed optimization currently on our catalog.
3. **Local↔official gap is real**: local public-only ≠ official public+private. We must aim for a safety margin, not a tight pass. With local mean ~75–76%, we have **zero margin** against the C=0 cliff if private set is harder.
4. **mcq is the dominant noise driver**: in 13f-4 quartet A1–A4, mcq fluctuated 40–63% across runs of the SAME binary. Reducing mcq variance alone could lift mean acc by ≥3pt without any model change.

---

## 1. Option menu (ranked: low effort/risk → high)

| # | Option | Effort | Risk to speed | Expected acc gain | Notes |
|---|--------|--------|---------------|-------------------|-------|
| **A** | **Disable thinking for mcq via chat template** (`preprocess_model.py`) | XS | None | **+5 to +20 pt on mcq** (mean) | Structural fix to runaway generation. See §1a for mechanism. |
| **B** | **Per-task `max_new_tokens` cap + extra stop sequences via `generation_config.json`** | XS | None | +1 to +3 pt (variance↓) | Companion to A; bounds worst-case output length even if A misses. |
| **C** | **GPTQ recalibration with larger / length-stratified sample set** (currently 90) | S | None | +0.5 to +2 pt | One-shot offline; budget ≤ 1.5 h confirmed by user. |
| **D** | **Selective layer keep-bf16** (un-quantize the most acc-sensitive linears) | S | Small (model size, prefill speed) | +1 to +2 pt | Already partial via sparse_qkv_w8; can extend to o_proj, gate. |
| ~~E~~ | ~~Promote KV from FP8_e5m2 → FP8_e4m3~~ | — | — | **TRIED, FAILED** | Test 30 (2026-04-22, 77.96%): mcq collapsed 96.67→53.33%; e4m3 made runaway worse. **Dropped from menu.** |
| **F** | **Promote KV from FP8 → BF16** with reduced max_running_requests | XS | **Large** (memory, batch↓) | +0.5 to +1.5 pt | Last-resort acc lever; expect S8/Smax to regress. |
| **G** | **AWQ (instead of GPTQ) calibration** | M | None | unknown ±2 pt | AWQ often better at int4 but our model is mostly w8; gain uncertain. |
| **H** | **SmoothQuant pre-pass before GPTQ** | M | None | +0.5 to +1.5 pt | Activation smoothing reduces quant error on outlier channels. |
| **I** | **Speculative decoding (eagle3 / draft-model)** for mcq fast-path | L | Medium (acc could drop if draft is poor) | acc neutral, speed +10–25% | Already scaffolded in CHANGE_0090; revisit after A–D. |

---

## 1a. Why Phase 14.1 (chat-template / generation_config) actually moves accuracy

**User's intuition is right that chat-template tweaks don't change what the model "knows".** They do, however, change **what tokens the model emits and when it stops** — and for this benchmark that is the dominant accuracy lever. Here is the concrete causal chain (all evidence already in repo):

### The mcq runaway pathology (recapped from `PROPOSAL_iteration_A0_mcq_runaway.en.md`)

The MiniCPM-SALA chat template currently sets `enable_thinking=True` by default. For an mcq question the model is supposed to:

```
<think> brief reasoning ... </think>
ANSWER: B
```

The eval harness extractor (`eval_model_001.py:178`) does:

```python
parts = pred.split('</think>')
return parts[-1].strip() if len(parts) > 1 else pred
```

So **scoring depends entirely on whether `</think>` appears in the output.**

Observed in Test 34a (and again in 13f-4 quartet):

| Task | mcq accuracy | mcq avg_out tokens | What happened |
|---|---|---|---|
| Lucky run | 96.67% | ≤ 1,000 | Model emitted `</think>` early → extractor returns the letter |
| Unlucky run | 40–53% | 10,000–11,000 | Thinking chain never closes; either truncates at `max_out_len`, or extractor falls back to the whole blob → no letter found → score 0 |

**This is not random; it is a bimodal failure.** The same binary scores 96% or 53% on mcq depending on whether the sampler happens to emit a single 8-token sequence (`</think>\n\nANSWER:`). That's why 13f-4 quartet showed 40–63% mcq spread on identical config.

### What Option A actually does

`preprocess_model.py` patches the model's `chat_template` (Jinja) so that `enable_thinking` defaults to **False** (or is selected per-task by inspecting the system prompt for `task=mcq`). With thinking disabled the model directly emits:

```
ANSWER: B
```

No `</think>` is needed because the extractor's `if len(parts) > 1` branch is bypassed and the entire short answer is returned. **Failure mode eliminated by construction.** Lower bound on expected gain: the current mcq mean is ~50–55% across the 13f-4 quartet; if Option A pushes it to a stable 90–96% (where the lucky runs already land), the **overall** mean accuracy moves +7 to +10pt (mcq is one of 5 tasks, weighted equally).

### What Option B actually does

Even with A, we want a hard ceiling. `generation_config.json` ships in the model directory and is honored by sglang's tokenizer/sampler. We add:

- `max_new_tokens` per-task profile (mcq ≤ 1024, qa ≤ 512, niah ≤ 1024, cwe/fwe ≤ 16384). Controlled via the same chat-template hook that sets `enable_thinking`, by injecting a per-task stop sequence list.
- Extra stop strings: `"</think>\n\nANSWER:"`, `"\n\nFinal answer:"`, `"<|endoftext|>"` — all already terminator-style; harmless if not present, decisive if present.

B caps the worst-case generation length so even if A's chat-template change misfires on some sample, we can't burn 11k tokens on a single mcq.

### Why this is NOT a harness edit

Both A and B live in files that ship in the submission tarball:
- `preprocess_model.py` → patches `tokenizer_config.json` / chat_template at submission-prep time
- `generation_config.json` → ships next to the model weights

`eval_model_001.py` is untouched. The official evaluator runs its own pristine copy of the harness; what changes is what tokens our model emits when the harness asks it to generate. **This is the only legal way to fix the runaway-mcq problem.**

### Bound on the upside

For mcq alone, top-5 teams reportedly land 95%+ stably. Our current 50–60% mean is squarely a generation-control problem, not a knowledge problem (the lucky runs prove the model can answer). So Phase 14.1 has a well-defined ceiling: **lift mcq from ~55% → ~90%** = **+7 pt overall** at zero speed cost. That is the largest single accuracy lever currently on the table.

---

## 2. Recommended phasing

### Phase 14.1 — Variance taming (no quant change)
1. **Option A**: investigate per-task `max_new_tokens` and stop-tokens via `generation_config.json` + chat template inside `preprocess_model.py`. Hypothesis: mcq runaway thinking is the variance driver. Iteration A-0 (in archived chats) already identified this; never fully fixed.
   - Concretely: add stop strings like `"</think>\n\nAnswer:"` plus a lower per-task token cap when the harness signals task=mcq via system prompt. Verify by reading `predictions.jsonl` after one run.
   - Eval-script integrity rule (`.github/copilot-instructions.md`) bans editing `eval_model_001.py`. We must achieve the cap server-side via `generation_config` / chat template / tokenizer stop tokens — NOT by harness changes.
2. **Option B**: tune `temperature_top_p_top_k` via `generation_config.json` for non-mcq tasks if Option A leaves residual variance.

**Validation**: 4 alternating accuracy runs (variant_default vs variant_phase14_1) under same conditions; pick winner if Δmean > 1pt and Δstd ≤ baseline.

**Expected outcome**: local mean 75–76% → 77–79% with much tighter std (1pt → 0.3pt).

### Phase 14.2 — Calibration upgrade
3. **Option C**: rebuild calibration set.
   - Current: 90 stratified samples from `perf_public_set.jsonl`.
   - Try: 150–200 samples, stratify by task AND by input-length bucket (since official set has more long-context).
   - Try: oversample mcq+qa (the two highest-volume tasks).
   - Try: include a small synthetic long-context probe set (8k/16k/24k tokens) to anchor calibration in the regime we will be evaluated on.
   - Re-run preprocess_model.py + accuracy. Keep best.
   - Cost budget confirmed by user: ≤ 1.5 h; 200 samples on fcloud H800-class fits comfortably (~30–45 min for the GPTQ pass + ~30–40 min for one accuracy run).
4. ~~Option E (fp8_e4m3)~~ — already disproved by Test 30 on 2026-04-22 (77.96% with mcq collapsed to 53.33%). Skip.

**Validation**: 2-3 alternating runs vs Phase 14.1 winner.

### Phase 14.3 — Mixed-precision extensions (only if 14.1+14.2 still leave ≥3pt gap to 78%)
5. **Option D**: extend the sparse_qkv_w8 idea to `o_proj` (already done in some variants) and possibly `gate_proj` for the top-K most-sensitive layers (use Hessian trace as proxy). Document model-size impact.
6. **Option H**: SmoothQuant alpha sweep (0.3, 0.5, 0.7) before GPTQ pass.

**Validation**: full 4-run quartet, compare mean & worst-of-batch.

### Phase 14.4 — Speculative decoding (parked path)
7. **Option I**: revisit eagle3 only after Phase 14.1–14.3 deliver. Goal there is speed, not acc, but it interacts (must verify eagle3 + GPTQ + flashinfer combo holds acc).

---

## 3. What we will NOT do in Round 14

- **Do not edit `eval_model_001.py`** (repo rule). All fixes live in `prepare_env.sh`, `preprocess_model.py`, `generation_config.json`, chat template, and sglang source.
- **Do not switch off the v20 baseline** (`SOAR_BACKEND_VARIANT=flashinfer`, drop force-dense). Phase 14.x compares against v20, not v18.
- **Do not chase a single high-variance lucky run**. Always 2–4 alternating runs before decision.
- **Do not pre-quantize and ship pre-quantized weights** (forbidden by official rules; we ship calibration scripts that run on-site).

---

## 4. User decisions (resolved 2026-04-30)

1. **Phase 14.1 mechanism** — clarified in §1a above: chat-template change is not "hoping the model gets smarter", it deterministically eliminates the `</think>`-emission failure mode that drives the bimodal mcq score (50–96%). Awaiting final go/no-go.
2. **Calibration cost**: ≤ 1.5 h confirmed acceptable. Phase 14.2 plan fits within budget.
3. **KV e4m3 (Option E)**: dropped — already disproved (Test 30 → 77.96% with mcq collapse). Will not retest.
4. **Submission cadence**: official submission **only when both speed and accuracy improve simultaneously** vs the last submitted package. Phase 14.1 alone (acc-only, speed unchanged) → no submission. Bundle 14.1 + a future speed win, OR wait until 14.1+14.2 land and verify speed has not regressed.

---

## 5. Cross-references

- v20 packaging: prepare_env.sh commit `205d8cb91` (push to minicpm-src), tarball at `benchmark/soar/demo_sala/minicpm_sala_submit_v20.tar.gz` (742.8 MB).
- Variance source: `docs/soar_2026_changes/PROPOSAL_round13f4_variance_quantification.en.md`.
- mcq-runaway prior analysis: `docs/soar_2026_changes/PROPOSAL_iteration_A0_mcq_runaway.en.md`.
- Optimization catalog: `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`.
- Strategic roadmap: `docs/soar_2026_changes/STRATEGIC_ROADMAP_TOP5.en.md`.

---

## 6. Recommended next action

**Approve Phase 14.1 (Option A + Option B)** as the next iteration. It is:
- lowest-risk (server-side only, no kernel/model touch),
- highest-leverage (mcq variance is by far the biggest contributor to local std),
- fastest to validate (one fcloud round of 4 runs ≈ 1 hour).

If approved, the next agent step will draft `CHANGE_0140_mcq_variance_taming.{en,zh}.md` with the exact `generation_config.json` / chat-template edits, then proceed to fcloud testing per the standard workflow.
