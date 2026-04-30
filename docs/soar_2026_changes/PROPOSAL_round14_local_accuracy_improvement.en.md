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
| **A** | **mcq stop-token / max-thinking-tokens cap** | XS | None | +1 to +3 pt mean (variance↓) | Pure server / chat-template change. Safest first step. |
| **B** | **Better generation_config defaults** (temp, top_p, repetition_penalty for non-mcq tasks) | XS | None | +0.5 to +1.5 pt | Per-task overrides via stop tokens, no harness edit. |
| **C** | **GPTQ recalibration with larger / smarter sample set** (currently 90 stratified) | S | None | +0.5 to +2 pt | One-shot offline; submission-time quant cost ≤ 5h budget. |
| **D** | **Selective layer keep-bf16** (un-quantize the most acc-sensitive linears) | S | Small (model size, prefill speed) | +1 to +2 pt | Already partial via sparse_qkv_w8; can extend to o_proj, gate. |
| **E** | **Promote KV from FP8_e5m2 → FP8_e4m3** | XS | Small (kernel support) | +0 to +1 pt | e4m3 has more mantissa, better for KV; need flashinfer support check. |
| **F** | **Promote KV from FP8 → BF16** with reduced max_running_requests | XS | **Large** (memory, batch↓) | +0.5 to +1.5 pt | Last-resort acc lever; expect S8/Smax to regress. |
| **G** | **AWQ (instead of GPTQ) calibration** | M | None | unknown ±2 pt | AWQ often better at int4 but our model is mostly w8; gain uncertain. |
| **H** | **SmoothQuant pre-pass before GPTQ** | M | None | +0.5 to +1.5 pt | Activation smoothing reduces quant error on outlier channels. |
| **I** | **Speculative decoding (eagle3 / draft-model)** for mcq fast-path | L | Medium (acc could drop if draft is poor) | acc neutral, speed +10–25% | Already scaffolded in CHANGE_0090; revisit after A–E. |

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
4. **Option E**: try `--kv-cache-dtype fp8_e4m3` if flashinfer attention path supports it on SM120 (confirm in sgl-kernel + flashinfer code). Pure runtime flip.

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

## 4. Open questions for the user

1. **Approve Phase 14.1 first?** It is XS effort, zero speed risk, and addresses the primary variance source. We can package a v20.1 candidate within one fcloud round if it lands.
2. **Calibration cost budget**: official requires quantization + eval ≤ 5h. Current 90-sample run takes ~20–30 min on fcloud H800-class; 200 samples is still well within budget. Confirm OK.
3. **KV dtype gamble**: are you OK with us probing `fp8_e4m3` (Option E) as a side-experiment during Phase 14.1? Independent variable.
4. **Submission cadence**: do you want one official submission per phase (14.1, 14.2, …) or wait until 14.1+14.2 land before re-submitting? Each submission burns one `team-beta` slot; current rank #19 (score 56.63), gap to #5 ≥79.55 = ~40% improvement still needed.

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
