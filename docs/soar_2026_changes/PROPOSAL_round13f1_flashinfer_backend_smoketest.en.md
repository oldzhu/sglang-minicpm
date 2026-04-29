# PROPOSAL — Round 13f-1: `--attention-backend flashinfer` smoke test on Test 12 baseline

## Status: PROPOSAL (awaiting approval)

## 1. Objective

Quick compatibility + speed sanity check: does running the GPTQ + FP8 KV + dense submission baseline (Test 12) under stock `--attention-backend flashinfer` (the official-site default) produce **any** measurable difference vs `--attention-backend minicpm_flashinfer --force-dense-minicpm`? The user recalls that an early run with the unquantized model under `flashinfer` "seemed to work" — we want to nail down whether this is a viable simpler config or whether it silently misroutes the lightning-attn / sparse_attention layers.

This is **not** a feature commit — it's an exploration test. Result will either become a CHANGE if it wins, or a documented dead-end note.

## 2. Hypothesis

- **Speed:** unlikely to be meaningfully faster. With `--force-dense-minicpm` the per-layer "is this a sparse layer" check is a single boolean read; the dispatch itself costs essentially zero. There is no measurable router overhead to remove.
- **Correctness:** **uncertain.** MiniCPM-SALA has 24 lightning-attn (linear/Mamba-style) layers and 8 sparse_attention layers. The `minicpm_flashinfer` backend is also the host of the **lightning-attn / FLA / SimpleGLA** path (`HybridLinearAttnBackend`). Pure `flashinfer` backend has no such hybrid hooks. Two ways the test can fail:
  1. Server fails to boot (model registers a layer that flashinfer cannot dispatch).
  2. Server boots but accuracy collapses (lightning-attn layers silently degrade to plain MHA on full KV — wrong math).
- **Best plausible outcome:** identical speed to Test 12, identical accuracy → confirms `flashinfer` is a working alias of the dense path; we then know we can run with the official-site default config. Marginal value, but cheap to verify.

## 3. Rule-compliance check

- No model file changes. No eval-script changes. Only swaps a server arg in `prepare_env.sh`.
- Compatible with submission packaging (server arg only).
- Does not introduce any new quantization, kernel, or scheduling change.

## 4. Files to touch

- `benchmark/soar/demo_sala/prepare_env.sh` — add a temporary branch (env-gated) that emits `--attention-backend flashinfer` instead of `--attention-backend minicpm_flashinfer --force-dense-minicpm` when `SOAR_BACKEND_VARIANT=flashinfer` is set. **Default behaviour unchanged.** Default still produces Test 12.
- No source edits inside `python/sglang/srt/`.

## 5. Validation plan

1. Sync to fcloud (no wheel rebuild).
2. Restart server with `SOAR_BACKEND_VARIANT=flashinfer` set (otherwise everything else is the Test 12 config: GPTQ + FP8_e5m2 KV + dense + torch.compile bs=8 + Test 20 server args).
3. **Tier 1 (boot smoke):** does the server come healthy within 60s? If no → log error class, abort, mark dead-end.
4. **Tier 2 (accuracy smoke):** if server boots, run `accuracy` on `--max-concurrent 8` (cheap variant). If `ori_accuracy < 75%` → abort, dead-end.
5. **Tier 3 (full):** if Tier 2 passes, run full accuracy + S₁/S₈/S∞.
6. Compare against Test 12 reference (S₁=121.71s, S₈=44.09s, S∞=35.86s, ori_acc=79.29%).

## 6. Pass / fail criteria

- **Pass (adopt as alternative):** ori_accuracy within ±1pt of Test 12 AND each speed tier within ±2% of Test 12. Adopt only if there's a packaging-related reason to prefer the official default.
- **Neutral (document & drop):** identical results — `flashinfer` is just a working alias, no upside, do not change baseline.
- **Fail (dead-end):** server fails to boot, or accuracy drops > 1pt, or speed regresses > 2% on any tier. Stop, write a one-paragraph note in TEST_RESULTS_TRACKING explaining the symptom, and do not pursue further.

## 7. Risk

- **Risk to current baseline: zero.** Default branch in `prepare_env.sh` is unchanged; only a non-default env-gated branch is added.
- **Cost:** ~1 fcloud round (1 boot + 1 accuracy + 3 speed runs).

## 8. Rollback

If the env-gated branch is judged messy, delete those lines from `prepare_env.sh`. No source edits to revert.

## 9. Next-step suggestions

This proposal is paired with `CHANGE_0136_minicpm_sparse_dense_len_flag.{en,zh}.md` (Option 3 — the genuinely interesting path). If Round 13f-1 passes, it does not affect Round 13f-2 / CHANGE_0136 work; if it fails, we just keep `minicpm_flashinfer --force-dense-minicpm`.
