# CHAT — Phase B FOS iter-6 (FOS=0 ablation + tokenizer fix)

Started: 2026-05-08 16:10 local. Ended: 2026-05-08 16:55 local.

## Round 1 — User: "go"

User invoked the next-step plan from
[CHANGE_0151_phase_b_four_over_six_004.en.md](../CHANGE_0151_phase_b_four_over_six_004.en.md):
Option C (persist tokenizer in NVFP4 export) + Option A2 (iter-6 = FOS=0
ablation). User indicated unavailable mid-session — agent proceeded
autonomously per copilot-instructions.

## Agent actions

1. Implemented **Option C** in
   [benchmark/soar/demo_sala/preprocess_model.py](../../../benchmark/soar/demo_sala/preprocess_model.py)
   by inserting `tokenizer.save_pretrained(str(dst))` after the streaming
   export. First attempt (commit `39c0045c5`) re-used the in-scope
   `tokenizer` local — but that local is `del`-ed earlier (~line 1587) to
   release the calibration-text closure. Iter-6 quant raised
   `UnboundLocalError`. Fixed by re-loading the tokenizer from `src` instead
   (commit `83921b207`).
2. Started fcloud instance, synced (post-pull sha = `83921b207`).
3. Re-quantized FOS=0 with `SAMPLES=32 sequential MAX_CALIB_SEQ_LEN=4096
   SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=0`. Quant
   completed; `tokenizer.save_pretrained complete` log present; chat-
   template + init-rope patches succeeded on first try (no manual cp
   needed).
4. Restarted server with `--quant-mode gptq` + Tier1 long-ctx env;
   wait-server ready after the second pass (~6 min cumulative for
   torch.compile warmup).
5. Ran accuracy: **67.33%**, duration 3163.71 s. Below 70% abort gate →
   skipped run-2 + speed.
6. Paused instance immediately (504 on first try, retry succeeded).

## Outcome

Per-task: cwe 70.00 / fwe 90.00 / mcq 43.33 / niah 90.00 / qa 43.33.
**−3.91pt vs iter-5 (71.24%)**, with regression concentrated on mcq
(−10pt) and qa (−6.67pt). FOS protects short-prompt / runaway-think tasks;
it is not the source of the residual ~1.89pt iter-5 → iter-1 gap.

Decision: **FOS stays enabled by default**; iter-5 is the current best
NVFP4-FOS configuration; further FOS-flag exploration parked.

## Cross-references

- [CHANGE_0151_phase_b_four_over_six_005.en.md](../CHANGE_0151_phase_b_four_over_six_005.en.md)
- [CHANGE_0151_phase_b_four_over_six_005.zh.md](../CHANGE_0151_phase_b_four_over_six_005.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) row `NVFP4-FOS-6`
- Commits on `minicpm-src/mixed_minicpm_cudagraph`:
  - `39c0045c5` preprocess(nvfp4): persist tokenizer files (initial, broken)
  - `83921b207` preprocess(nvfp4): reload tokenizer from src (fix UnboundLocalError)
  - (this commit) docs(0151): iter-6 NVFP4-FOS FOS=0 ablation result

## Open questions / follow-ups

1. Confirm iter-5 reproducibility with run-2 + speed bench.
2. Decide submission package route between iter-5 NVFP4-FOS and current
   GPTQ baseline using S1/S8/Smax data.
3. Consider mcq-runaway mitigation via server-side `generation_config`
   (no eval-script edits, per repo rules).
