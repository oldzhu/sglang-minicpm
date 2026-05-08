# CHAT — Phase B FOS iter 2 (2026-05-07 22:00)

Bilingual sibling: [CHAT_phase-b-fos-iter2_20260507_2200.zh.md](CHAT_phase-b-fos-iter2_20260507_2200.zh.md)

## Round summary

User asked the agent to continue the iter-2 retry after manually hard-restarting
the fcloud instance to clear the 84 GiB GPU leak that survived pause/resume.

## Sequence of actions

1. **Precheck** — GPU 0 MiB used (clean). `/root/sglang-minicpm`,
   `/root/submission_sim`, `/root/data` all gone (hard restart wiped non-persistent
   storage). `/root/models` survived.

2. **Setup** — `python3 scripts/fcloud/fcloud_workflow.py setup` — uploaded
   `submission_sim.tar` (731 MB) and `data.tar.gz` (8 MB), cloned
   `sglang-minicpm`, copied python/ + demo_sala/ scripts. ~10 min.

3. **Quantize attempt 1 (botched env)** — sync exec via `fcloud_exec.py` with
   `source ./prepare_env.sh | tail -20`. **Bug**: pipe forks `source` into a
   subshell, exports never reach the parent. Quantize ran with **defaults**:
   `calibration_samples=32` (sequential, no task filter), `max_calib_seq_len=4096`.
   FOS pct_m4=43.14% identical to iter 1.

4. **Quantize attempt 2 (correct env)** — `source ./prepare_env.sh >/tmp/prep.log
   2>&1; echo SAMPLES=$SOAR_GPTQ_CALIBRATION_SAMPLES …; python3 -u
   preprocess_model.py …`. Verified env: `SAMPLES=90 SEQLEN=16384
   TASKS=qa,mcq,cwe FOS=1`. Quantize OK. **FOS pct_m4=43.14% AGAIN** — confirms
   FOS scale selection is deterministic from weights only; calibration data does
   not affect M=4 vs M=6 picks.

5. **Tokenizer copy** — manual export does not include tokenizer files. Copied
   from source MiniCPM-SALA dir.

6. **Restart server** — conservative-scheduling args confirmed in launch log:
   `--chunked-prefill-size 32768 --prefill-max-requests 1 --schedule-conservativeness
   1.0 --quantization modelopt_fp4 --kv-cache-dtype fp8_e5m2 --enable-torch-compile
   --torch-compile-max-bs 8`. Server up after ~5 min cudagraph capture.

7. **Smoke** — `<think>` block returned, OK.

8. **Accuracy run 1** — `fcloud_workflow.py accuracy` (used `--model-path` flag
   after first attempt failed because default GPTQ tokenizer path didn't exist).
   `fcloud_exec` timed out at 3600s but eval kept running on server side.
   Verified completion via `wc -l predictions.jsonl`. Result: **ori 60.73%** /
   norm 75.92%, duration 3614.90s. Big drops on niah (93→73), fwe (92→70),
   cwe (74→60).

9. **Accuracy run 2** — `ori 63.31%` / norm 79.14%, duration 3631.80s. Failure
   mode rotated: this run's qa collapsed to 36.67% with fwe runaway-think
   (avg_out=15792).

10. **Speed bench** — S1=173.69, S8=45.95, Smax=34.39 (vs iter 1: 175.08/47.37/31.01;
    Smax slowed 11% due to torch_compile_max_bs=8).

11. **Pause instance** — HTTP 504 first call, HTTP 200 retry.

## Outcomes

- **Iter 2 mean ori = 62.02%** vs iter 1 mean ≈ 73.13% — **clean −11pt regression**.
- Variance still ~3pt run-to-run; failure mode rotates between tasks (mcq → niah/cwe → fwe → qa).
- Both iter-2 hypotheses **rejected**:
  - Conservative scheduling did not eliminate runaway-think variance.
  - Longer + qa/mcq/cwe-stratified calibration **made things worse** on
    niah/fwe (likely due to changed activation amax).
- FOS pct_m4 is **deterministic from weights only** (43.14% across all 3
  quantize runs of this session). Calibration changes affect only activation
  amax, not FOS scale picks.

## Verdict and recommendation

**Park Phase B FOS.** Two iterations failed the C ≠ 0 gate. NVFP4 weight-only
quantization (with or without FOS) appears to lose accuracy through activation
quantization on long-context paths (niah/fwe) that the 16-element-block FP4
input quantizer cannot represent. Fixing this needs mixed precision or a
different `input_quantizer` config — both are large efforts.

Recommended next direction: **return to GPTQ + FP8 KV + dense baseline** (T12
family, ori 79.29%, norm 99.11%, C=1.0) and pursue optimization vectors from
[OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
that have not been exhausted.

## Cross-references

- New continuation docs:
  - [CHANGE_0151_phase_b_four_over_six_001.en.md](../CHANGE_0151_phase_b_four_over_six_001.en.md)
  - [CHANGE_0151_phase_b_four_over_six_001.zh.md](../CHANGE_0151_phase_b_four_over_six_001.zh.md)
- New TEST_RESULTS rows: NVFP4-FOS-1b, NVFP4-FOS-2, NVFP4-FOS-2b in
  [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md).
- Branch HEAD: `a6b34a41a` (no new code commits this round; iter-2 prepare_env
  changes were already pushed in the previous session).

## Open questions for the user

1. Approve "park Phase B FOS, return to GPTQ baseline" recommendation?
2. If you want to explore one more NVFP4 angle before parking: do you prefer
   plain NVFP4 with iter-1 default calibration (cheap A/B to test the "iter 1
   was just lucky calibration" hypothesis) OR mixed-precision QKV-int8/MLP-NVFP4
   recipe (large effort, multi-day work)?
