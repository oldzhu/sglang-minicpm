# CHAT — Phase B FOS Iter 3 (scheduling A/B on iter-2 ckpt)
Date: 2026-05-08 02:50

## User asks (this round)

1. "let us retest NVFP4-FOS iter 1 2 times again for accuracy and speed
   benchmark before make decision" — overrides earlier "park FOS" verdict.
2. Refines: "for `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`, I would like to keep it
   the same as iter 2" (= 16384) — reuse iter-2 ckpt; only revert scheduling.
3. Sets abort gate: "if 1st time accuracy is lower than 70%, then let us stop
   the instance and set `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` as iter1 and then
   requant to test."

## Agent answers / actions

- Patched `benchmark/soar/demo_sala/prepare_env.sh` line 61: replaced
  `export SOAR_TIER1_LONG_CONTEXT=0` (hardcoded) with
  `export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"` so the
  caller can override while the iter-2 default (0) is preserved.
- Committed `fb6ee34d8` and pushed to `minicpm-src/mixed_minicpm_cudagraph`.
- Verified console JWT (5.82 days remaining), resumed instance.
- Discovered fcloud `/root/sglang-minicpm` is not a git clone → `sync` falls
  back to force-copy from a stale snapshot, so the patched file did NOT land.
  Worked around by uploading the patched file directly via base64 over
  `fcloud_exec` to `/root/submission_sim/prepare_env.sh`.
- Restarted server with iter-3 env override:
  `SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1`
  `SOAR_TIER1_LONG_CONTEXT=1 SOAR_TORCH_COMPILE_MAX_BS=24`.
  Live `/get_server_info` confirms chunk=65536, prefill_max_req=4,
  sched_cons=0.8, max_run=24, quant=modelopt_fp4, kv=fp8_e5m2.
- Ran accuracy run-1 (iter-2 ckpt + iter-1 sched). ori = **68.20%**
  (cwe 77.67 / fwe 76.67 / mcq 56.67 / niah 80.00 / qa 50.00).
- ABORT GATE TRIGGERED (<70%). Skipped run-2 + speed bench.
- Paused instance via console API.

## Outcomes

- Tier1 scheduling alone recovers ~6pt of the iter-1→iter-2 regression
  (60.73% → 68.20%) but is not sufficient — still 5pt below iter-1.
- Both calibration content/seqlen AND scheduling contribute to FOS accuracy.
- Next step (per user's pre-set abort plan): iter-4 = re-quantize at
  `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` keeping iter-2's stratified 90
  qa,mcq,cwe content; test with iter-3 server config. Awaiting user
  confirmation to proceed (would need new instance start + ~25 min quantize).

## Cross-references

- Patch commit: `fb6ee34d8`.
- Doc: [CHANGE_0151_phase_b_four_over_six_002.en.md](../CHANGE_0151_phase_b_four_over_six_002.en.md)
- Test row: TEST_RESULTS_TRACKING.md → NVFP4-FOS-3.
- Predictions: fcloud `/root/data/outputs/20260508_025542/predictions.jsonl`.
- Memory: `/memories/session/phase_b_fos_result.md` (iter-3 section appended).
