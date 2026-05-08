# CHANGE 0151 — Phase B FourOverSix, continuation 002

Companion to [CHANGE_0151_phase_b_four_over_six_001.en.md](CHANGE_0151_phase_b_four_over_six_001.en.md).
Continuation 001 had concluded "park FOS"; user overrode and asked for one more
A/B before final park decision.

## Background

Iter 2 changed two variables simultaneously vs iter 1:

| Knob | Iter 1 | Iter 2 |
|------|--------|--------|
| Calibration | sequential 32 samples (no task filter) | stratified 90 samples, `qa,mcq,cwe` only |
| `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` | 4096 | 16384 |
| Scheduling | Tier1 long-ctx (chunk=65536, prefill-max-req=4, sched-cons=0.8) | Conservative Test 12 (chunk=32K, prefill-max-req=1, sched-cons=1.0) |
| `SOAR_TORCH_COMPILE_MAX_BS` | 24 | 8 |

Iter 1 mean ori = ~73%; iter 2 mean ori = ~62% (−11pt). To attribute the
regression we need to isolate scheduling from calibration. Iter 3 holds the
iter-2 ckpt fixed and reverts only the scheduling.

## Implementation

### Server-arg patch

`benchmark/soar/demo_sala/prepare_env.sh` line 61, inside the
`SOAR_QUANT_PROFILE == nvfp4_fos` block, was changed from:

```bash
export SOAR_TIER1_LONG_CONTEXT=0
```

to

```bash
export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"
```

This preserves the iter-2 default (`0` → conservative scheduling) while
letting the caller override it (e.g., `SOAR_TIER1_LONG_CONTEXT=1` from
`fcloud_workflow.py restart-server --env`). `SOAR_TORCH_COMPILE_MAX_BS`
already used the `${VAR:-8}` pattern, so no patch needed there.

Commit: `fb6ee34d8` on `minicpm-src/mixed_minicpm_cudagraph`.

### fcloud sync caveat

`/root/sglang-minicpm` on the fcloud instance is not a git clone (it was
extracted from a `submission_sim.tar` snapshot during instance setup),
so `fcloud_workflow.py sync` falls back to a force-copy from that stale tree
rather than a `git pull` of the patched local file. Workaround: upload the
patched `prepare_env.sh` directly via `base64 | base64 -d` over `fcloud_exec`:

```bash
B64=$(base64 -w0 benchmark/soar/demo_sala/prepare_env.sh)
python3 scripts/fcloud/fcloud_exec.py exec \
  "echo '$B64' | base64 -d > /root/submission_sim/prepare_env.sh"
```

### Iter-3 server launch

```
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

Live `/get_server_info` (verified before accuracy run):

```
chunked_prefill = 65536
prefill_max_req = 4
sched_cons      = 0.8
max_run         = 24
quantization    = modelopt_fp4
kv_cache_dtype  = fp8_e5m2
```

## Pre-set abort gate

Per user instruction:
> if 1st time accuracy is lower than 70%, then let us stop the instance and
> set `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` as iter1 and then requant to test.

If iter-3 run-1 ori_accuracy < 70% → skip run-2 + speed bench, pause instance,
move to iter-4 (re-quantize with calib_seq_len=4096).

## Result — iter-3 run-1

`outputs/20260508_025542/predictions.jsonl`

| Metric | Value |
|--------|-------|
| ori_accuracy (Average Score) | **68.20%** |
| cwe | 77.67% |
| fwe | 76.67% |
| mcq | 56.67% |
| niah | 80.00% |
| qa | 50.00% |
| Total Duration | 3179.25 s |
| Output TPS | 252.24 |

**ABORT GATE TRIGGERED.** Run-2 + speed bench skipped.

## Comparison

| Metric | Iter-1 (mean) | Iter-2 (mean) | Iter-3 run-1 |
|--------|---------------|---------------|--------------|
| ori_accuracy | ~73.1% | ~62.0% | 68.20% |
| Δ vs iter-1 | — | −11pt | −5pt |
| Δ vs iter-2 | +11pt | — | +6pt |

Tier1 scheduling alone recovers ~6pt of the ~11pt iter-1→iter-2 regression.
The remaining ~5pt is attributable to calibration changes
(seqlen 4096→16384 and/or content sequential→stratified-qa,mcq,cwe).

## Next step — iter-4 plan

Re-quantize the FOS ckpt with `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` while
keeping iter-2's stratified 90 qa,mcq,cwe (same content, shorter seqlen).
This isolates seqlen impact while holding content constant — different from
iter-1's sequential 32 mix.

```bash
SOAR_QUANT_PROFILE=nvfp4_fos \
SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
SOAR_GPTQ_CALIBRATION_SAMPLES=90 \
SOAR_GPTQ_CALIBRATION_SAMPLING=stratified \
SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe \
SOAR_GPTQ_CALIBRATION_SEED=20260320 \
python3 preprocess_model.py --input <BF16> --output /root/models/MiniCPM-SALA-NVFP4-FOS
```

Then test with iter-3 server config (TIER1=1, TORCH_COMPILE_MAX_BS=24).

If iter-4 also <70% → calibration *content* (qa,mcq,cwe stratified) is the
dominant variable vs iter-1's broader sequential mix. Then:
- iter-5 option A: drop `TASK_INCLUDE` filter (mix all 5 tasks)
- iter-5 option B: `SAMPLES=32` sequential to mimic iter-1 exactly
- option C: Park FOS for good.

## Validation commands

```bash
# Verify live server config matches iter-1 family
curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool

# Re-run accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## Rollback

Revert commit `fb6ee34d8` (one-line patch). Default behavior for `nvfp4_fos`
is unchanged when caller does not pass `SOAR_TIER1_LONG_CONTEXT`, so this
patch is backward-compatible with iter-2.

## Cross-references

- Continuation 001: [CHANGE_0151_phase_b_four_over_six_001.en.md](CHANGE_0151_phase_b_four_over_six_001.en.md)
- Test row: TEST_RESULTS_TRACKING.md → NVFP4-FOS-3
- Chat log: chat/CHAT_phase-b-fos-iter3_20260508_0250.en.md
