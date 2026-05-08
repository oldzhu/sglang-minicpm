# CHANGE 0151 — Phase B FourOverSix, continuation 005

Companion to [CHANGE_0151_phase_b_four_over_six_004.en.md](CHANGE_0151_phase_b_four_over_six_004.en.md).

Iter-6 follows iter-5 (71.24%) and runs **Option A2**: keep iter-5's
calibration recipe (`SAMPLES=32 sequential`, `MAX_CALIB_SEQ_LEN=4096`,
default `TASK_INCLUDE=qa,mcq,cwe`) but **disable FourOverSix**
(`SOAR_NVFP4_FOUR_OVER_SIX=0`). Goal: test whether FOS itself causes the
residual ~1.89pt gap to iter-1 (~73.13%).

This iteration also delivers a small infrastructure fix (Option C) so future
NVFP4 quantizations are self-contained.

## Background

Open question after iter-5: with calibration recipe pinned to iter-1's
configuration, iter-5 reached 71.24% — still ~1.89pt below iter-1. Two
hypotheses:

1. **FOS is neutral or harmful** → iter-1's 73.13% was achievable without
   FOS, and our residual gap is just FOS overhead.
2. **FOS is positive** → iter-5 ≈ iter-1 modulo fcloud variance, and the gap
   is noise.

Iter-6 tests this directly by re-quantizing with `FOS=0` and the same
calibration recipe.

## Code change (Option C — tokenizer persistence)

`benchmark/soar/demo_sala/preprocess_model.py::run_nvfp4_quantization`
streams the per-Linear FP4 weights to `dst` but never calls
`tokenizer.save_pretrained(dst)`. The local `tokenizer` is also `del`-ed
mid-function (~line 1587) to free the calibration-text closure. Result: every
NVFP4 quant required a manual `cp` of `tokenizer.json`,
`tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json` from
`src` before the chat-template patch could run.

Fix: re-load the tokenizer from `src` after the streaming export and call
`save_pretrained(dst)` so the dst is self-contained.

```python
# preprocess_model.py, after the config.json rewrite block:
try:
    _tok_for_save = AutoTokenizer.from_pretrained(
        str(src), trust_remote_code=trust_remote_code
    )
    _tok_for_save.save_pretrained(str(dst))
    print("[preprocess] NVFP4 tokenizer.save_pretrained complete", flush=True)
except Exception as _tok_exc:
    print(
        f"[preprocess] NVFP4 tokenizer.save_pretrained FAILED: {_tok_exc!r}",
        flush=True,
    )
    raise
```

Commits: `39c0045c5` (initial, used wrong scope), `83921b207` (final, reload
from `src`).

Verification (iter-6 quant log):
```
[preprocess] NVFP4 saving 1067 tensors to /root/models/MiniCPM-SALA-NVFP4-FOS
[preprocess] NVFP4 tokenizer.save_pretrained complete
[preprocess] NVFP4 manual export complete
[preprocess][change-0140] chat_template.jinja patched (v2): mcq prompts ...
[preprocess][init-rope-patch] mode=nvfp4 dst: ... replaced 2 _init_rope headers, 2 else-branches
```

No manual `cp` was needed in iter-6; the chat-template patch and init-rope
patch ran on the first try.

## Iter-6 commands

```bash
# (1) Re-quantize FOS=0 keeping iter-5 calibration recipe
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && \
   cd /root/submission_sim && source ./prepare_env.sh && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=0 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
   SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'

# (2) Restart server (gptq quant-mode auto-swaps to modelopt_fp4 via profile)
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=0 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24

python3 scripts/fcloud/fcloud_workflow.py wait-server

# (3) Accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## Result

Per-task accuracy (run-1 only — abort gate triggered):

| task | iter-5 (FOS=1) | iter-6 (FOS=0) | Δ |
|------|---------------:|---------------:|---:|
| cwe  | 70.67% | 70.00% | −0.67 |
| fwe  | 92.22% | 90.00% | −2.22 |
| mcq  | 53.33% | 43.33% | **−10.00** |
| niah | 90.00% | 90.00% | 0.00 |
| qa   | 50.00% | 43.33% | **−6.67** |
| **avg ori_accuracy** | **71.24%** | **67.33%** | **−3.91** |
| acc duration | 2485.20 s | 3163.71 s | +678.51 s |

Bucket breakdown (iter-6): len_0_4k 43.33% (mcq), len_4k_32k 57.50%,
len_32k_128k 81.25%.

| Iter | Samples | Sampling | TASK_INCLUDE | calib_seq_len | FOS | Scheduling | ori_accuracy |
|------|---------|----------|--------------|---------------|-----|------------|--------------|
| 1    | 32      | sequential | (default qa,mcq,cwe) | 4096 | 1 | Tier1 | ~73.13% |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | 1 | Tier1 | 66.00% (ABORT) |
| 5    | 32      | sequential | qa,mcq,cwe | 4096 | 1 | Tier1 | **71.24%** |
| 6 (this) | 32 | sequential | qa,mcq,cwe | 4096 | **0** | Tier1 | **67.33%** (ABORT) |

## Verdict

**FOS is *protective*, not the source of the residual gap.** Disabling FOS
on the iter-5 calibration recipe lost 3.91pt overall, with the regression
concentrated on **mcq (−10pt)** and **qa (−6.67pt)** — the same short-tail /
runaway-think failure modes we have seen before. cwe / niah / fwe were all
within ±2pt of iter-5, so FOS does not affect long-context retrieval here;
its protection is on the small short-prompt tasks.

The 1.89pt residual gap iter-5 → iter-1 is therefore most plausibly
**fcloud-side variance** (scheduling, mcq runaway-think probability, BF16
non-determinism in the modelopt requantize-resmooth pass), not an
optimization signal.

Acc duration also grew +678s with FOS=0 (longer mcq/qa generations from
worse prompts hitting `max_tokens` more often). This matches the runaway-
think pattern observed in earlier GPTQ Test 30/32/33 runs.

## Decision

1. **FOS stays enabled** for the NVFP4 path.
2. **Iter-5 (71.24%)** is the current best NVFP4-FOS configuration. We
   accept the 1.89pt residual gap to iter-1 as variance.
3. **Park further FOS-flag exploration.** Future NVFP4 work should focus on
   variance reduction (sched determinism, mcq generation_config) or on
   alternate quant approaches, not on the FOS flag itself.

## Rollback

No source rollback needed for the FOS flag (default already 1). To rollback
the tokenizer-save fix:

```bash
git revert 83921b207 39c0045c5
```

## Next steps

| # | Idea | Expected | Effort | Risk |
|---|------|----------|--------|------|
| 1 | Iter-5 ckpt → run-2 + speed bench (variance probe + S1/S8/Smax) | confirm 71.24% reproducibility; obtain speed numbers for submission planning | 1h fcloud | low |
| 2 | mcq-runaway mitigation (generation_config: lower temperature/top-p, stop tokens, max_tokens cap revisit on server side only) | recover some of mcq −10pt seen in iter-6, possibly +1–3pt on iter-5-style runs | medium (server config + safety analysis to avoid breaking long-context tasks) | medium |
| 3 | Compare iter-5 NVFP4-FOS vs current GPTQ baseline at S1/S8/Smax for submission decision (NVFP4 may win on prefill throughput at long ctx; GPTQ wins on S1) | data for go/no-go on NVFP4 submission | 1h fcloud | low |
| 4 | If we want to push NVFP4 accuracy further: experiment with TASK_INCLUDE=all (drop the qa,mcq,cwe filter) at SAMPLES=32 sequential | unknown — may hurt or help; iter-1 used the filter so likely neutral | 1h fcloud | low |

Recommendation: pick (1) next to confirm iter-5 is reproducible, then (3)
to inform the submission package decision.
