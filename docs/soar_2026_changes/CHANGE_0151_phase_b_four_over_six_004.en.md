# CHANGE 0151 — Phase B FourOverSix, continuation 004

Companion to [CHANGE_0151_phase_b_four_over_six_003.en.md](CHANGE_0151_phase_b_four_over_six_003.en.md).

Iter-5 follows the iter-4 verdict (FOS parked; calibration *content* is the
regression source) and tests **Option A**: re-quantize with iter-1's
calibration count and sampling mode while keeping FOS=1, calib_seq_len=4096,
and Tier1 scheduling.

## Background

Iter-4 isolated `calib_seq_len` and ruled it out as the regression source
(66.00% with seqlen=4096 vs 68.20% with 16384, both stratified-90
qa,mcq,cwe). The hypothesis remaining is that **calibration sample count and
sampling mode** dominate: iter-1 used `SAMPLES=32 sequential` (default
mixed-task distribution); iter-{2,3,4} used `SAMPLES=90 stratified` with
`TASK_INCLUDE=qa,mcq,cwe`.

| Iter | Samples | Sampling | TASK_INCLUDE | calib_seq_len | Scheduling | ori_accuracy |
|------|---------|----------|--------------|---------------|------------|--------------|
| 1    | 32      | sequential | (default qa,mcq,cwe via prepare_env) | 4096 | Tier1 | ~73.13% |
| 2    | 90      | stratified | qa,mcq,cwe | 16384 | Conservative | ~62.02% |
| 3    | 90      | stratified | qa,mcq,cwe | 16384 | Tier1 | 68.20% |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | Tier1 | 66.00% (ABORT) |
| 5 (this) | **32** | **sequential** | qa,mcq,cwe (default) | 4096 | Tier1 | **71.24%** |

Note: `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` defaults to `qa,mcq,cwe` in
prepare_env.sh (line 210), so dropping the explicit env var still yields the
same task filter as iter-{2,3,4}. Only `SAMPLES` and `SAMPLING` differ from
iter-4.

## Implementation

No source patches. Per-invocation env override only:

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && \
   cd /root/submission_sim && source ./prepare_env.sh && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
   SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'
```

Resulting calibration summary (logged):

```
calibration_sampling={"available": 90, "mode": "sequential",
  "records_after_task_filter": 90, "records_before_task_filter": 150,
  "seed": 20260320, "selected": 32, "selected_buckets": {"all": 32},
  "task_balance": true, "task_filter_applied": true,
  "task_include": ["qa","mcq","cwe"], "use_prompt_tokens": true}
```

`pct_m4 = 43.14%` (unchanged from iter-4 — FOS scale-selection statistics
turn out to be near-identical regardless of which 32 / 90 calibration samples
within the same 90-record qa,mcq,cwe pool are used).

### Tokenizer-files copy fix (one-shot)

The streaming-export NVFP4 path in `preprocess_model.py` does not call
`tokenizer.save_pretrained(dst)`, so a fresh `dst` directory ends up missing
`tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` /
`special_tokens_map.json`. This blocked the first server start with:

```
ValueError: Unrecognized configuration class
  ...MiniCPMSALAConfig... to build an AutoTokenizer.
```

Fix applied for this run (manual one-shot, **not** persisted):

```bash
cp /root/models/openbmb/MiniCPM-SALA/tokenizer.json \
   /root/models/openbmb/MiniCPM-SALA/tokenizer.model \
   /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json \
   /root/models/openbmb/MiniCPM-SALA/special_tokens_map.json \
   /root/models/MiniCPM-SALA-NVFP4-FOS/

# Re-run mcq chat-template patch (change-0140) since it had been skipped
# when tokenizer_config.json was absent at the original preprocess pass.
python3 -c "
import sys; sys.path.insert(0, '/root/submission_sim')
from pathlib import Path
import preprocess_model as pm
pm._patch_chat_template_for_mcq(Path('/root/models/MiniCPM-SALA-NVFP4-FOS'))
"
```

A follow-up source patch is recommended (next iteration) to call
`tokenizer.save_pretrained(dst)` inside `run_nvfp4_quantization` so future
NVFP4 quants are self-contained.

### Server launch

`--quant-mode gptq` selects the NVFP4 branch in prepare_env.sh (which
swaps `--quantization gptq_marlin` → `--quantization modelopt_fp4` based on
`SOAR_QUANT_PROFILE=nvfp4_fos`). `--quant-mode noquant` is wrong for NVFP4
(it omits `--quantization`, which causes the loader to default to
`ModelOptFp8Config` and crash with "ModelOptFp8Config only supports static
FP8 quantization in SGLang"). This was discovered and corrected during this
iteration.

```bash
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

## Result — iter-5 run-1

`outputs/20260508_071351/predictions.jsonl`

| Metric | Value |
|--------|-------|
| ori_accuracy (Average Score) | **71.24%** |
| Total Duration | 2485.20 s |
| Total Tokens | In=8,644,406  Out=938,189 |
| FOS pct_m4 | 43.14% |

Per-task:

| Task | Iter-3 | Iter-4 | **Iter-5** |
|------|--------|--------|-----------|
| cwe  | 77.67  | 66.67  | **70.67** |
| fwe  | 76.67  | 80.00  | **92.22** |
| mcq  | 56.67  | 46.67  | **53.33** |
| niah | 80.00  | 93.33  | **90.00** |
| qa   | 50.00  | 43.33  | **50.00** |

Iter-5 passes the >70% abort gate (71.24%). Run-2 + speed bench were not
executed (single accuracy run was the user's defined scope for this round).

## Comparison vs prior iterations

| Iter | Samples | Sampling | TASK_INCLUDE | seqlen | ori_acc | Δ vs iter-1 |
|------|---------|----------|--------------|--------|---------|-------------|
| 1    | 32      | sequential | qa,mcq,cwe | 4096 | ~73.13% | — |
| 2    | 90      | stratified | qa,mcq,cwe | 16384 | ~62.02% | −11pt |
| 3    | 90      | stratified | qa,mcq,cwe | 16384 | 68.20% | −5pt |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | 66.00% | −7.13pt |
| **5** | **32** | **sequential** | qa,mcq,cwe | **4096** | **71.24%** | **−1.89pt** |

Going from 90-stratified → 32-sequential at the same `qa,mcq,cwe` pool
recovers **+5.24pt** vs iter-4 (with FOS=1 still on). This confirms that
**within the qa,mcq,cwe-restricted pool, the smaller sequential-32 sample
selection is materially better calibration content for NVFP4 weights** than
stratified-90 — likely because:

- `samples=32 sequential` = first 32 records of the public set, which
  happens to mix the three task buckets in roughly the same proportion as
  the eval distribution.
- `samples=90 stratified` over-represents long-output / high-FOS-score
  samples, biasing the per-channel scale statistics for downstream layers.

A residual ~1.9pt gap to iter-1's 73.13% remains — most plausible candidates
are FOS itself (iter-1 likely had `FOS=0`) and/or fcloud-instance variance.

## Next-step options

| Option | Description | Cost | Rationale |
|--------|-------------|------|-----------|
| A1 | Iter-5 + run-2 + S1/S8/Smax speed bench (verify variance + speed footprint of the new ckpt) | 1 fcloud session ~70 min | Confirm the 71.24% is reproducible; baseline speed for FOS-32 ckpt. |
| A2 | Re-quantize with `SAMPLES=32 sequential` and `FOS=0` (plain NVFP4) | 1 fcloud iter | Tests whether the residual ~1.9pt gap to iter-1 is FOS-induced. If A2 ≥ 73%, FOS itself is the regression source and should be permanently parked. |
| B | Drop `TASK_INCLUDE` filter entirely (use all 5 tasks) at `SAMPLES=32 sequential FOS=1` | 1 fcloud iter | Tests whether the qa,mcq,cwe filter (vs full 5-task mix) still costs accuracy at the smaller sample count. |
| C | Persist tokenizer-save fix in `preprocess_model.py` | trivial code patch | Removes the manual cp step from future NVFP4 iters. |

Recommendation: **C first** (trivial), then **A2** (cheapest direct test of
"is FOS itself worth keeping?"). If A2 closes the gap to iter-1, FOS is
permanently parked.

## Validation commands

```bash
# Verify _init_rope patch
grep -c "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# Expect: 2

# Verify tokenizer files present
ls /root/models/MiniCPM-SALA-NVFP4-FOS/tokenizer*

# Re-run accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## Rollback

No source changes. To restore iter-4 ckpt: re-run the iter-4 quant command
in continuation 003 (`SAMPLES=90 SAMPLING=stratified`).

## Side-effect: setup script now does a real `git clone`

This iteration also fixed the long-standing `fcloud_workflow.py sync` issue
by replacing the tarball-upload path in `step_setup` with a real
`git clone --depth 1 --branch mixed_minicpm_cudagraph https://github.com/oldzhu/sglang-minicpm.git`.
After the clone, `sync` uses `git pull` to fetch new commits and only
falls back to force-copy when the diff cannot be computed. The
no-new-commits case is now reported as `(no new commits)` and skips the
copy/sgl-kernel-build path entirely.

Manual fallback (for instances whose `/root/sglang-minicpm` is missing
`.git/`):

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/sglang-minicpm && \
   git clone --depth 1 --branch mixed_minicpm_cudagraph \
     https://github.com/oldzhu/sglang-minicpm.git /root/sglang-minicpm'
```

`copilot-instructions.md` was updated with the same recipe.

## Cross-references

- Continuation 003: [CHANGE_0151_phase_b_four_over_six_003.en.md](CHANGE_0151_phase_b_four_over_six_003.en.md)
- `_init_rope` patcher: [CHANGE_0152_init_rope_transformers5_compat.en.md](CHANGE_0152_init_rope_transformers5_compat.en.md)
- Test row: TEST_RESULTS_TRACKING.md → NVFP4-FOS-5
- Chat log: chat/CHAT_phase-b-fos-iter5_20260508_1430.en.md
