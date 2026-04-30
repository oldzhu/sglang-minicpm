# CHANGE_0130_sm120_marlin_prefill_dispatch

**Date**: 2026-04-21
**Status**: fcloud-tested and failed accuracy; local rollback prepared
**Feature scope**: Baseline-path SM120 Marlin optimization for GPTQ + FP8 KV + dense

## 1) Background & Motivation
- Problem statement:
  Profiling on the current safe baseline shows GEMM is the dominant bottleneck, accounting for about 85.3% of prefill time and 63.5% of decode time. The current Marlin GPTQ path in [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu] still executes through SM80-style `mma.sync.aligned.m16n8k16` logic. Earlier tile-expansion work in CHANGE_0125 compiled successfully but produced no measurable gain because the scorer correctly kept selecting narrow tiles for MiniCPM-SALA's weight shapes.
- Why this should improve speed:
  The current SM120 path still uses a generic scoring heuristic. A bounded first iteration can specialize the Marlin exec-config selection for SM120 prefill-heavy workloads on MiniCPM-SALA without changing model semantics or quantization format. The aim is to improve real kernel choice and runtime behavior for medium/large `M`, where the new official dataset is dominated by long-context prefills.
- Target stage(s):
  Prefill first, with secondary effect on decode only if the dispatch change also improves medium-batch GEMMs.

## 2) SOAR Rule-Compliance Check
- Allowed by rules because:
  This is pure inference-kernel and dispatch optimization on the official MiniCPM-SALA model, fully within the competition's allowed optimization scope.
- Not violating constraints (prefix cache/concurrency/reproducibility):
  No prefix-cache tricks, no concurrency-rule changes, no hidden model replacement, no pre-quantized model submission. The active path remains the documented GPTQ + FP8 KV + dense submission baseline.
- Expected impact on correctness coefficient C:
  Low accuracy risk. This iteration does not change model architecture, prompts, server evaluation logic, or weight format. The target is `C = 1.0` retention.

## 3) Plan Before Code Change
- Files/functions to modify:
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
    - `get_thread_config_list(...)`
    - `score_sm120_candidate(...)`
    - SM120 auto-config selection path near `get_exec_config(...)`
    - diagnostic logging around `log_sm120_exec_config_once(...)`
  - `sgl-kernel/csrc/gemm/marlin/marlin_template.h`
    - inspection only at first; change only if a minimal, isolated staging/load-path improvement is justified
  - `python/sglang/srt/layers/quantization/marlin_utils.py`
    - verify workspace sizing and call path assumptions if kernel-side dispatch changes require Python-side coordination
  - `python/sglang/srt/layers/quantization/gptq.py`
    - verify no extra Python-side shape/path constraints block the new dispatch behavior
  - `benchmark/soar/demo_sala/prepare_env.sh`
    - only if a new server arg or env knob is needed for controlled A/B validation
- Minimal diff strategy:
  Start with dispatch/scoring specialization only. Do not attempt a full SM120-native MMA rewrite in this iteration. Keep the baseline quantization path and server args unchanged unless a gated debug flag is needed.
- Rollback plan:
  Keep all changes isolated to Marlin selection/debug logic so a revert can be a single small patch. If any regression appears, revert CHANGE_0130 and return to the current best baseline config immediately.

## 4) Actual Code Change
- Commit/patch summary:
  Implemented a bounded SM120 prefill-aware scoring policy in the Marlin auto-config path. The change is intentionally small and isolated to `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`.
- Final modified files:
  - `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
- Key logic differences:
  - Added `sm120_prefill_policy_enabled()` with env knob `SGLANG_MARLIN_SM120_PREFILL_POLICY`.
  - Added `is_sm120_prefill_shape(prob_m)` with current threshold `M >= 64`.
  - Added `get_sm120_wave_cap(prob_m)` so prefill-heavy shapes cap the wave-ratio contribution earlier than the default path.
  - Added a guarded `prefill_tile_bonus` that only applies when the candidate already fills most of the machine and has at least one effective wave, to avoid blindly preferring large tiles for under-filled shapes.
  - Extended `log_sm120_exec_config_once(...)` to print `policy=prefill|default` so server logs can confirm which path is active on fcloud.
  - No model code, server args, quantization format, or baseline serving path changed.

## 5) Validation Commands
### Correctness
```bash
cd /home/oldzhu/sglang
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

### Speed
```bash
cd /home/oldzhu/sglang
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### sgl-kernel rebuild on fcloud
```bash
cd /root/sglang-minicpm/sgl-kernel
export CXX=g++ CC=gcc
export CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=10G
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
cp dist/sgl_kernel-*.whl /root/submission_sim/
cd /root/submission_sim
source prepare_env.sh
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | 99.11% normalized (Test 12 baseline family) | 95.50% normalized | -3.61 pts |
| Accuracy / ori_accuracy | 79.29% (Test 12 baseline family) | 76.40% | -2.89 pts |
| S1 benchmark_duration (s) | 110.54s (Test 25B tuning baseline) | not run | — |
| S8 benchmark_duration (s) | 40.54s (Test 25B tuning baseline) | not run | — |
| S∞ benchmark_duration (s) | 33.59s (Test 25B tuning baseline) | not run | — |

### 6.1) fcloud Validation Result (2026-04-22)
- Remote flow completed: incremental `sgl-kernel` rebuild, wheel copy into `/root/submission_sim`, `prepare_env.sh`, server restart, readiness check, full accuracy evaluation.
- Accuracy result: `ori_accuracy=76.40%`, `overall_accuracy=95.50%`, so this variant falls below the 97% survival threshold and would receive `C=0`.
- Failure signature:
  - `mcq=50.00%`
  - `qa=50.00%`
  - short-task average output length inflated to `11143.1` tokens on `mcq`
- Decision: do not run speed benchmarks for this variant. Revert CHANGE_0130 and treat the heuristic as accuracy-unsafe.

## 7) Risk Assessment
- Accuracy risk:
  Low. Kernel-dispatch tuning on the same GPTQ path should not materially change outputs, but any change in reduction/order of operations must still be validated with the full accuracy test.
- Stability risk:
  Medium. The Marlin path is performance-critical CUDA code, and bad dispatch choices could reduce occupancy or expose latent shape issues.
- Reproducibility risk:
  Low if the iteration is limited to deterministic selection heuristics and small logging changes.

## 7.5) Current Verification Status
- Editor diagnostics: clean for `gptq_marlin.cu` after patching.
- Local host wheel build: intentionally not used as an authority for `sgl-kernel`; per repo instructions, CUDA wheel validation must be done on fcloud.
- fcloud validation completed:
  - incremental wheel rebuild: done
  - server restart with rebuilt wheel: done
  - accuracy check: failed (`76.40% / 95.50% normalized`, `C=0`)
  - speed benchmark: intentionally skipped because the variant is not submission-safe
- Local recovery state:
  - CHANGE_0130 scoring heuristic has been reverted locally
  - next remote action should be a baseline revalidation after the user restarts fcloud

## 8) Rollback Instructions
1. Revert the CHANGE_0130 patch from `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` and any corresponding Python-side coordination changes.
2. Restore the current baseline by keeping `benchmark/soar/demo_sala/prepare_env.sh` on GPTQ + FP8 KV + dense with the existing tuned args.

## 9) Next-Step Suggestions
- Do not pursue this specific prefill-aware scoring heuristic further in the submission path.
- Revalidate the reverted baseline on fcloud to confirm accuracy returns to the known-safe range.
- If SM120 GEMM work resumes, move to a more explicit kernel-path investigation rather than heuristic tile biasing.
