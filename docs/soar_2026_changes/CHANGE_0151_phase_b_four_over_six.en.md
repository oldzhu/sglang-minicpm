# CHANGE_0151 — Phase B FourOverSix NVFP4 (first end-to-end run)

## Background and motivation
- Continuation of `PROPOSAL_phase_b_four_over_six_nvfp4_20260505.en.md` and `CHANGE_0150_phase_a_nvfp4_baseline.en.md`.
- Goal: NVFP4 weight-only quantization with the FourOverSix (FOS) per-block scale-selection patch (try `M=4` AND `M=6` per 16-elt block, pick the one with smaller MSE), aiming to recover accuracy on the same speed budget as plain NVFP4.
- This iteration covers the **first end-to-end run** on the fcloud RTX PRO instance: quantize → load → smoke → accuracy. Speed (S1/S8/Smax) was deliberately **skipped** because accuracy fell below the elimination threshold (see Results).

## Rule-compliance statement
- Quantization is performed **on-site** by `preprocess_model.py` invoked from `prepare_model.sh`, satisfying the "no pre-quantized weights" rule.
- All weights/scales remain in standard NVFP4 layout (uint8 packed nibbles + `weight_scale` fp8 + `weight_scale_2` + `input_scale`); the FOS patch only changes which scale value is chosen per 16-element block. The on-disk format is loadable by sglang's stock `modelopt_fp4` loader (`Detected nvfp4 checkpoint` in server log).
- No new external dependencies; uses the already-pinned `nvidia-modelopt 0.43.0`.

## Detailed implementation plan (before change)
- Reuse Phase A's NVFP4 calibration path in `run_nvfp4_quantization` (modelopt `mtq.quantize` with `NVFP4_DEFAULT_CFG`, calibration over `perf_public_set.jsonl`).
- Activate `_install_four_over_six_patch` BEFORE the export pass so that every call to `NVFP4QTensor.get_weights_scaling_factor` evaluates both M=4 and M=6 candidates per block.
- Replace modelopt's `export_hf_checkpoint(...)` with a **manual streaming export**, because:
  1. modelopt 0.43.0's `export_hf_checkpoint` keeps a hidden fp16 reference per Linear (~148 MiB / Linear leak observed in Plan-A diagnostic), causing GPU OOM around layer ~307/560 on the 84 GiB SM120 GPU.
  2. The fcloud container has a **64 GiB CPU cgroup cap** (`/sys/fs/cgroup/memory.max=68719476736`), so a "move-to-CPU first" workaround also gets SIGKILLed.
- The manual export quantizes one Linear at a time, copies the packed weight + 3 scales + bias to CPU, then frees the Linear's GPU weight before moving to the next module. Periodic `gc.collect()`+`torch.cuda.empty_cache()` every 64 Linears.

## Actual code changes (after change)

Files (all on branch `mixed_minicpm_cudagraph`):
- [benchmark/soar/demo_sala/preprocess_model.py](benchmark/soar/demo_sala/preprocess_model.py)
  - New manual streaming export inside `run_nvfp4_quantization` replacing `export_hf_checkpoint(...)`.
  - Imports `requantize_resmooth_fused_llm_layers`, `is_quantlinear`, `QUANTIZATION_NVFP4`, `NVFP4QTensor`, `to_quantized_weight`, `get_quant_config`, `convert_hf_quant_config_format`, `_patch_revert_weight_conversion`/`_unpatch_revert_weight_conversion` from modelopt internals.
  - Per-Linear loop:
    - skip non-quantized formats (`QUANTIZATION_NONE`)
    - skip non-Linear modules via `is_quantlinear(_sub)` (commit `f14c3f3e8` — fixes `NotImplementedError` on the wrapping `MiniCPMSALAForCausalLM` whose descendants register as NVFP4)
    - compute `weight_scale_2` from quantizer state, then `weight_scale` (FOS hook fires here), then packed `weight`
    - move all 4 tensors + bias to CPU, then `_sub._parameters["weight"] = None` to free GPU memory
    - drop `_amax`/`_scale` quantizer buffers
  - Walk `named_parameters()` then `named_buffers()` for the rest, **filtering out non-persistent buffers** via each module's `_non_persistent_buffers_set` (commit `829128503` — without this, rotary `cos_cached`+`sin_cached` add ~16 GiB of fp32 to the saved checkpoint).
  - Save with `model.save_pretrained(state_dict=cpu_state_dict, save_modelopt_state=False)` after `_patch_revert_weight_conversion()`; merge `quantization_config` into `config.json`.
- [benchmark/soar/demo_sala/prepare_env.sh](benchmark/soar/demo_sala/prepare_env.sh)
  - `nvfp4_fos` profile branch sets `SOAR_NVFP4_FOUR_OVER_SIX=1` (existing).
  - `--quantization modelopt_fp4` server flag for both `nvfp4` and `nvfp4_fos` (existing).

Commits this iteration:
- `a2cbedd65` preprocess(nvfp4): replace export_hf_checkpoint with manual streaming export
- `f14c3f3e8` preprocess(nvfp4): skip non-Linear modules in manual streaming loop
- `829128503` preprocess(nvfp4): skip non-persistent buffers in manual export

## Validation commands

Quantize:
```bash
cd /root/submission_sim && \
  export SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 SOAR_QUANT_FORCE=1 && \
  source prepare_env.sh && \
  python3 -u preprocess_model.py \
    --input  /root/models/openbmb/MiniCPM-SALA \
    --output /root/models/MiniCPM-SALA-NVFP4-FOS \
    --mode   nvfp4
```

Smoke:
```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0.0}'
```

Accuracy:
```bash
cd /root/data && python3 eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --data_path /root/data/perf_public_set.jsonl --concurrency 32
```

## Result summary

| Metric | NVFP4-FOS (this iter.) | GPTQ baseline (test 12, ref) |
|--------|------------------------|------------------------------|
| Checkpoint size | 6.5 GiB (2 shards) | 8.0 GiB (sparse_qkv_w8) |
| Quantize peak GPU | 34.20 GiB | n/a |
| FOS pct picked M=4 | 43.14% (224 / 521.1M blocks) | n/a |
| Server load mem usage | 7.31 GiB | ~12 GiB |
| Smoke (`2+2`) | coherent `<think>` output | OK |
| Local accuracy (avg) | **75.98%** | 79.29% |
| → cwe | 74.33% | n/a |
| → fwe | 92.22% | n/a |
| → mcq | 63.33% | (high in baseline) |
| → niah | 93.33% | n/a |
| → qa | 56.67% | (high in baseline) |
| → len_0_4k | 63.33% | n/a |
| → len_4k_32k | 70.25% | n/a |
| → len_32k_128k | 83.58% | n/a |

**Verdict: FAIL accuracy gate.** 75.98% < 77% → normalized accuracy ≤ 97% → C = 0 (eliminated).
The accuracy gap is concentrated in `mcq` (63.33%) and `qa` (56.67%); long-context retrieval (`niah` 93.33%) and counting (`fwe` 92.22%) survive.

Speed (S1/S8/Smax) **not run** this iteration — irrelevant while C=0. Will be run after accuracy is recovered.

## Rollback

Revert the 3 commits and the FOS-export branch is gone:
```bash
git revert --no-edit 829128503 f14c3f3e8 a2cbedd65
git push minicpm-src mixed_minicpm_cudagraph
```
The `nvfp4` profile (no FOS) and the GPTQ baseline are unaffected and remain the live submission config.

## Diagnosis: where the 3 pp accuracy loss likely comes from

1. **Per-block FOS objective is not aligned with downstream attention/MLP loss.**
   The current patch picks `M ∈ {4, 6}` per 16-elt block by minimizing local MSE of weight reconstruction. That is **not** the same as minimizing output error after the full GEMM + activation + softmax. For long-context QA/MCQ where the model needs to attend to a single token in 100k context, even tiny per-block scale errors compound multiplicatively along Q/K/V/O projections and across 32 layers.
2. **No layer/projection-aware skip-list.** GPTQ baseline keeps Q/K/V at w8 (`sparse_qkv_w8` preset) precisely because attention projections are accuracy-critical. FOS-NVFP4 uniformly applies fp4 to all Linears. The Q-proj and K-proj are likely the dominant accuracy sinks.
3. **`mcq`/`qa` rely on tightly-bound numeric reasoning** and short answer spans — exactly the cases most sensitive to logit precision near the decision boundary. `niah` (find a needle) and `fwe` (count occurrences) are robust to scale noise.
4. **No FOS calibration objective override.** `_install_four_over_six_patch` runs only at export-time scale picking; calibration `_amax` was collected with the unpatched scale formula. There may be a calibration-export mismatch.

## Next-step suggestions

In recommended priority order:

A. **Layer-aware FOS skip-list (cheapest, keep speed).** Skip FOS for Q/K/V projections (or for the first/last 4 layers); use plain NVFP4 there. Expected gain: recover ~1.5-2 pp of mcq/qa.

B. **Mixed-precision fallback for QKV.** Keep weight-fp4 for MLPs (`gate_proj`/`up_proj`/`down_proj`) and **w8** for attention QKV. Requires a different on-disk layout per Linear, which is non-trivial in the modelopt loader path. Higher reward, higher effort.

C. **Run a Phase A baseline (no FOS) to isolate FOS impact.** If plain NVFP4 also drops to ~76%, the issue is "fp4 weights are too coarse for this model" rather than "FOS picks wrong M". Cheap to run (same code path with `SOAR_NVFP4_FOUR_OVER_SIX=0`).

D. **Re-calibrate with the FOS-aware scale function active during the calibration forward passes**, not only at export. May improve scale-picking statistics.

E. **Park Phase B and re-prioritize.** The official scoring is `Performance × Correctness`; with C=0 there is no score regardless of speed. If A–D do not lift accuracy ≥ 78% (with safety margin) within a small number of iterations, return to GPTQ + FP8 baseline and pursue speed optimizations from `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`.

## Cross-references

- `PROPOSAL_phase_b_four_over_six_nvfp4_20260505.en.md` — design for FOS
- `CHANGE_0150_phase_a_nvfp4_baseline.en.md` — pure NVFP4 prior iteration
- `PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md` — original idea pairing
- `SM120_RTX_PRO_HARDWARE.md` — GPU constraints (84 GiB, NVFP4 = 593 TFLOPS)
- TEST_RESULTS_TRACKING.md — accuracy table updated this iteration
