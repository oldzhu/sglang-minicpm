# CHAT — Phase B FourOverSix end-to-end (2026-05-06)

## Round 1 — User: "go"

User asked the agent to resume execution of Phase B FourOverSix on fcloud:
quantize NVFP4-FOS → smoke → accuracy + S1/S8/Smax → docs → pause.

Agent ran:
- Cleaned `/root/models/MiniCPM-SALA-NVFP4-FOS` and re-launched
  `preprocess_model.py --mode nvfp4` with `SOAR_QUANT_PROFILE=nvfp4_fos`.
- Quantize succeeded: `EXIT=0`, peak GPU 34.20 GiB, FOS pct_m4=43.14%,
  but checkpoint was 23 GiB (1163 tensors).

Diagnosed bloat: 16 GiB of fp32 rotary `cos_cached`/`sin_cached` were being
saved because `named_buffers()` returns non-persistent buffers too.

Fix (commit `829128503`): in the manual export buffer walk, build a set of
`(module_path, local_buf_name)` from each module's `_non_persistent_buffers_set`
and skip any buffer whose full name is in that set.

Re-ran quantize: ckpt now **6.5 GiB**, 1067 tensors. EXIT=0.

## Round 2 — Server bring-up

- Restarted server with `--model-path /root/models/MiniCPM-SALA-NVFP4-FOS`
  + env `SOAR_QUANT_PROFILE=nvfp4_fos` + `SOAR_NVFP4_FOUR_OVER_SIX=1`.
- `wait-server` timed out at 5 min because CUDA-graph capture on this config
  takes 366 s (7 batch sizes × torch.compile).
- After capture finished, `/health` returned 200; `Detected nvfp4 checkpoint`
  in server log, `mem_usage=7.31 GB` (expected for fp4 + bf16 norms/embed).
- Smoke test (`What is 2+2?`) returned coherent `<think>` text.

## Round 3 — Accuracy

- First attempt failed because `--api_base http://127.0.0.1:30000/v1`
  produced a double `/v1/v1/models` URL → 404. Fixed by passing
  `--api_base http://127.0.0.1:30000` (matches `API_BASE` in `fcloud_workflow.py`).
- Eval ran 150 samples in 2697.48 s.

**Result: ori_accuracy = 75.98%** (below 77% gate, C = 0).

Per-task: cwe 74.33, fwe 92.22, mcq 63.33, niah 93.33, qa 56.67.
Per-length: 0_4k 63.33, 4k_32k 70.25, 32k_128k 83.58.

## Decision: skip speed run, document, pause

Speed run skipped — at C=0 the final score is 0 regardless of duration.
Paused fcloud instance (third call after 504 / 500 retries; final `HTTP 200 任务已暂停`).

## Outcomes

- 3 commits on `mixed_minicpm_cudagraph` pushed to `minicpm-src`:
  - `a2cbedd65` manual streaming export (replaces leaky modelopt export)
  - `f14c3f3e8` skip non-Linear modules in streaming loop (fixes NotImplementedError on the wrapping causal LM)
  - `829128503` skip non-persistent buffers (fixes 16 GiB rotary-cache bloat)
- Manual export design verified: 6.5 GiB ckpt, peak 34.20 GiB GPU, full server load + KV alloc + cudagraph capture + smoke + accuracy 150/150 with EXIT=0.
- **Accuracy is the blocker.** FOS algorithm as implemented (per-block MSE between M=4 and M=6) does not preserve accuracy on this model.

## Next steps proposed (in CHANGE_0151)

A. Layer-aware FOS skip-list (skip Q/K/V or first/last layers)
B. Mixed precision: MLP fp4 + QKV w8 (much higher effort)
C. Run plain NVFP4 (no FOS) baseline to isolate FOS impact
D. Re-calibrate with FOS-aware scale function active during calibration
E. Park Phase B and return to GPTQ + FP8 baseline speed optimizations

## Cross-references

- [CHANGE_0151_phase_b_four_over_six.en.md](../CHANGE_0151_phase_b_four_over_six.en.md)
- [CHANGE_0151_phase_b_four_over_six.zh.md](../CHANGE_0151_phase_b_four_over_six.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) — new "Phase B" subtable, row NVFP4-FOS-1
- Commits: a2cbedd65, f14c3f3e8, 829128503 on `mixed_minicpm_cudagraph`
