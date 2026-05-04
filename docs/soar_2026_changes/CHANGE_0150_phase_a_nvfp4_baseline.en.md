# CHANGE 0150 — Phase A: NVFP4 Baseline (uniform NVFP4) — first end-to-end attempt

Date: 2026-05-04
Branch: `mixed_minicpm_cudagraph`
Commits: `aa1304292` (initial Phase A wiring), `0a56da668` (modelopt 0.43 / torch 2.9 compat fix)

## Background

Phase A of the champion-blog reproduction roadmap (see `PROPOSAL_phase_a_nvfp4_baseline_design_20260504.en.md`):
quantize all linear weights of MiniCPM-SALA to **uniform NVFP4** (block_size=16, FP8 E4M3 per-block scale)
via `nvidia-modelopt`, export with sglang's existing `modelopt_fp4` loader, and confirm the pipeline
runs end-to-end before adding FourOverSix per-block adaptation in Phase B.

## Code changes

1. `benchmark/soar/demo_sala/prepare_env.sh`
   - New `SOAR_QUANT_PROFILE` switch (`gptq` | `nvfp4` | `nvfp4_fos`).
   - Conditional install block for `nvidia-modelopt` (gated by profile).
   - Server-arg branch swaps `--quantization gptq_marlin` ↔ `--quantization modelopt_fp4`.
2. `benchmark/soar/demo_sala/preprocess_model.py`
   - New `run_nvfp4_quantization(...)` that calls `mtq.quantize` with `NVFP4_DEFAULT_CFG`,
     excludes `lm_head`/`o_gate`/`z_proj`/`norm`/`embed_tokens`, and exports via
     `modelopt.torch.export.export_hf_checkpoint`.
   - New `mode='nvfp4'` argparse option, mode resolver prefers `SOAR_QUANT_PROFILE`.

## Dependency findings (modelopt vs torch 2.9)

- modelopt **0.31.0** (initially planned) imports `torch.onnx._type_utils` which was **removed in torch 2.8+**.
  → ImportError on our pinned `torch==2.9.1+cu128`.
- modelopt **0.43.0** is the first release compatible with torch 2.9. The matching
  `nvidia-modelopt-core` (the Cython kernels) tops out at **0.33.1** at the time of writing — that
  combo is the verified-working set on fcloud (2026-05-04).
- Transitive packages exercised by `import modelopt.torch.quantization`:
  `cppimport`, `pulp`, `onnx`, `pydantic`, `rich`, `torchprofile` (all installed `--no-deps` so our
  pinned `torch`/`transformers`/`gptqmodel`/`flash-attn`/`huggingface-hub` stay frozen).
- **Important:** running `uv pip install nvidia-modelopt[torch]` *without* `--no-deps` upgrades torch
  to 2.10 silently. We force `torch==2.9.1` after the modelopt install (paranoia path, currently
  not in `prepare_env.sh` but documented).

`prepare_env.sh` was updated in commit `0a56da668` to install the verified
combination with `--no-deps` and the verified extra transitive deps.

## End-to-end validation outcome

| Step | Result |
| --- | --- |
| `prepare_env.sh` with `SOAR_QUANT_PROFILE=nvfp4` | ✅ completes; emits `--quantization modelopt_fp4` |
| `python3 -c 'import modelopt.torch.quantization as mtq'` | ✅ works (modelopt 0.43.0) |
| sglang server load of pre-existing `/root/models/MiniCPM-SALA-NVFP4` | ✅ 6.65 GB GPU mem; cuda graph capture finishes |
| Single short prompt ("The capital of France is") | ⚠️ outputs "Paris" then degenerates into garbage repeating "unedoc.com/abc/u.com/..." |
| 150-sample accuracy eval (concurrency=32) | ⏰ client-side 3600 s timeout after ~62/150 (~41 %); per-request latency dominated by long-context items (some > 5 min/item) |
| Decode throughput on the short prompt | ~57 tok/s (64 tokens in 1.118 s) — **fine** for short context, the eval-time slowness is from very long prompts |

**Key finding 1 — quality:** uniform NVFP4 (block_size=16, FP8 scale, no adaptive M)
collapses the model on long-form generation. This is consistent with the champion blog's
explicit warning: they had to introduce **FourOverSix per-block adaptive scaling** *before*
the model became usable. Our first iteration is the no-FOS baseline that was always going to
be too lossy.

**Key finding 2 — speed at long context:** even after the model is loaded with
`modelopt_fp4`, the latency on long-context items is dramatically worse than the GPTQ
baseline. Likely cause: sglang's stock `modelopt_fp4` loader does **not** route through
the SM120 FP4 tensor-core kernel; it dequantizes back to BF16 per linear (the
"experimental and subject to change" log line on load supports this). Quantification
of the kernel path is a follow-up.

## Files added/changed

- `benchmark/soar/demo_sala/prepare_env.sh` — modelopt install gate + arg swap
- `benchmark/soar/demo_sala/preprocess_model.py` — `run_nvfp4_quantization`, mode='nvfp4'
- `docs/soar_2026_changes/RESEARCH_week7_champion_review_20260504.{en,zh}.md`
- `docs/soar_2026_changes/PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.{en,zh}.md`
- `docs/soar_2026_changes/PROPOSAL_phase_a_nvfp4_baseline_design_20260504.{en,zh}.md`
- `docs/soar_2026_changes/chat/CHAT_week7-champion-review_20260504_1700.{en,zh}.md`

## Conclusions for the next iteration

1. Uniform NVFP4 is **not viable** as a submission baseline (quality collapse) — exactly
   what the champion blog said.
2. Phase A's value is the **plumbing**: profile switch, modelopt install path, server
   arg swap, sglang `modelopt_fp4` loader integration are now all proven on fcloud.
3. The existing `/root/models/MiniCPM-SALA-NVFP4` checkpoint pre-dates this iteration —
   we still have not run our own `run_nvfp4_quantization` end-to-end (the eval timed out
   before we re-quantized). **TODO** for the next session: run
   `SOAR_QUANT_PROFILE=nvfp4 bash prepare_model.sh --input ... --output ...` to confirm
   our calibration code path works.
4. Two viable next directions:
   - **Phase B (FourOverSix)** — implement adaptive M=6/M=4 per-block scaling inside
     a custom modelopt `forward_loop`, target the ≥99 % accuracy point the champion
     achieved.
   - **Investigate kernel path** — confirm whether sglang's `modelopt_fp4` loader on
     SM120 actually uses FP4 tensor cores; if it falls back to BF16 dequant, the
     speed gain we expect from FP4 is illusory until we add a custom kernel (similar
     to how we added the W4A8 FP8 GEMM in CHANGE_0090).

## Validation commands

```bash
# Re-establish env on fcloud (sets SGLANG_SERVER_ARGS with --quantization modelopt_fp4)
cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 bash prepare_env.sh

# Sanity-check server with existing NVFP4 weights
python3 -m sglang.launch_server \
  --model-path /root/models/MiniCPM-SALA-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  $SGLANG_SERVER_ARGS

# Short-prompt smoke test
curl -s -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":64,"temperature":0.0}}'
```

## Rollback

Set `SOAR_QUANT_PROFILE=gptq` (the default) — every NVFP4 code path is feature-gated.
No state is persisted in the env or model layout.
