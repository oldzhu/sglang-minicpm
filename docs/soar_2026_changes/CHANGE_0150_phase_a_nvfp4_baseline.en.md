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
baseline. Initial hypothesis (BF16 dequant fallback) was **wrong** — code review of
`python/sglang/srt/layers/quantization/modelopt_quant.py` shows `ModelOptFp4LinearMethod.apply`
actually does:

1. `x_fp4, x_scale = fp4_quantize(x, layer.input_scale_inv)` — BF16→NVFP4 cast on every step
   (flashinfer `fp4_quantize` on SM120, sgl-kernel `scaled_fp4_quant` elsewhere; gated by
   `is_sm120_supported()` at import time).
2. `out = fp4_gemm(x_fp4, w_fp4, x_scale, w_scale, alpha, out_dtype, w_n)` — flashinfer
   `mm_fp4` (cutlass backend) → SM120 FP4 cutlass GEMM. Asserts confirm `weight.dtype == uint8`
   (packed FP4) and `weight_scale.dtype == float8_e4m3fn`. **Weights are never dequantized.**
3. The kernel writes BF16/FP16 output for the next layer.

So the FP4 tensor cores **are** being used. The likely real causes of long-context slowness,
in priority order:

1. **Quality collapse → runaway generation.** Uniform NVFP4 produces gibberish, so mcq/qa items
   hit `max_tokens=65536` instead of stopping at the answer. The ">5 min/item" observations are
   most likely items generating ~65k tokens before timeout, not slow tokens. Same failure mode
   as the "Paris → unedoc.com/abc/u.com..." degeneration on the 6-token smoke test.
2. **Per-token activation re-quantization overhead.** BF16→FP4 cast happens on every linear,
   every step. Cheap on short prompts but at chunk=65536 × prefill_max_requests=4 it could
   swamp the FP4 GEMM gain.
3. **Attention is BF16, not FP4.** Q/K/V projections do FP4 GEMM but the flashinfer attention
   kernel consumes BF16, so we still pay FP4→BF16 round-trips around attention.

Cause #1 likely dominates and can only be separated from #2/#3 after FourOverSix restores
quality. Probes (a) nsys kernel dispatch verification, (b) `fp4_quantize` vs `mm_fp4` time
ratio, (c) short-context mcq probe with `max_tokens=512` are queued as investigation tasks.

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
   - **Investigate kernel path** — code review confirms FP4 tensor cores **are** used
     (no Marlin / no BF16 dequant). Remaining unknowns: actual `cutlass_scaled_fp4_mm`
     residency in nsys timeline, ratio of `fp4_quantize` overhead vs `mm_fp4` time, and
     a short-context-only accuracy probe to separate quality collapse from kernel speed.

## Probe results — kernel dispatch + microbench (2026-05-05)

`scripts/fcloud/probe_nvfp4_kernel.py` run on the fcloud SM120 instance with the
v22 wheel set. Confirms `is_sm120_supported() == True` and that `flashinfer.fp4_quantize` /
`flashinfer.mm_fp4` (cutlass backend) execute cleanly. Microbench of typical projection
shapes (K=4096, ffn N=10880, qkv N=4096):

| label | M | BF16 ms | FP4 e2e ms | mm_fp4 only | quant only | quant % | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-1   × qkv | 1 | 0.019 | 0.038 | 0.043 | 0.007 | 17 % | **0.49×** |
| decode-1   × ffn | 1 | 0.035 | 0.038 | 0.034 | 0.007 | 18 % | **0.92×** |
| prefill-512× qkv | 512 | 0.160 | 0.036 | 1.661¹ | 0.007 | 18 % | 4.40× |
| prefill-512× ffn | 512 | 0.476 | 0.075 | 0.091 | 0.007 | 9 %  | 6.38× |
| chunk-4096 × qkv | 4096 | 1.118 | 0.267 | 0.274 | 0.012 | 5 %  | 4.19× |
| chunk-4096 × ffn | 4096 | 2.706 | 0.683 | 0.651 | 0.018 | 3 %  | 3.96× |
| chunk-65536× qkv | 65536 | 15.45 | 3.95 | 3.89 | 0.55 | 14 % | 3.91× |
| chunk-65536× ffn | 65536 | 41.25 | 10.28 | 10.21 | 0.55 | 5 %  | 4.01× |

¹ first-call JIT/heuristic outlier — end-to-end path warmed differently; ignore.

Findings:
1. **FP4 kernel is healthy and gives the expected ~4× speedup** at prefill/chunk shapes.
   The 4× ratio matches SM120's BF16 (148 TF) → FP4 (593 TF) hardware quotient.
2. **`fp4_quantize` activation cast is not a bottleneck** (≤10 % of total at prefill/chunk
   sizes, ≤14 % even at chunk=65536). This rules out per-token re-quantize as the dominant
   cost on long context.
3. **Decode-1 is slower with FP4 (0.49–0.92×).** Memory-bound regime + 2-kernel launch
   overhead beats the compute saving. NVFP4 alone is not a decode-step win — to gain at
   decode we would need to fuse `fp4_quantize` into the previous op or batch decode steps.
4. **Therefore long-context slowness on Phase A test was NOT the kernel.** With
   ~4× GEMM speedup, prefill should be faster, not slower. The dominant cause must be
   **runaway thinking from quality collapse** — at `max_tokens=65536` a single broken
   item can spend minutes generating gibberish. This re-confirms the priority on Phase B
   (FourOverSix) to restore quality before drawing any further speed conclusions.

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
