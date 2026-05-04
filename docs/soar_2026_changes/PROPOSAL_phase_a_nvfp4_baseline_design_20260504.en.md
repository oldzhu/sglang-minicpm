# Phase A Design — NVFP4 baseline (no FourOverSix, no Medusa)

**Date**: 2026-05-04
**Author**: Agent (awaiting user approval before any code change)
**Parent proposal**: [PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md) (§4 Phase A)
**Predecessor build**: v22 (`SOAR_TORCH_COMPILE_MAX_BS=24`, commit `234f3fed8`)

---

## A0. Goal of Phase A (verbatim)

Stand up the **NVFP4 weight + FP8_e5m2 KV + dense + Marlin/Cutlass FP4 GEMM** path end-to-end on fcloud SM120, behind a single env switch, with **accuracy ≥ 78% local** (margin floor) and **Smax ≤ 35s local**. **No** FourOverSix, **no** Medusa. Just prove the FP4 weight pipeline is healthy.

Ship as **v23** if it passes; else revert to v22 byte-equivalent by unsetting one env var.

## A1. Why NVFP4 (not MXFP4 / not modelopt_fp8)

| Format | Block size | Scale dtype | Spec value set | sglang loader | SM120 GEMM TF | Pick? |
|---|---|---|---|---|---|---|
| MXFP4 | 32 | E8M0 | E2M1 | `mxfp4` | uses cutlass mxfp4 | No — strictly less accurate for dense weight (scale is 8-bit power-of-2 only) |
| **NVFP4** | **16** | **FP8 (E4M3)** | **E2M1, ±{0,0.5,1,1.5,2,3,4,6}** | `modelopt_fp4` | **593 TF (FP4 tensor cores)** | **Yes** |
| GPTQ W4A16 (current) | 128 | FP16 | INT4 | `gptq_marlin` | 148 TF (BF16 Marlin) | No — leaves 4× headroom |
| modelopt_fp8 | per-tensor | FP32 | E4M3 | `modelopt_fp8` | 296 TF | No — ½ the FP4 throughput |

NVFP4 is also the format the Week-7 champion / arXiv:2512.02010 use; FourOverSix (Phase B) is a per-block scale-picking refinement of NVFP4, so the format must match.

## A2. Pipeline choice — **NVIDIA Model Optimizer (modelopt) + sglang `modelopt_fp4` loader**

We will **NOT** extend `gptqmodel` to emit NVFP4 in Phase A. Reasons:

1. `gptqmodel 5.7.0` (the wheel pinned in `prepare_env.sh`) has **no native NVFP4 export**. Adding one is non-trivial and is exactly what Phase B (FourOverSix-inside-GPTQ) will do. We don't want to mix the two.
2. sglang `modelopt_quant.py` already loads NVFP4 checkpoints via `hf_quant_config.json` / `quantization_config.quant_algo == "NVFP4"`. We only need to produce a checkpoint in that format.
3. nvidia-modelopt is Apache 2.0 — **rule-compliant**.
4. modelopt's NVFP4 calibration is MSE-based round-to-nearest within E2M1 lattice, which **matches** the M=6-only baseline of the FourOverSix paper (FoS adds the M=4 alternative in Phase B).

### A2.1 Phase A install line (added to `prepare_env.sh`)

```
uv pip install --no-deps "nvidia-modelopt[hf]==0.31.0" "scikit-learn"
```
(0.31.0 is the latest tag with stable NVFP4 export and a known-good wheel for cu128. Pinned exact version, `--no-deps` to avoid yanking torch/transformers.)

> If install fails on fcloud (network), fallback path (A2.2) below kicks in.

### A2.2 Fallback if modelopt is unavailable

We keep a small in-tree NVFP4 quantizer (`benchmark/soar/demo_sala/nvfp4_quantize.py`, ~120 LOC, Apache 2.0 by us) that does:
```
for each Linear:
    for each row, for each block of 16:
        block_max = max(|w|)
        scale_amax = block_max / 6.0      # 6.0 = max NVFP4 magnitude
        block_scale_fp8 = quant_to_fp8_e4m3(scale_amax)
        w_int4 = round_to_lattice(w / dequant(block_scale_fp8), {0,±.5,±1,±1.5,±2,±3,±4,±6})
        pack two int4 into uint8
emit hf_quant_config.json with quant_method=modelopt, quant_algo=NVFP4, group_size=16
```
This is what we would have to write *anyway* for Phase B (FourOverSix is `min(err_M=6, err_M=4)` over this same routine), so the work is **not wasted** if modelopt path is blocked.

**Decision: try modelopt first; fall back to in-tree if modelopt install fails.** Pick made at preprocess time by `import nvidia_modelopt` try/except.

## A3. Module-include / exclude policy

Same as current GPTQ pipeline — must be enforced because `o_gate` and `z_proj` in lightning-attn layers are tiny gating projections that lose disproportionate accuracy under any 4-bit quant.

| Module | Action |
|---|---|
| `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | NVFP4 |
| `mlp.gate_proj`, `up_proj`, `down_proj` | NVFP4 |
| `self_attn.o_gate`, `z_proj` | **Exclude** (keep BF16) |
| `lm_head` | **Exclude** (keep BF16) |
| All norms, all embeddings | **Exclude** (keep BF16) |

`hf_quant_config.json` `exclude_modules` field carries this list — sglang's `ModelOptFp4Config.from_config` honors it (verified at modelopt_quant.py:1012-1017).

**Note**: this drops the current `sparse_qkv_w8` mixed-precision (W8 on QKV of sparse-attn layers). Phase A is intentionally **uniform NVFP4**. If accuracy regresses below 78%, we'll restore W8 QKV mix in a follow-up (Phase A.1).

## A4. Server-side wiring (no code change in sglang)

`sglang.launch_server` already auto-detects NVFP4 via:
```
config.json
└── quantization_config: {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", ...}
```
or via separate `hf_quant_config.json`. The `modelopt_fp4` loader in `python/sglang/srt/layers/quantization/modelopt_quant.py` (already in-tree, 1700 LOC, untouched) handles it.

We just **drop** `--quantization gptq` (currently set explicitly?) and let auto-detect kick in. If anything is set, we override to `--quantization modelopt_fp4`.

KV cache stays at `fp8_e5m2` (current v22 default). FP4 KV (`SOAR_FP4_KV_CACHE=1`, CHANGE_0131) **stays off** in Phase A — that's a separate axis.

## A5. Files to change (Phase A only)

| File | Type of change | LOC estimate |
|---|---|---|
| `benchmark/soar/demo_sala/prepare_env.sh` | + `SOAR_QUANT_PROFILE` env (default `gptq`); when `nvfp4`, set `SGLANG_SERVER_ARGS` quant flag to `modelopt_fp4` and pin a different `MODEL_PATH` env override; ensure `nvidia-modelopt` is `uv pip install`ed | ~30 |
| `benchmark/soar/demo_sala/preprocess_model.py` | + new `run_nvfp4_quantization(...)` function; dispatcher in `main()` reads `SOAR_QUANT_PROFILE` and routes; copies `_patch_chat_template_for_mcq` and exclude/include logic; emits `hf_quant_config.json` (modelopt path) or writes one ourselves (fallback) | ~150 (new) + ~20 (dispatch) |
| `benchmark/soar/demo_sala/nvfp4_quantize.py` | **NEW**, fallback in-tree quantizer (only used if modelopt import fails) | ~120 |
| `docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.{en,zh}.md` | **NEW**, results doc filled in after fcloud test | ~80 each |

Files **NOT** touched in Phase A:
- `gptqmodel_minicpm_sala.py` (untouched — only used when `SOAR_QUANT_PROFILE=gptq`)
- any file under `python/sglang/srt/` (modelopt_fp4 path is upstream-supported)
- eval script (per repo guardrail)

## A6. Env-gate spec (single switch)

Add to `prepare_env.sh`:

```bash
# Phase A: weight quantization profile selector.
#   gptq      = current v22 baseline (sparse_qkv_w8 GPTQ W4A16)
#   nvfp4     = uniform NVFP4 weights via modelopt (or in-tree fallback)
#   nvfp4_fos = NVFP4 with FourOverSix adaptive M=6/M=4 [Phase B]
export SOAR_QUANT_PROFILE="${SOAR_QUANT_PROFILE:-gptq}"
```

Both `preprocess_model.py` (offline, on-site quantization) and the server-arg block read this single var:
- preprocess: dispatches to `run_gptq_quantization` vs `run_nvfp4_quantization`.
- server: when `nvfp4*`, replaces `--quantization gptq` with `--quantization modelopt_fp4` (or relies on auto-detect; we'll verify both work).

**Default stays `gptq`** so v22 byte-equivalence is preserved.

## A7. Validation plan

### A7.1 Local sanity (no fcloud)
1. Build env in dev container, run `python -c "from nvidia_modelopt.torch.quantization import quantize"` → confirms install.
2. Run `preprocess_model.py --mode nvfp4` against a tiny stub model (10-layer toy) → confirms output `hf_quant_config.json` matches sglang's expected schema.

### A7.2 fcloud quantization run
1. `start-instance` (with user approval).
2. `sync` to fcloud.
3. On fcloud: `cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 bash prepare_model.sh --input <bf16> --output <out>` — wall-clock target ≤ 3h (matches current GPTQ).
4. Inspect output: file size, `quantize_config.json` / `hf_quant_config.json`, count of NVFP4 vs BF16 modules.

### A7.3 fcloud serving
1. `SOAR_QUANT_PROFILE=nvfp4 python3 scripts/fcloud/fcloud_workflow.py restart-server`.
2. `wait-server` health check.
3. Smoke: `curl /v1/models` returns `quantization_method=modelopt_fp4`.
4. `accuracy` run. Pass = ori_accuracy ≥ 78%.
5. `speed --variant all`. Pass = Smax ≤ 35s.
6. `pause-instance`.

### A7.4 Bitwise-style numerical sanity (offline)
For 5 random Linear weights, dequant the NVFP4 stored weight back to BF16 and compute MSE vs original BF16; require **MSE < 5e-3** per weight. This is a quantizer-correctness check independent of the model.

## A8. Pass / fail / decision matrix

| Outcome | Action |
|---|---|
| Calibration finishes ≤ 3h, acc ≥ 78%, Smax ≤ 35s | **Ship as v23**. Document in CHANGE_0150. Move to Phase B. |
| Calibration finishes, acc 76-78% | Restore `sparse_qkv_w8` mix on attn QKV (Phase A.1, ~half day). |
| Calibration finishes, acc < 76% | Stop Phase A. Investigate exclude list (lm_head? embedding? gate norms?). Worst case revert to v22. |
| Calibration finishes, Smax > 35s | NVFP4 GEMM not engaged on SM120 — investigate `--quantization` arg / cutlass kernel dispatch. May need `--enable-flashinfer-cutlass-fp4` or similar. |
| Calibration crashes / OOM | Investigate per-block memory; reduce calib batch to 1; if persistent → **fallback to A2.2** in-tree quantizer. |
| modelopt install fails on fcloud | **fallback to A2.2** automatic. |

## A9. Rollback

```bash
unset SOAR_QUANT_PROFILE   # or SOAR_QUANT_PROFILE=gptq
```
Restart server. v22 byte-equivalent.

If pre-quantized NVFP4 weights are already on disk, leave them — they're not loaded when profile=gptq. (Disk savings come from A's smaller weights, optional cleanup later.)

## A10. Effort & timeline (no estimates of calendar time, just dependency order)

1. Approve this Phase-A design.
2. Implement env-gate + dispatcher + modelopt path + fallback quantizer (single PR, on this branch).
3. Local stub-model sanity (A7.1 + A7.4).
4. User approves fcloud run.
5. fcloud run (A7.2 + A7.3).
6. Document in CHANGE_0150.{en,zh}.md.
7. If pass → v23 tarball + leaderboard submission window.
8. If fail → diagnose; either A.1 sub-iteration or revert.

## A11. Open questions for user

1. **Is `nvidia-modelopt` install allowed in `prepare_env.sh`?** It's Apache 2.0 and on PyPI. If you prefer to avoid the dep entirely, we go straight to the in-tree fallback (A2.2) — same outcome, ~120 extra LOC under our control. **My recommendation**: try modelopt first since it's the reference implementation and saves us debugging the quantizer arithmetic.
2. **Do we keep `--force-dense-minicpm` in Phase A?** Yes by default (matches v22). If you want to also flip to native sparse mode in the same step, that's a second axis and I'd put it in Phase A.2.
3. **Retain MXFP4 KV cache (`SOAR_FP4_KV_CACHE=1`) in Phase A?** Default off (v22 baseline = FP8 e5m2). We can A/B it after Phase A passes.
4. **Submission packaging**: Phase A v23 tarball would include the new `nvfp4_quantize.py` (~5KB) and a slightly larger `prepare_env.sh`. We should NOT pre-quantize and pack NVFP4 weights — quantization must run on-site (rule §3). Confirm.

## A12. Cross-references

- Parent: [PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md)
- Research: [RESEARCH_week7_champion_review_20260504.en.md](RESEARCH_week7_champion_review_20260504.en.md)
- sglang NVFP4 loader: `python/sglang/srt/layers/quantization/modelopt_quant.py:863-1100`
- Existing FP4 KV plumbing (separate axis): [CHANGE_0131_nvfp4_kv_p2_plumbing.en.md](CHANGE_0131_nvfp4_kv_p2_plumbing.en.md)
- Hardware: [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md) (FP4 = 593 TF, 4× BF16)
- nvidia-modelopt: https://github.com/NVIDIA/TensorRT-Model-Optimizer (Apache 2.0)
- FourOverSix paper (Phase B reference): arXiv:2512.02010

---

## Awaiting user response

Reply **approve A-design** to start implementation as described.
Reply with **answers to A11** to adjust scope (e.g. skip modelopt, go straight to in-tree quantizer).
Reply **adjust** with any structural change.

Recommendation: **approve A-design** with default answers (try modelopt first, keep dense, FP8 KV, on-site quantization). This is the lowest-risk path and aligns with the champion's recipe.
