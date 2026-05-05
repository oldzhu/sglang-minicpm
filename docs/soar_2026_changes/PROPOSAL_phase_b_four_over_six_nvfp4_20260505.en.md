# PROPOSAL — Phase B: FourOverSix adaptive NVFP4 (per-block M=6/M=4 scale)

Date: 2026-05-05
Branch: `mixed_minicpm_cudagraph`
Status: **proposal — awaiting approval before any code change**
Depends on: `CHANGE_0150_phase_a_nvfp4_baseline.{en,zh}.md`

## 0. TL;DR

Phase A confirmed:
- FP4 cutlass kernel on SM120 gives ~4× over BF16 on prefill/chunk shapes.
- Uniform NVFP4 collapses model quality (mcq generates gibberish like "unedoc.com/abc/u.com…").

Phase B applies the **FourOverSix** technique from the public champion blog:
for each NVFP4 block of 16 weights, choose between two scaling modes (M=6 or M=4)
based on per-block reconstruction error. This restores accuracy while keeping the
NVFP4 storage layout — so **no kernel work is required**: the same `flashinfer.mm_fp4`
cutlass path Phase A validated still applies.

Goal: pass a smoke test (coherent answer to a short prompt), then re-run accuracy +
S1/S8/Smax on fcloud, then decide whether to submit or layer FP8 KV / fused norms on top.

## 1. Background

### NVFP4 storage and what "M" actually is

NVFP4 stores **two** quantized things per 16-weight block:

1. **Per-element 4-bit codes** — 16 of them, drawn from the fixed lattice
   `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`. They are packed two per byte, so a `(N, K)`
   weight tensor becomes a `uint8` tensor of shape `(N, K/2)`.
2. **Per-block scale `s`** — exactly one FP8 E4M3 number per 16-element block,
   shape `(N, K/16)`, dtype `float8_e4m3fn`. Dequant is just `w ≈ code · s`.

The lattice is fixed by the NVFP4 standard (and is what `flashinfer.mm_fp4` /
`cutlass_scaled_fp4_mm` decodes in hardware). **`M` is not stored anywhere** — it is
only the **calibration-time rule** for choosing `s`:

```
M = 6 rule:   s = max(|w_block|) / 6     # block-max maps to lattice extreme ±6
M = 4 rule:   s = max(|w_block|) / 4     # block-max maps to lattice point ±4;
                                         # anything > 4·s gets clipped
```

Same lattice, same storage layout — only `s` differs. The kernel doesn't care which
rule was used; it just sees a per-block FP8 scale.

### When does each rule win?

The block-MSE function decides; the intuition cuts both ways:

- **M=6 wins** when the block has one outlier and most values are tiny. Smaller `s`
  preserves near-zero resolution; the lone outlier still rounds to lattice point ±6.
- **M=4 wins** when the block is "dense at the top" — many values clustered near the
  block-max. M=6 stretches the lattice all the way to ±6 even though codes ±5, ±6 are
  barely used, so most active values round between widely-spaced points {±2, ±3, ±4}.
  M=4 packs the lattice {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4} entirely inside the active
  range and rounds those mid values to closer points. The cost is values strictly
  between `4·s_M4` and `max` get clipped to ±4·s_M4, but if the block-max itself was
  the only such value, the clipping error is small.

The champion blog reports ~40–43 % of blocks pick M=4 in their LLaMA / Qwen runs;
how the split looks on MiniCPM-SALA-90 is empirical.

**FourOverSix** = at calibration time, for each block independently pick
`M ∈ {4, 6}` that minimizes the per-block reconstruction MSE
`||round_to_lattice(w/s) · s − w||²`.

The result is still standard NVFP4 storage (uint8 packed codes + FP8 E4M3 scale per
block), so `flashinfer.fp4_quantize` / `mm_fp4` / `cutlass_scaled_fp4_mm` all still
work without modification. Only the **scale chosen at calibration time** changes.

### Why this is allowed by the rules

- Apache-2.0, reproducible, explainable: yes — fully on-site quantization, deterministic.
- No private data: only the public 90-sample stratified calibration set.
- Quantize + eval ≤ 5 h: FourOverSix is per-block O(16) extra compute; calibration time
  unchanged.
- Submission ≤ 2 GB: same NVFP4 storage layout as Phase A.

## 2. Objective and expected gain

### What success looks like

| Metric | Baseline GPTQ v22 | Phase A (uniform NVFP4) | Phase B target |
|---|---|---|---|
| Smoke test (single short prompt) | coherent | gibberish | **coherent** |
| ori_accuracy (90 mcq+qa+niah+cwe+fwe) | 79.29 % | not measurable (timeout) | **≥ 78 %** |
| normalized accuracy | 99.11 % (C=1.0) | 0 (eliminated) | **≥ 99 %** (C=1.0) target, ≥ 97 % (C≥0.92) hard floor |
| S1 / S8 / Smax (local set) | 121.71 / 44.09 / 35.86 s | timed out | **≥ 0.5×** of baseline (i.e. ≤ 240 s S1) |
| Official score (estimate) | 30.04 (#22) | 0 | If FP4 GEMM 4× shows up on long-context official set: notable jump |

The expected **speed** gain is bounded above by what we measured in Phase A's microbench:
~4× on prefill/chunk shapes, ~1× (or worse) on decode. Whether this materializes in
the harness depends on prefill/decode ratio, attention dominance, and concurrency.
Phase B is primarily about **unblocking** the speed measurement, not chasing more speed.

### What could go wrong (so we're realistic)

- Even with FourOverSix, MiniCPM-SALA-90 is a heavily fine-tuned model on long-context
  tasks; its weight distribution may not match the LLaMA / Qwen models the champion
  validated on. Accuracy drop > 2 pp is possible.
- Outlier layers (e.g. MoE-style gating or rotary-applied projections) may need to stay
  at higher precision regardless. Our existing exclude list already keeps `o_gate`,
  `z_proj`, `lm_head`, `norm`, `embed_tokens` at BF16; we may need to add more.
- Decode-bound benchmarks could *regress* relative to GPTQ baseline because Phase A's
  microbench showed FP4 decode-1 = 0.49–0.92× BF16. If Smax is decode-bound on the
  official set, Phase B could lose points there even if accuracy is intact.

## 3. Rule-compliance check

| Rule | Compliance |
|---|---|
| On-site quantization | ✅ runs in `prepare_model.sh --input … --output …`, same flow as Phase A |
| ≤ 2 GB submission | ✅ same NVFP4 layout as Phase A (~2.7–3 GB safetensors → still fits with `lm_head`, `embed_tokens` BF16) — needs measurement at end of Phase B |
| ≤ 5 h quantize + eval | ✅ FourOverSix adds < 1 % calibration overhead |
| No private/eval data leakage | ✅ same 90-sample stratified calibration set we used for GPTQ |
| Apache-2.0, reproducible | ✅ deterministic given calibration set |
| Accuracy > 97 % (C ≠ 0) | ⚠ must measure; this is the gating criterion of Phase B itself |
| Don't bypass `--flush-cache` / fixed concurrency | ✅ no runtime change |
| Eval-script integrity | ✅ no change to `eval_model_001.py` |

## 4. Implementation plan (before change — no edits made yet)

### 4.1 Two candidate code paths

**Both produce the identical checkpoint** (NVFP4 with FourOverSix-chosen scales) —
they only differ in *where in the call stack* the per-block M choice is made. Since
FourOverSix needs only the weight tensor (no activations), there is no calibration-data
benefit to running it inside modelopt's loop.

**(B1) Override modelopt's NVFP4 quantizer (deeper)**
- Subclass `modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor` (or whatever
  the 0.43 path is — to be verified during impl) and replace its scale-selection logic.
- Pro: the quantizer naturally re-uses modelopt's calibration data.
- Con: tied to modelopt internals; may break on minor version bump; harder to verify
  by inspection.

**(B2) Post-hoc scale rewrite (shallower) — recommended**
- Let modelopt do its standard NVFP4 calibration end-to-end (same as Phase A).
- Right before `export_hf_checkpoint`, walk every weight quantizer module, read the
  original BF16 weight, recompute the per-block scale by FourOverSix, **rewrite both
  the FP8 E4M3 scale tensor and the uint8 4-bit code tensor in-place**, then export.
  (We rewrite the codes too because changing `s` changes which lattice point each
  weight rounds to. Storage layout stays standard NVFP4.)
- Pro: completely isolated to our code; simple to test in a notebook with a single
  weight tensor; trivially rollback-able.
- Con: We do roughly 2× the per-block work at calibration end (compute MSE for both
  M values), but this is one pass over weights, milliseconds total.

**Decision: implement B2 first.** If it works, we don't need B1 at all. If MSE-only
selection turns out to be insufficient (e.g. activation-aware selection is required),
we revisit B1 then.

### 4.2 Algorithm (B2) — pseudocode

```python
# For each Linear layer that was NVFP4-quantized (i.e. weight_quantizer.is_enabled)
W = layer.weight  # original BF16 weight, shape (N, K)
B = 16            # NVFP4 block size

# Reshape to (N, K/B, B) so each row is split into K/B blocks of 16.
Wb = W.view(N, K // B, B)

block_max = Wb.abs().amax(dim=-1)  # (N, K/B), per-block max abs

# Two candidate scales
s6 = block_max / 6.0
s4 = block_max / 4.0

# Quantization error for each candidate (FP8 E4M3 rounding of s itself ignored here;
# we'll redo the FP8 E4M3 cast at the end so the final stored scale is legal).
def err(scale):
    # quantize each block with this scale, dequantize, return MSE
    q = round_to_nvfp4_lattice(Wb / scale.unsqueeze(-1))
    deq = q * scale.unsqueeze(-1)
    return ((deq - Wb) ** 2).sum(dim=-1)  # (N, K/B)

e6 = err(s6)
e4 = err(s4)

pick_m4 = e4 < e6  # (N, K/B) boolean

new_scale = torch.where(pick_m4, s4, s6)

# Cast scale back to FP8 E4M3 (NVFP4 storage requires this dtype for the per-block scale)
new_scale_fp8 = new_scale.to(torch.float8_e4m3fn)

# Re-quantize weights with the new scale and write packed uint8 codes
new_codes_uint8 = pack_nvfp4(round_to_nvfp4_lattice(Wb / new_scale_fp8.unsqueeze(-1).to(torch.float32)))

# Overwrite layer's stored quantizer state
layer._weight_quantizer.amax = ...   # whatever modelopt expects to derive scale
layer.weight = new_codes_uint8       # whatever attribute name modelopt uses post-quantize
# (exact attribute names verified during impl)
```

The exact modelopt attribute names depend on 0.43's internal API and need to be probed
on fcloud (1-screen jupyter session) before writing the patch. This probe is part of
implementation step 1 below.

### 4.3 Files to change

1. `benchmark/soar/demo_sala/preprocess_model.py`
   - Add helper `_apply_four_over_six(model)` that walks all NVFP4'd Linear layers and
     rewrites scales as described.
   - Call it inside `run_nvfp4_quantization` between `mtq.quantize(...)` and
     `export_hf_checkpoint(...)`.
   - Add an env-var gate `SOAR_NVFP4_FOUR_OVER_SIX=1` (default ON for `nvfp4_fos`
     profile, OFF for `nvfp4`) so we keep Phase A's uniform NVFP4 reproducible.

2. `benchmark/soar/demo_sala/prepare_env.sh`
   - The `SOAR_QUANT_PROFILE=nvfp4_fos` branch is already accepted by validation.
   - Make `nvfp4_fos` set `SOAR_NVFP4_FOUR_OVER_SIX=1` before calling preprocess.
   - Server-side: identical to `nvfp4` (`--quantization modelopt_fp4`).

3. (No sglang source change.)

4. (No sgl-kernel change — kernels reused.)

### 4.4 Implementation steps (after approval)

| Step | What | Where | Validation |
|---|---|---|---|
| 1 | Probe modelopt 0.43 NVFP4 internals | fcloud jupyter, ad-hoc | Print attribute names of one quantized Linear; document in impl PR |
| 2 | Write `_apply_four_over_six` (B2) | local edit `preprocess_model.py` | Standalone unit test on a synthetic (256, 4096) tensor: confirm MSE drops, scale stays FP8 E4M3, decode round-trip OK |
| 3 | Wire `nvfp4_fos` profile | `prepare_env.sh` + arg dispatch | `bash -n prepare_env.sh`; print SGLANG_SERVER_ARGS |
| 4 | Sync to fcloud, run quantize | `python3 scripts/fcloud/fcloud_workflow.py sync` + manual `prepare_model.sh` | Quantize completes, output dir size sane |
| 5 | Smoke test: single short prompt | curl one mcq | Coherent answer, not "unedoc.com" gibberish |
| 6 | Full accuracy eval | `fcloud_workflow.py accuracy` | Read `predictions.jsonl`; compare ori_accuracy to baseline |
| 7 | Speed S1/S8/Smax | `fcloud_workflow.py speed --variant all` | Record in `TEST_RESULTS_TRACKING.md` |
| 8 | Pause instance | `fcloud_workflow.py pause-instance` | (mandatory cost-saving rule) |

### 4.5 Test and benchmark commands

```bash
# Local: dry-run sanity
cd /home/oldzhu/sglang/benchmark/soar/demo_sala
python3 -c "
import preprocess_model as p
import torch
# unit test on synthetic data
W = torch.randn(256, 4096, dtype=torch.bfloat16) * 0.05
W[0, 0] = 5.0  # outlier in one block
# call helper, assert output shapes and that some blocks pick M=4
"

# Fcloud (after sync):
ssh-equivalent → exec on fcloud:
cd /root/submission_sim
SOAR_QUANT_PROFILE=nvfp4_fos bash prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/MiniCPM-SALA-NVFP4-FOS

# Validate quantize ran (check modelopt log line + output dir)
ls -la /root/models/MiniCPM-SALA-NVFP4-FOS

# Restart server with new model
SOAR_QUANT_PROFILE=nvfp4_fos source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --host 0.0.0.0 --port 30000 \
  "${SGLANG_SERVER_ARGS[@]}"

# Smoke test
curl -s http://localhost:30000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'

# Full accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy

# Speed
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### 4.6 Result summary table (to be filled after run)

| Metric | GPTQ v22 baseline | Phase A (uniform NVFP4) | Phase B (FourOverSix) |
|---|---|---|---|
| ori_accuracy | 79.29 % | n/a (timeout) | TBD |
| normalized | 99.11 % | 0 | TBD |
| S1 (s) | 121.71 | timed out | TBD |
| S8 (s) | 44.09 | timed out | TBD |
| Smax (s) | 35.86 | timed out | TBD |
| Output dir size | 4.4 GB (GPTQ + lm_head BF16) | 6.2 GB | TBD (expected ~6.2 GB) |
| Notes | Submission baseline | Phase A plumbing only | Decision point: submit or revert |

### 4.7 Rollback

- All Phase B changes are gated by `SOAR_NVFP4_FOUR_OVER_SIX=1` / `nvfp4_fos` profile.
- To roll back: set `SOAR_QUANT_PROFILE=gptq` (the v22 production baseline) — single
  env var, no code revert needed.
- If we want to revert the source change too: `git revert <phase-B-commit>`; the
  uniform `nvfp4` profile (Phase A) and `gptq` profile (v22) both keep working.

## 5. Risks (consolidated)

| Risk | Mitigation |
|---|---|
| MSE-only block selection is too crude → accuracy still drops | Have B1 (modelopt-internal override) as fallback; can also weight MSE by activation magnitude using calibration-set Hessian |
| Outlier layers need higher precision | Already excluding 5 patterns; can extend exclude list per layer-name regex |
| Modelopt 0.43 internal attributes change in next minor version | Pin `nvidia-modelopt==0.43.0` (already pinned in `prepare_env.sh`) and assert attribute exists |
| Decode regression at Smax | Document; if real, layer Phase B with future "FP4 ffn + GPTQ attn" mixed-precision proposal |
| Output dir > 2 GB for submission | `lm_head` + `embed_tokens` BF16 are the bulk; can share with `tie_word_embeddings` if model supports it (verify) |

## 6. Next-step suggestions (after Phase B lands)

1. If accuracy is good and speed neutral / slightly positive:
   - Re-enable FP8 KV cache on top (`--kv-cache-dtype fp8_e5m2`) — orthogonal to Phase B.
   - Re-enable `--enable-fused-qk-norm-rope` and `--enable-mixed-chunk` (already in
     `prepare_env.sh` SGLANG_SERVER_ARGS, just confirm they don't fight `modelopt_fp4`).
2. If accuracy is good but Smax regresses:
   - Mixed precision: keep attention in GPTQ-W4A8, FFN in NVFP4. Needs runtime
     dispatch logic; new proposal.
3. If accuracy still collapses despite FourOverSix:
   - Activation-aware FourOverSix (weight MSE × activation magnitude).
   - Sensitivity analysis: per-layer accuracy delta vs BF16 reference; auto-exclude
     top-K most sensitive layers.

## 7. Approval ask

Please approve **B2 (post-hoc scale rewrite)** as the implementation route. After
approval, I will:
1. Probe modelopt 0.43 internals on fcloud (one short jupyter exec) to confirm
   attribute names.
2. Implement `_apply_four_over_six` in `preprocess_model.py` and wire `nvfp4_fos`.
3. Push to `minicpm-src/mixed_minicpm_cudagraph` and run the validation sequence.
4. Write the matched `CHANGE_0151_phase_b_four_over_six.{en,zh}.md` after results
   land, then pause the instance.
