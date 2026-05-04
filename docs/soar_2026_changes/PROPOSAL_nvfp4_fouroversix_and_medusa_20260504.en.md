# PROPOSAL — Reproduce Week 7 champion: NVFP4 FourOverSix + Medusa GLA verify

**Date**: 2026-05-04
**Author**: Agent (awaiting user approval)
**Predecessor**: v22 (`SOAR_TORCH_COMPILE_MAX_BS=24` default-on, commit `234f3fed8`)
**Source research**: [RESEARCH_week7_champion_review_20260504.en.md](RESEARCH_week7_champion_review_20260504.en.md)
**Risk**: HIGH (cross-cutting changes — new quant pipeline + new model heads + new attention path)
**Expected gain**: target Smax ~16-22s (~30-50% speedup), official score 30 → 50-65 range to enter top 5/6

## 0. Why this is the right next step

- Local server-arg sweeps have plateaued (#2-A only gained 3% Smax; v18→v22 official ≈ flat).
- Top-2 (88.35 / 86.68) jumped +7-10pt in single submission with this exact recipe.
- Hardware FP4 throughput on SM120 is **593 TFLOPS** = 4× BF16 (148) and 2× FP8 (296). NVFP4 is the only path that exploits this. (See [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md).)
- Medusa is 100% decode-focused and gains in **all three** speed tiers — directly attacks the dominant Smax weight.

## 1. Objective (combined two-axis)

**Axis A — Quantization upgrade**: switch weights from current GPTQ W4A16 (sparse_qkv_w8) to **NVFP4 + FourOverSix adaptive scale**, keeping FP8_e5m2 KV cache and dense mode.

**Axis B — Speculative decoding**: train and ship Medusa-style heads (K=1, depth-1 tree) wired through GLA-state-aware tree verify on MiniCPM-SALA's hybrid attention.

## 2. Rule-compliance check

| Rule | Status |
|------|--------|
| Quantization done **on-site** within 5h budget | NVFP4 GPTQ on 90 calibration samples ≈ 90-180 min (similar to current GPTQ_minicpm_sala calibration). FourOverSix is a per-block decision step (~ms each) — adds a few minutes total. ✅ |
| Submission ≤ 2GB | NVFP4 weights ≈ 0.5× current GPTQ size; Medusa heads ≈ 30-100MB depending on K and width. Well within 2GB. ✅ |
| Speculative heads allowed | Explicitly permitted by official rules ("speculative heads count toward 2GB"). ✅ |
| Reproducible / explainable / Apache 2.0 | All upstream sglang Medusa + Marlin NVFP4 code is Apache 2.0; FourOverSix is open algorithm (paper+code per arXiv:2512.02010). ✅ |
| Eval interface unchanged | We only change weights + add heads; eval script untouched. ✅ |

## 3. Risk table

| Risk | Severity | Mitigation |
|------|---------|------------|
| NVFP4 + Marlin kernel path doesn't yet support our `sparse_qkv_w8` mixed scheme | High | Phase A first emits **uniform** NVFP4 (drop the W8 QKV mix); compare accuracy. If accuracy floor cleared, move on; else revisit hybrid. |
| FourOverSix accuracy gain insufficient → C drops below 0.96 | Med | We re-enable FP16 lm_head + FP8 KV (already in v22 baseline). Worst case revert to GPTQ W4A16. |
| Medusa head training takes too long / requires GPU days | Med | Use lightweight K=1 (single head, depth-1) — minimal params (~20-50M). Train on 8×A100/H100 for 6-12h or rent. Distill from main model on 50-200K-token corpus. |
| Tree verify's GLA state-fork breaks correctness on long context | High | Strict unit test: compare token-by-token output of Medusa-on vs Medusa-off on 100 long-context samples; require bitwise match (Medusa rejects → falls back to main token, must equal main-only path). |
| Submission grows beyond 2GB after head + new wheel | Low | NVFP4 saves ~3.5GB over current GPTQ; head size negligible. |
| Time budget — full pipeline takes weeks | Med | Phase A (NVFP4 alone) is shippable independently; Phase B can land later. We bank Axis A score first. |

## 4. Phased plan (4 phases, each independently shippable)

### Phase A — NVFP4 baseline (no FourOverSix, no Medusa)
**Goal**: confirm NVFP4 + Marlin W4A16 path runs end-to-end on our model, lands accuracy ≥ 78%.

Files to touch (estimate):
- `benchmark/soar/demo_sala/preprocess_model.py` — switch GPTQ format to `nvfp4`.
- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` — add NVFP4 quantization config branch (or use upstream gptqmodel NVFP4 if available).
- `benchmark/soar/demo_sala/prepare_env.sh` — point `MODEL_PATH` to NVFP4 model dir; ensure quant flag set; remove `--quantization gptq` from server args, replace with whatever sglang's NVFP4 marker is.
- Add env switch `SOAR_QUANT_PROFILE={gptq,nvfp4,nvfp4_fos}` so we can A/B locally.

Validation:
1. Calibration runs cleanly on 90 samples within ~3h.
2. Server boots; `/v1/models` reports nvfp4.
3. Local accuracy ≥ 78% (baseline floor).
4. Local speed: expect Smax ≈ unchanged or slightly better (Marlin NVFP4 same kernel class as Marlin W4A16; GDDR7 BW dominates on SM120).

Decision: ship as v23 if accuracy passes; revert to v22 otherwise.

### Phase B — FourOverSix on top of NVFP4
**Goal**: adaptive scale picks recover ~0.5-1pt accuracy.

Files to touch:
- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` — inject per-block scale comparison before GPTQ iteration:
  ```python
  scale_m6 = compute_block_scale(W_block, M=6)
  scale_m4 = fp8(scale_m6.float() * 1.5)
  err_m6 = mse(W_block, dequant(W_block, scale_m6))
  err_m4 = mse(W_block, dequant(W_block, scale_m4))
  scale = scale_m4 if err_m4 < err_m6 else scale_m6
  # then proceed with GPTQ within this scale frame
  ```
- Env gate: `SOAR_QUANT_PROFILE=nvfp4_fos`.
- Log per-layer M=4 ratio (expect 40-43% per champion).

Validation:
1. M=4 ratio in 35-50% range; MLP layers > attn QKV.
2. Accuracy ≥ NVFP4-only by ≥ 0.3pt.
3. Throughput identical to NVFP4 (no kernel change).

Ship as v24 if accuracy gain confirmed.

### Phase C — Medusa K=1 head training (offline, off-fcloud)
**Goal**: train 1 Medusa head on hidden_size→vocab projection, predicts t+1 token, GLA-state-aware (no GLA fork yet — straight-line single token).

Hardware: needs an H100/A100 box for ~12-24h. Off-fcloud since fcloud only has RTX 6000D Blackwell. User-provided rental or owned hardware.

Pipeline:
1. Freeze main model (NVFP4-FoS quantized or BF16 — need to test both orderings).
2. Build dataset: SOAR-distribution proxy (long-context QA from public datasets — RULER, LongBench).
3. Train head with W₁ zero-init for ~5-10K steps, target lm-CE loss as auxiliary signal.
4. Distill checkpoint: ~30-80MB.

Files added:
- `benchmark/soar/demo_sala/medusa_head/` — head module + training script.
- `benchmark/soar/demo_sala/medusa_train.py` — entrypoint.
- Heads checkpoint to be packaged into submission tarball.

Validation: top-1 accept rate ≥ 0.5 on held-out long-context set.

### Phase D — Medusa tree-verify with GLA state fork (sglang server side)
**Goal**: integrate trained heads into sglang's MiniCPM-SALA backend with correct GLA state branching.

Files to touch (largest scope):
- `python/sglang/srt/models/minicpm_sala.py` — load Medusa head; route last hidden state through head; extend forward to accept `tree_input_ids` and `tree_position_ids`.
- `python/sglang/srt/layers/attention/minicpm_backend.py` — for each verify forward:
  - Detect tree input (multiple candidate positions);
  - For each GLA layer: snapshot `h_parent`; for each branch run recurrence independently from `h_parent`; collect per-branch outputs;
  - Standard MHA layers: use upstream tree mask (already supported in sglang Eagle/Medusa path).
- `python/sglang/srt/speculative/medusa_*.py` — wire scheduler; produce candidates; gather acceptance.
- `benchmark/soar/demo_sala/prepare_env.sh` — add `--speculative-algorithm medusa`, `--speculative-draft-model-path /root/.../medusa_head`, etc.

Validation:
1. Bitwise-equivalence test: 100 prompts, Medusa-off vs Medusa-on with `temperature=0` — outputs **must** match exactly token-by-token.
2. accept rate logged ≥ 0.4.
3. local Smax ≤ 25s (vs current 32.54s) → 23% gain target.
4. Accuracy floor maintained.

Ship as v25.

## 5. Files to change (consolidated)

| Phase | File | Type |
|-------|------|------|
| A | `benchmark/soar/demo_sala/preprocess_model.py` | edit |
| A | `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` | edit |
| A | `benchmark/soar/demo_sala/prepare_env.sh` | edit (add `SOAR_QUANT_PROFILE` env gate, swap quant flag) |
| B | `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` | edit (FourOverSix block) |
| C | `benchmark/soar/demo_sala/medusa_head/` | new |
| C | `benchmark/soar/demo_sala/medusa_train.py` | new |
| D | `python/sglang/srt/models/minicpm_sala.py` | edit (head wire-in) |
| D | `python/sglang/srt/layers/attention/minicpm_backend.py` | edit (GLA fork) |
| D | `python/sglang/srt/speculative/medusa_*.py` | edit/new |
| D | `benchmark/soar/demo_sala/prepare_env.sh` | edit (speculative args) |

## 6. Test commands per phase

### Phase A
```
# offline (local box, GPTQ container)
python3 benchmark/soar/demo_sala/preprocess_model.py --quant nvfp4 --input <bf16> --output <nvfp4>
# fcloud
SOAR_QUANT_PROFILE=nvfp4 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### Phase B (same as A, profile=nvfp4_fos)

### Phase C (off-fcloud)
```
python3 benchmark/soar/demo_sala/medusa_train.py \
  --base-model <bf16-or-nvfp4-fos> \
  --train-data ruler+longbench-proxy.jsonl \
  --num-heads 1 --depth 1 \
  --output-dir medusa_head/
```

### Phase D
```
SOAR_QUANT_PROFILE=nvfp4_fos SOAR_MEDUSA_ENABLE=1 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 7. Success / failure matrix per phase

| Phase | Pass | Fail-revert |
|-------|------|-------------|
| A | acc ≥ 78%, Smax ≤ 35s | revert to v22; investigate Marlin NVFP4 path |
| B | acc gain ≥ +0.3pt vs Phase A, M=4 ratio in 35-50% | drop FoS, ship Phase A as v23 |
| C | head accept rate ≥ 0.4 on held-out | retrain with more data / different freeze order |
| D | bitwise match Medusa-off vs Medusa-on at T=0; Smax ≤ 25s; acc ≥ 78% | revert speculative server arg; ship Phase B as v24 |

## 8. Rollback per phase

Each phase ships independently with env-gate. Worst case at each gate: unset `SOAR_QUANT_PROFILE` / `SOAR_MEDUSA_ENABLE`, restart, regress to prior tarball.

## 9. Estimated effort

- Phase A: 3-7 days (mainly NVFP4 path debug + accuracy tuning).
- Phase B: 1-2 days (small algorithmic addition).
- Phase C: 2-5 days incl. data + train + tune.
- Phase D: 5-10 days (sglang Medusa wiring + GLA fork is the hardest piece).

## 10. Next-step suggestions if some phase fails

- A fails (NVFP4 path broken): file an upstream sglang issue; meanwhile try **AWQ-NVFP4** alternative.
- B fails (FoS gain marginal): not blocking; ship A as v23.
- C fails (head not accurate enough): try EAGLE / EAGLE3 instead — they're also documented in sglang and have similar gain on long-context decode.
- D blocked on GLA fork: limit Medusa to MHA-only verify path; gain reduced but still positive on dense layers.

## 11. Cross-references

- Research source: [RESEARCH_week7_champion_review_20260504.en.md](RESEARCH_week7_champion_review_20260504.en.md)
- Champion blog: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- FourOverSix paper: arXiv:2512.02010
- Medusa paper: ICML 2024 (Cai et al.)
- SM120 hardware: [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)
- Optimization catalog: [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- Predecessor: v22 (`234f3fed8`)
- Leaderboard: team-beta #22 score 30.04; top-5 cutoff 50.66 (gap +68%)

---

## Awaiting

Reply **approve A** to start with Phase A (NVFP4 baseline) — the lowest-risk, most-shippable first step. Reply **approve all** to commit to the full A-D plan up front. Reply **adjust** with phase reordering or scope cut.

Recommendation: **approve A first**. NVFP4 alone may already shift score 30 → 40+ if Marlin NVFP4 path is smooth on SM120; that buys us breathing room while Phase D's deeper work proceeds.
