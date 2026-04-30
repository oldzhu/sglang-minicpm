# PROPOSAL: Iteration M2.0 — Per-Layer NVFP4 KV Cache Sensitivity Ablation

**Date**: 2026-04-26
**Status**: PROPOSAL — awaiting user approval before any code change
**Predecessor**: [`PLAN_post_v18_baseline_M2_kv_mixed_20260426_1029.md`](./PLAN_post_v18_baseline_M2_kv_mixed_20260426_1029.md)
**Baseline**: commit `8d1e4d12b` (v18 surgical revert, 2026-04-26)

---

## 1. Background and motivation

### 1.1 Strategic context

The Week 5 SOAR champion (智算一队 semifinal) reported the following with **mixed FP8 + NVFP4 KV cache** on their MiniCPM-SALA submission:

| KV strategy | Their accuracy | Outcome |
|---|---|---|
| Pure FP8 KV | ~80% | Baseline |
| Pure NVFP4 KV | ~75% | Below acceptable threshold |
| **Mixed: first/last layers FP8, middle layers NVFP4** | ~80% | **Near-FP4 bandwidth at near-FP8 accuracy** |

Their methodology: **layer-wise sensitivity ablation** — flip one layer's KV from FP8 to FP4, measure accuracy delta, repeat for all layers, then keep the sensitive layers at FP8 and put the robust layers at FP4.

### 1.2 Why this attacks our weakness

- Our worst official tier is **Smax** (current: 2746-2917s) — long-context decode.
- Official speed dataset: ~68% of inputs are 32K-512K tokens; long-context decode is **memory-bandwidth-bound** on KV reads.
- Halving KV-cache bandwidth (FP8 → FP4) attacks the dominant Smax bottleneck directly.
- Champion has validated the recipe on the **same model family** (MiniCPM-SALA) — applicability is high-confidence.

### 1.3 Important corrections to prior knowledge

| Misconception | Correction |
|---|---|
| "We already tested NVFP4 and it fails" | We tested **W4A4** (weights+activations FP4) in Test 21; KV cache stayed FP8. NVFP4 KV cache is an **untested axis**. |
| "Catalog says NVFP4 NOT VIABLE" | The catalog wording is scoped to weight quantization only. KV cache FP4 is unburned. |
| "Need to write FP4 memory pool from scratch" | SGLang upstream **already ships** `MHATokenToKVPoolFP4` ([memory_pool.py:1085](../../python/sglang/srt/mem_cache/memory_pool.py#L1085)) and `--kv-cache-dtype fp4_e2m1` flag. |
| "Ablation needs 32 runs" | MiniCPM-SALA has only **8 standard-attention layers** (`mixer_type==minicpm4`); the 24 lightning (SimpleGLA) layers use recurrent state with no paged KV cache. Ablation is **8 runs**. |

---

## 2. Rule-compliance check

| Rule | Status |
|------|--------|
| Use v18 as baseline | ✅ Branch HEAD = `8d1e4d12b` (v18 surgical revert) |
| Only push to `minicpm-src` | ✅ All commits target `minicpm-src/mixed_minicpm_cudagraph` |
| Never modify `eval_model_001.py` | ✅ M2.0 does not touch the eval harness |
| Maintain accuracy ≥ 78% (C ≥ 0.92) | ✅ M2.0 is read-only in terms of model behavior — only flips KV dtype per layer; per-layer flipping is reversible via env var |
| Keep submission ≤ 2GB | ✅ M2.0 adds no compiled artifacts |
| Quantization on-site | ✅ KV cache dtype is runtime-only |
| All server args via `prepare_env.sh` | ✅ M2.0 config passes through `SGLANG_SERVER_ARGS` |

---

## 3. Goals and non-goals

### 3.1 Goals
1. Empirically determine, for each of the 8 standard-attention layers, the **accuracy delta** when its KV cache is flipped from FP8 to NVFP4.
2. Identify a **robust subset** of layers whose collective FP4 conversion preserves accuracy ≥ 78% (≥1pt margin over C=0.92 floor).
3. Produce a **go/no-go decision** for committing to M2.1 (real per-layer mixed pool implementation).
4. Estimate **expected bandwidth savings** if the robust subset is converted, to confirm the eventual gain is worth the implementation effort.

### 3.2 Non-goals
- M2.0 will **not** ship a real mixed-pool implementation (that is M2.1+).
- M2.0 will **not** modify the attention backend, memory pool internals, or any kernel.
- M2.0 will **not** change the submission package — its outputs are documentation + per-layer CSV.

---

## 4. Approach selection: **Option B (fake-quant), then Option A (real FP4) if upstream works**

### 4.1 Option A — real FP4 storage via existing upstream pool
- Set `--kv-cache-dtype fp4_e2m1` globally and run accuracy. If catastrophic (matches champion's 75%), the pool works correctly. We then need a **per-layer override hook** to flip one layer at a time — but writing this hook requires modifying [`model_runner_kv_cache_mixin.py:583`](../../python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L583) so two pools (FP8 + FP4) coexist and each attention layer indexes the right one.
- Pros: numerics are exactly what M2.1 will deliver; bandwidth savings observable.
- Cons: requires non-trivial upstream-code modification before we even have ablation data.

### 4.2 Option B — fake-quant in BF16 storage (RECOMMENDED for M2.0)
- Add a thin hook in attention forward: `K = (K / scale).round().clamp(-6, 6) * scale; V = same` when env var enables FP4 simulation for that layer.
- Storage stays BF16, **but the numerical effect is identical to true FP4 round-trip**.
- Pros: tiny code surface (~30 lines, single file); fully reversible via env var; no pool-creation modification needed; accuracy signal identical to Option A.
- Cons: no actual bandwidth reduction, so cannot directly confirm Smax gain — but that confirmation is M2.1's job.

**Decision**: M2.0 uses Option B. The accuracy signal is the only thing M2.0 needs, and Option B gives it at minimum risk. Option A is deferred to M2.1 where the real pool integration happens.

### 4.3 Sanity check before per-layer ablation
Run **one** preliminary test with **all 8 standard layers** in fake-FP4 mode to verify:
- Accuracy lands near champion's pure-FP4 number (~75%).
- The fake-quant hook actually executes (not silently a no-op).
- The model doesn't crash or generate garbage.

If this preliminary test gives ~78%+ (i.e., MiniCPM-SALA is more FP4-tolerant than champion's model), we may not even need per-layer ablation — could go FP4 on all 8.
If it gives ≤70%, our model is more sensitive than champion's; ablation must be precise.

---

## 5. Detailed implementation plan

### 5.1 File changes (M2.0 only)

| File | Change | LOC | Type |
|------|--------|-----|------|
| `python/sglang/srt/layers/attention/minicpm_flashinfer.py` (or wherever standard-attn forward lives) | Add fake-quant hook gated by env var `SGLANG_KV_FP4_SIM_LAYERS` | ~30 | MODIFY |
| `benchmark/soar/demo_sala/m20_ablation_runner.py` | New: orchestrator script | ~150 | NEW |
| `benchmark/soar/demo_sala/m20_ablation_results.csv` | New: per-layer results table | output | NEW (generated) |

### 5.2 Hook design (pseudocode)

```python
# In attention forward, after K and V are computed for this layer:
fp4_sim_layers = os.environ.get("SGLANG_KV_FP4_SIM_LAYERS", "")  # e.g. "0,5,7" or "all"
if fp4_sim_layers:
    layer_set = parse_layer_set(fp4_sim_layers, layer_id, num_attn_layers=8)
    if layer_id in layer_set:
        K = fake_quant_fp4_e2m1(K)  # round to FP4 grid, dequant back to BF16
        V = fake_quant_fp4_e2m1(V)
```

Where `fake_quant_fp4_e2m1` implements per-block (e.g., block_size=16) absmax scaling + round-to-nearest on the FP4 E2M1 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`, then dequantizes back.

### 5.3 Ablation runner design

```python
# m20_ablation_runner.py — pseudocode
STANDARD_LAYER_IDS = read_from_config_json("mixer_types")  # the 8 layers where mixer_type=="minicpm4"

results = []
# Phase 0: baseline (no FP4) and full-FP4 sanity
for label, env_val in [("baseline_fp8", ""), ("all_8_fp4_sim", "all")]:
    set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=env_val)
    acc = run_accuracy_eval()
    results.append({"label": label, "layers_fp4": env_val, "acc": acc})

# Phase 1: per-layer ablation (8 runs)
for lid in STANDARD_LAYER_IDS:
    set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=str(lid))
    acc = run_accuracy_eval()
    results.append({"label": f"only_layer_{lid}_fp4", "layers_fp4": str(lid), "acc": acc})

# Phase 2 (optional, only if Phase 1 shows clear sensitive set):
# subset test — confirm "all robust layers FP4 simultaneously" still passes
robust_set = [lid for lid, acc in phase1 if acc >= 78.0]
set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=",".join(robust_set))
acc = run_accuracy_eval()
results.append({"label": "robust_subset_fp4", ...})

write_csv(results)
```

### 5.4 Validation commands

```bash
# After approval, on local repo:
git checkout -b m20_kv_fp4_ablation 8d1e4d12b
# ... implement hook + runner ...
git push minicpm-src m20_kv_fp4_ablation

# On fcloud:
python3 scripts/fcloud/fcloud_workflow.py sync
# For each of 10 runs (1 baseline + 1 all-FP4 + 8 per-layer + 1 robust-subset):
SGLANG_KV_FP4_SIM_LAYERS="<value>" python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

Total compute: ~10 × (3 min restart + 5 min accuracy) ≈ **80 min**.

---

## 6. Decision matrix (post-ablation)

| Scenario | Outcome | Next action |
|----------|---------|-------------|
| All-FP4 already gives ≥78% acc | MiniCPM-SALA is highly FP4-tolerant | Skip M2.1 hook complexity; commit to all-8-layers-FP4; go straight to real pool integration |
| Subset gives ≥78%, all-FP4 < 78% | Mixed precision is the right answer | **GO** to M2.1 with the discovered subset |
| Best subset < 78% but ≥ 76% | Risky but possibly worth pursuing if speed gain is large | Conditional GO; hold real-pool work until we've confirmed Smax gain via M2.1 dry-run |
| Best subset < 76% | KV FP4 path is dead | **NO-GO**; pivot to other catalog items (SM120 native MMA, FP8 weights, etc.) |

---

## 7. Result summary template (to fill after run)

| Run | Layers in FP4 | acc_ori | mcq | qa | cwe | fwe | niah | Notes |
|-----|---------------|---------|-----|-----|-----|-----|------|-------|
| 0 | none (baseline) | TBD | | | | | | |
| 1 | all 8 standard | TBD | | | | | | |
| 2 | only L_a | TBD | | | | | | |
| ... | ... | | | | | | | |
| 9 | only L_h | TBD | | | | | | |
| 10 | robust subset | TBD | | | | | | |

(Layer IDs to be filled from `config.json` on fcloud.)

---

## 8. Rollback instructions

M2.0 changes are env-gated and additive:

```bash
# Disable: simply unset env var (default behavior is unchanged FP8)
unset SGLANG_KV_FP4_SIM_LAYERS

# Full revert (remove the hook code):
git revert <m20_hook_commit>

# Discard branch:
git branch -D m20_kv_fp4_ablation
git push minicpm-src --delete m20_kv_fp4_ablation
```

Submission package is **never** affected by M2.0 (the branch is local-only until M2.1 ships).

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| MiniCPM-SALA is fully FP4-sensitive (no robust subset) | Medium | M2 dies cheap with only ~80 min of fcloud time burned |
| Fake-quant doesn't perfectly match real FP4 numerics (e.g., we miss FP4 min-norm or denormal rounding edge cases) | Low | Validate by also running 1× true FP4 (Option A) at the end of M2.0 to compare with Option B's full-8 result; if they match within 0.5pt, fake-quant is trustworthy |
| KV bandwidth from 8 standard layers is small fraction of total decode bandwidth (vs SimpleGLA state I/O for 24 layers) | Medium | Use SGLang built-in profiler during baseline to measure; if standard-attn KV reads are <30% of decode bandwidth, M2 ROI shrinks and we should not commit to M2.1 even if acc passes |
| Hook adds latency overhead even when FP4 disabled | Very low | Env var read at server startup, layer-set frozen; no per-token overhead |

---

## 10. Next-step suggestions (post-M2.0)

| Phase | Trigger | Effort |
|-------|---------|--------|
| **M2.1** | M2.0 GO decision | 3-5 days: per-layer dtype config in `model_runner_kv_cache_mixin.py`, dual-pool coexistence, attention layer indexing fix |
| M2.2 | M2.1 ships | 2-3 days: fcloud full speed test with real FP4 storage; expected +15-30% Smax |
| M3 (parallel) | Independent | 1-2 days: rebuild local `speed_{s1,s8,smax}.jsonl` with long-context inputs to match official distribution |
| M4 (stretch) | Only if M2 underwhelms | 2+ weeks: SM120-native MMA / mxfp8 GEMM (per CHANGE_0125_001 recommendation) |

---

## 11. Open questions for user

1. **Confirm M2.0 scope**: Option B (fake-quant) only, then decide on M2.1 based on results — agreed?
2. **Branch strategy**: create a separate `m20_kv_fp4_ablation` branch, or commit to `mixed_minicpm_cudagraph` directly?
3. **Validation of fake-quant fidelity**: include the optional 1× real FP4 cross-check at end of M2.0 (adds ~10 min to budget)?
4. **KV bandwidth profiling**: do an upfront `nsys` profile to measure standard-attn-KV bandwidth share before committing to ablation?

---

## Appendix A — Why fake-quant in BF16 gives identical accuracy to true FP4

Real FP4 KV cache flow:
```
write:  K_bf16 → quantize_to_fp4(K_bf16, scale) → store as packed uint8 + scale
read:   load packed uint8 + scale → dequantize_to_bf16 → K_bf16'
```

Fake-quant flow:
```
write:  K_bf16 → quantize_to_fp4 → dequantize_to_bf16 → store as bf16 (same K_bf16')
read:   load bf16 (already K_bf16')
```

`K_bf16'` (the dequantized representation) is **bitwise identical** in both flows. Therefore attention's downstream computation produces identical logits, identical token IDs, identical accuracy. The only difference is bandwidth/storage, which M2.0 doesn't measure.

Reference: this is the standard fake-quant technique used in PyTorch QAT (`torch.fake_quantize_per_tensor_affine`) and is mathematically rigorous.
