# RESEARCH — Week 7 Champion (香草小张) Technical Review

**Date**: 2026-05-04
**Source**: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
**Champion**: 香草小张 (HUST undergrad team) — Week 7 (2026-04-29), score **88.35** (#1)
**Predecessor jumps**: prior best ~81-82 → 88.35 (+~7pt) in single submission
**Implication**: top 1-2 (88.35 / 86.68) clearly broke decisively away from rest (#3 = 67.8) — same recipe both teams.

## Two pillars of champion's gain

### 1. NVFP4 weight quantization with **FourOverSix** adaptive block scale

**Base path**: GPTQ + NVFP4 (FP4 E2M1) + Marlin W4A16 decode.

**NVFP4 representable values**: `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`. Max absolute = 6.

**Standard NVFP4 problem**: each block normalized to `[-6, 6]` via `M=6` scale upper bound.
The values 4 and 6 are far apart → for blocks where weights cluster in `[2/3·6, 6]` the
quantization error concentrates because nothing represents `4 < |x| < 6` precisely
except the 4 and 6 endpoints themselves (the spacing between 4 and 6 is wide).

**FourOverSix idea** (paper: Cook/Guo/Xiao/Lin/Han, MIT+NVIDIA, arXiv:2512.02010):
For each block, choose between **M=6** and **M=4** scale upper bounds — pick the one
with smaller dequant MSE. M=4 trades off the (4, 6] range for finer resolution in the
heavy [2, 4] region (block weights mapped to [-4, 4] with `2, 3, 4` all available).

```
# M=4 corresponds to block scale ≈ M=6's scale × 1.5
scale_m4 = fp8(scale_m6.float() * 1.5)
mse_m6   = ((W_block - dequant(W_block, scale_m6)) ** 2).mean()
mse_m4   = ((W_block - dequant(W_block, scale_m4)) ** 2).mean()
final_scale = scale_m4 if mse_m4 < mse_m6 else scale_m6
```

**Output format unchanged**: standard NVFP4 = 4-bit weight + FP8 block scale.
Inference kernel **does not need any change**. Throughput equal to baseline NVFP4.

**Champion's integration**: embed adaptive scale-pick into GPTQ:
```
estimate block scale → compare M=6 vs M=4 dequant error
                    → pick lower-error scale
                    → enter GPTQ weight optimization iteration
```
Decide M first, then run GPTQ within that frame (avoids scale-vs-GPTQ interaction).

**Observed**: 40-43% of blocks pick M=4. MLP layers have higher M=4 ratio than
attention QKV. Stable, measurable accuracy gain across all eval tasks.

### 2. Medusa speculative decoding adapted to GLA hybrid attention

**Base concept** (Cai et al. ICML 2024): no separate draft model — add multiple
lightweight prediction heads on top of main model's last hidden state. Each head
`k` predicts the `k`-th future token:
```
p_t^(k) = softmax( W₂⁽ᵏ⁾ · (SiLU(W₁⁽ᵏ⁾ · h_t) + h_t) )
```
`W₁` initialized to **zero** so early training behaves like main model (stable).

**Tree attention verify**:
1. Each head emits top-`s` candidates.
2. Cartesian product builds a candidate tree (e.g., 2 heads × top-2 = 4 paths).
3. **One** main-model forward pass verifies the entire tree by modifying attention
   mask so each node only sees its prefix (not sibling branches).
4. Pick longest accepted prefix.

**MiniCPM-SALA wrinkle — GLA recurrent state**: most layers are GLA (gated linear
attention) with recurrence
```
h_t = exp(-γ) · h_{t-1} + k_t · v_t^T
```
Tree verify breaks this: sibling branches must each fork from the same parent
state. Linear inheritance from previous candidate corrupts the recurrent state
with cross-branch history → wrong answers.

**Champion's fix**: per-verify-path **GLA state branching logic** — each branch
forks `h_{parent}` independently, no cross-pollination.

**Training data weighting**: weighted resampling of training data toward eval
distribution → measurable accept-rate gain over uniform sampling.

**Numbers reported**:
- Medusa K=1 verify overhead ≈ 0.39 ms — well under one decode step.
- Steady positive decode-throughput gain at all concurrency levels.

## Why this combination wins the leaderboard

| Lever | Acc effect | Speed effect | Layers touched |
|-------|-----------|--------------|----------------|
| FourOverSix | +0.5-1pt (recovers C tier room) | None directly; allows W4A16 with safer accuracy floor | All linear |
| Medusa K=1 | Lossless (verify rejects mismatches) | Decode tokens/step ~+15-30% across S1/S8/Smax | New heads + GLA forking |
| **Combined** | C ≥ 0.96 maintained | Decode dominant on Smax | — |

Decode-side gains stack across **all three** speed tiers (S1/S8/Smax all 30%+ weight).
Champion went from sub-82 → 88.35 with this single submission window.

## What we still don't know (need to investigate)

1. **Training corpus** for Medusa heads — champion says "weighted toward eval
   distribution" but eval-set text is private; their proxy distribution is
   undisclosed. Implication: we must construct our own proxy (long-context
   QA/CWE samples from public sources).
2. **Quantization-aware training of Medusa heads** — unclear if heads were
   trained on FP16 main model or NVFP4 quantized main model. Order matters
   for accept rate.
3. **Tree shape** (heads × top-s × depth) — not stated. Standard Medusa-1 paper
   uses 5 heads × top-s with depth-2 trees. K=1 in champion's text could mean
   "1 head" OR "depth 1". Verify-overhead 0.39 ms is consistent with shallow tree.
4. **Marlin kernel** — they keep stock Marlin W4A16 path (no kernel changes).
   Our codebase already has Marlin via sgl-kernel; check NVFP4 W4A16 support.

## Cross-references in our repo

- Existing Marlin work: [docs/soar_2026_changes/CHANGE_0075_*](CHANGE_0075_marlin_kv.md) (if applicable).
- SM120 hardware notes: [docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md) — FP4 = 593 TFLOPS (4× BF16).
- Optimization catalog: [docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md).
- Leaderboard 2026-05-04: top 1-2 = 88.35 / 86.68; team-beta = 30.04 (#22). Gap to #5 = +20.62 (68%).

## Action: see proposal

A separate proposal document drafts a phased plan to reproduce these two
optimizations: [PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md).
