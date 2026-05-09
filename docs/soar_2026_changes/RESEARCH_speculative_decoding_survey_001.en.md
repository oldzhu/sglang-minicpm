# RESEARCH — Speculative Decoding Survey for MiniCPM-SALA (iter 001)

Date: 2026-05-09
Branch: `mixed_minicpm_cudagraph`
Companion: [PROPOSAL_medusa_minicpm_sala_001.en.md](PROPOSAL_medusa_minicpm_sala_001.en.md)

Goal: provide a side-by-side decision matrix for the speculative-decoding family on the SOAR 2026 baseline (GPTQ + FP8_e5m2 KV + dense + Tier1 + flashinfer, commit `ac91b1afe`). Used to justify Medusa as the recommended next move.

## 1. Common factors that constrain every option

The MiniCPM-SALA architecture imposes 3 constraints that **every** speculative-decoding scheme must respect, regardless of how the draft tokens are produced:

1. **24 GLA (Lightning Attention) recurrent layers** with state `h_t = exp(−γ)·h_{t−1} + k_t·v_tᵀ`. Tree verify of K candidates needs a per-branch state fork; a linear-history implementation will silently corrupt sibling branches.
2. **8 dense self-attention layers** (and standard KV cache pages). Tree-mask plumbing is well-trodden in sglang's EAGLE worker; reusable.
3. **2 GB submission tarball cap**. Anything draft-side that ships weights must be tiny (~100 MB). Trained drafts that share the main `lm_head` save hundreds of MB.

Anything that ignores #1 fails like our **Test 22 EAGLE3** (random draft, accept_rate=0.26, S1 +65 %, C=0). The lesson is not "EAGLE is broken" but "the GLA-fork bug must be solved at the framework layer regardless of which draft generator you bolt on top".

## 2. Methods compared

| Method | Draft producer | Training cost | Submission size | GLA-fork required? | sglang support today | Best concurrency |
|---|---|---|---|---|---|---|
| **Medusa** | K MLP heads on last hidden state, share `lm_head` | Hours on 1 GPU | ~50–100 MB | **Yes** | None | S1 ★★★ |
| EAGLE / EAGLE3 | Small autoregressive draft model (1 transformer layer + classifier) | Days on multi-GPU | ~200–400 MB | **Yes** | Full (`speculative-algorithm EAGLE3`) | S1 ★★★ |
| MTP (DeepSeek-V3 / Meta) | Extra transformer blocks at the end of main model trained with auxiliary loss | Co-trained at pre-train time | Adds blocks to main weights | **Yes** | Partial (`NEXTN`) | S1 ★★ |
| Lookahead / SpS / n-gram | Lookup of previously-emitted n-grams from the same request | Zero | Zero | No (no recurrent history) | Full (`NGRAM`) | S1 ★★ |
| Self-speculative (skip layers) | Use main model with last few layers skipped as draft | Zero | Zero | **Yes (worse)** — skipping lightning layers changes recurrence | None | S1 ★★ |
| Standalone draft model | Separate small LM | Days | ~500 MB+ | **Yes** | Full (`STANDALONE`) | S1 ★★ |

(★ = expected speedup magnitude on this baseline; based on champion's reported behavior + Test 22 evidence + sglang docs.)

## 3. Per-method analysis

### 3.1 Medusa (recommended — see PROPOSAL_001)

**Mechanism.** K lightweight MLP heads on the main model's last hidden state predict positions t+1, t+2, …, t+K. Top-s candidates per head form a Cartesian product → tree → main model verifies all paths in one forward → longest accepted prefix wins.

**Pros**
- **Smallest submission cost.** With shared `lm_head`, K=2 heads ≈ 64 MB; trivially fits 2 GB.
- **Lossless when verify is correct.** Same accuracy as no-spec baseline.
- **Verify overhead ≈0.39 ms at K=1** (champion's measurement). Lower than one decode step.
- **Independent of weight quantization.** Works on top of GPTQ, FP8, NVFP4, BF16 alike. We can ship it on GPTQ now, retrofit to NVFP4 later if FOS is recovered.
- **No co-training of the main model.** Heads train alone, main weights stay byte-identical. Submission accuracy doesn't drift.

**Cons**
- Needs the GLA state-fork plumbing (real engineering work).
- sglang has zero Medusa code; we are building it from scratch.
- Heads are SALA-specific; can't reuse champion's weights even if open-source (ours is a quantized GPTQ build, theirs NVFP4).

**Risk profile.** Concentrated almost entirely in correctness of the GLA-fork (R2 of the proposal). Once the byte-identity gate at `accept_threshold=1.0` passes, speed wins are essentially free.

### 3.2 EAGLE / EAGLE3

**Mechanism.** A small autoregressive draft model (1 transformer block + classifier) is trained on top of the main model's hidden states. EAGLE3 adds multi-token training and a richer verification tree.

**Pros**
- **Already integrated in sglang** (`--speculative-algorithm EAGLE3`).
- Higher accept rates than Medusa in published results on Llama-class models (article reports 0.6+).
- Tree-verify infra reusable.

**Cons**
- **Training cost is 5–10× Medusa's.** EAGLE3 needs several GPU-days on calibration data; our compute budget for offline work is limited.
- **Submission size 200–400 MB** for draft layers + classifier (no `lm_head` reuse). Still under 2 GB but eats budget.
- **Same GLA-fork blocker** as Medusa. We hit it once already in Test 22 (commit 548c8c153, mem-frac 0.72) — accept_rate=0.26 with random draft, S1 +65 %, C=0. Trained draft would help accept_rate but not the underlying state-fork bug; it would still silently corrupt sibling branches.
- **Co-dependency on tokenizer/vocab.** SALA's vocab is non-standard; we'd need to rebuild the draft from scratch.

**Verdict.** EAGLE3 dominates Medusa **only after** GLA-fork is solved AND we can afford 5+ GPU-days of training. Today neither is true. **Defer behind Medusa.** If Medusa K=2 caps out at accept_rate ≈ 0.5, revisit EAGLE3 as the upgrade path.

### 3.3 MTP (Multi-Token Prediction, DeepSeek-V3 / Meta-style)

**Mechanism.** Extra transformer blocks at the end of the main model are trained with an auxiliary "next-k" prediction loss during the main model's pretraining. Inference uses these blocks to draft multiple tokens; main model verifies.

**Pros**
- Highest reported accept rates in the literature (DeepSeek-V3 reports 0.85+ on long-context).
- Tightest integration → smallest verify mismatch.

**Cons (decisive)**
- **Requires co-training the main model** — we cannot retrain MiniCPM-SALA's base weights (the baseline accuracy is the Olympic measurement; modifying main weights breaks that immediately).
- **Submission rule conflict**: rules forbid pre-quantized weight submissions. Re-pretraining for MTP and re-quantizing on-site within 5 hours is infeasible.
- Still subject to GLA-fork.

**Verdict.** **Hard rule-out.** MTP requires base-model training we cannot do in this competition.

### 3.4 N-gram / Lookahead / SpS (sglang `NGRAM`)

**Mechanism.** Maintain a history table of n-grams from previously emitted tokens for the **same request**. When the prefix matches, propose the historical continuation as the draft.

**Pros**
- **Zero training, zero submission size.**
- **Zero GLA-fork concern** (no recurrent draft model state — every token is verified by the main model from scratch).
- Already in sglang.

**Cons**
- Accept rate is highly distribution-dependent. Strong on repetitive code/text (where prefix repeats often). On the SOAR competition tasks (qa/mcq/cwe/fwe/niah), only `niah` and `cwe` have strong repetition; `qa`, `mcq`, `fwe` are mostly novel-token streams where n-gram lookup misses.
- Champion explicitly chose Medusa over n-gram, suggesting n-gram alone is insufficient on this workload.
- Speedup ceiling is much lower than learned drafts (typically +10–15 % at S1).

**Verdict.** **Cheap insurance, not a primary lever.** Could be turned on as a free supplement to Medusa once correctness is locked. Not worth pursuing on its own.

### 3.5 Self-speculative decoding (skip layers)

**Mechanism.** Use the same main model with the last N transformer blocks skipped as the draft.

**Pros**
- Zero training, zero submission size.

**Cons (decisive)**
- **Skipping any of the 24 lightning layers breaks the recurrent state evolution** — the "draft" is no longer just a less-accurate version of the main model, it produces a fundamentally different output distribution. Accept rates collapse.
- Skipping only the 8 dense self-attention layers leaves no useful "shorter" path (lightning layers do most of the FLOPs).

**Verdict.** **Architecturally unfavorable for SALA.** Skip.

### 3.6 Standalone draft model (sglang `STANDALONE`)

**Mechanism.** Train a separately-architected small LM as the draft.

**Pros**
- Already supported in sglang.

**Cons**
- Submission size: trained tiny LM ≥ 200 MB.
- Training cost similar to EAGLE3.
- **Worst accept-rate per parameter** of all options (no shared hidden states with main model).
- Same GLA-fork concerns.

**Verdict.** **Strictly dominated by Medusa and EAGLE3.** No reason to consider.

## 4. Decision matrix

Score each option on 6 axes (1–5; higher = better):

| Method | Speedup at S1 | Accuracy safety | Training cost (lower=better) | Submission size (smaller=better) | sglang support | GLA-fork pain (lower=better) | **Total** |
|---|---|---|---|---|---|---|---|
| Medusa | 5 | 5 | 4 | 5 | 1 | 3 | **23** |
| EAGLE3 | 5 | 4 | 2 | 3 | 5 | 3 | 22 |
| n-gram | 2 | 5 | 5 | 5 | 5 | 5 | 27¹ |
| MTP | 5 | 5 | 0 | 2 | 2 | 3 | rule-out |
| Self-spec | 2 | 1 | 5 | 5 | 1 | 0 | 14 |
| Standalone | 3 | 4 | 2 | 2 | 5 | 3 | 19 |

¹ N-gram's high score reflects it being free and safe — but its low S1 ceiling means it complements rather than replaces a learned draft. Recommended as a stack-on after Medusa lands.

## 5. Recommendation

1. **Primary: Medusa** (per PROPOSAL_medusa_minicpm_sala_001). Highest expected gain on S1 (which holds 40 % of the score), smallest submission cost, no main-model retraining. The GLA-fork engineering work is unavoidable for any learned-draft method on SALA.
2. **Insurance: n-gram (`--speculative-algorithm NGRAM`).** Ship-as-a-safety-net once Medusa correctness is locked. Cost ≈ zero. Could deliver a few extra percent on cwe/niah where prefix repetition is highest.
3. **Backup: EAGLE3.** Only if Medusa K=2 caps below 0.5 accept_rate. Pre-condition is the same GLA-fork plumbing being already built (so the work transfers).
4. **Hard rule-out: MTP, Self-spec, Standalone.** Either rule-incompatible or strictly dominated.

## 6. Concrete next steps (no code yet)

- Approve PROPOSAL_medusa_minicpm_sala_001 R1 (plumbing spike). Cost: 2–3 days code + 1 fcloud round.
- Optionally schedule a free `NGRAM` smoketest in parallel (purely a server-arg change; zero code; zero risk). Could yield a quick small win unrelated to the Medusa work.

## 7. Open questions

- Has the champion released their Medusa head training code? If yes, we may be able to reuse their data-curation script (with their permission). If no, we replicate from the article description.
- Does sglang's existing tree-mask code in `eagle_utils.py` cleanly factor out for Medusa's dense head topology? (Answer pending R1 spike.)
- Will torch.compile play with the verify forward graph or do we need to disable compilation under spec mode? (Champion does not say; assume off in R1, revisit in R3.)

## References

- Medusa paper: Cai et al., ICML 2024 — https://arxiv.org/abs/2401.10774
- EAGLE / EAGLE3: Li et al. — https://arxiv.org/abs/2503.01840
- DeepSeek-V3 MTP: DeepSeek tech report
- N-gram speculative decoding: sglang docs `advanced_features/speculative_decoding.ipynb`
- Champion article: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- Our prior EAGLE3 attempt: TEST_RESULTS_TRACKING.md Test 22-acc; CHANGE_0072_three_path_optimization_research.en.md
