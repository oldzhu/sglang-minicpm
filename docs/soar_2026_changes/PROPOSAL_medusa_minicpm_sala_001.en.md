# PROPOSAL — Medusa speculative decoding on MiniCPM-SALA (iter 001)

Status: **PROPOSAL — not yet implemented; awaiting approval**
Date: 2026-05-09
Branch: `mixed_minicpm_cudagraph`
Baseline: GPTQ sparse_qkv_w8 + FP8_e5m2 KV + dense + Tier1 + flashinfer + torch.compile bs=24 (commit `ac91b1afe`)
Best local numbers (today's retest, `GPTQ-FP8-DENSE-retest-newinst`): ori_acc=77.47%, norm=96.83%, **S1=110.68 / S8=40.33 / Smax=32.53**.

## 1. Motivation

The Champion of SOAR Week-7 ("香草小张") reports adding **Medusa speculative decoding with GLA-state forking** to MiniCPM-SALA delivers stable end-to-end throughput gains across all concurrency levels, with K=1 verify overhead **≈0.39 ms** (article: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q). Medusa specifically attacks our largest scoring leverage: **S₁ (40 % of final score)** is decode-bound and weight-bandwidth-limited at concurrency=1, which is exactly Medusa's strongest regime.

Our submission's score formula is:
```
Final Score = (S₁·0.40 + S₈·0.30 + S∞·0.30) × C
S_N = (Duration_best / Duration_player) × 100
```
At today's S1=110.68 s vs the leader's official best (we know v18-A was 426 s and the leader is faster), every −1 s on our local S1 is a directly-measurable score gain provided correctness coefficient C stays at 1.0.

## 2. Why Medusa beats every other speed lever in our catalog

| Lever | Status / Result | Verdict |
|---|---|---|
| Marlin tile re-tuning (CHANGE_0125) | Neutral on Test 27 | Saturated |
| W4A8 / FP8-blockwise GEMM (CHANGE_W4A8_001) | Test 27a: S1 +118% | Hostile |
| Sparse path on HEAD | Round 13d, R13e, CHANGE_0136: hangs / pre-existing bugs | Blocked |
| Aggressive scheduling (Tier1) | v22 default already | Saturated |
| `torch.compile` bs sweep (#2A bs=24) | Smax −3.2%, accuracy stable | Already shipped |
| **Medusa K=1/K=2** | **Champion-validated, untouched in our codebase** | **Pursue** |
| EAGLE3 with trained draft | Test 22 with random draft was C=0 disaster; untested with trained draft | Backup option (see RESEARCH doc) |

Speed-side ROI: Medusa K=1 with accept_rate p≈0.5 yields ≈ 1+p effective tokens per decode step → expected **−25–33 % on S1**, smaller on S8/S∞. Quantitatively, a 25 % S1 reduction (110.68→83 s) at constant relative position translates to roughly **+10 score points** in the S₁ bucket alone (since the leader's S1 sets `Duration_best` and our share moves up linearly in the inverse of our duration).

## 3. Rule-compliance check (SOAR 2026)

Quoting the official rules (per `.github/copilot-instructions.md`):

- ✅ **"Speculative heads allowed (count toward 2GB)"** — Medusa heads (3 × MLP, ≤100 MB) fit easily.
- ✅ **"Code: Apache 2.0, reproducible, explainable"** — sglang is Apache-2.0; our additions (model wrapper + worker glue + heads training script) will be Apache-2.0.
- ✅ **"Quantization + evaluation time ≤ 5 hours"** — head training is **offline**; only quant + eval count. No impact.
- ✅ **"All files ≤ 2GB total"** — current tarball ~731 MB; +100 MB heads stays under 2 GB.
- ✅ **"Submission is reproducible"** — head weights ship with the tarball; no external download needed at submission time.
- ✅ **Lossless when implemented correctly** — Medusa verify rejects any draft that diverges from the base model's argmax/sampled token, so accuracy can only differ from the no-spec baseline due to (a) numerical noise at the verify step or (b) bugs in the GLA state-fork path.

**Key correctness risk.** MiniCPM-SALA has 24 GLA (Lightning Attention) layers carrying recurrent state `h_t = exp(−γ)·h_{t−1} + k_t·v_tᵀ`. Tree-verify of K candidate paths must fork from the **parent** GLA state for each sibling, otherwise the recurrent state contaminates branches with foreign history → silent accuracy loss. This is precisely the bug that made **CHANGE_0072 / Test 22 EAGLE3** fail (`accept_rate=0.26`, `S1 +65 %`, `C=0`). Solving GLA-fork is the central technical task of this proposal.

## 4. Where we stand vs the champion's recipe

| Component | Champion | Us today |
|---|---|---|
| Base quant | NVFP4 FP4 + FourOverSix | GPTQ W4A16 sparse_qkv_w8 |
| GLA state-fork in tree verify | Implemented | **Missing** |
| Medusa heads (trained on eval-distribution-weighted data) | Implemented | **Missing** |
| Tree-attention mask glue | Implemented | sglang has it for EAGLE; needs adaptation |
| K (heads) | K=1 reported, K≥2 implied | TBD |
| Verify overhead | ≈0.39 ms at K=1 | TBD |

Note: the champion uses NVFP4 + Medusa together. Our local NVFP4-FOS run (CHANGE_0151_007) showed NVFP4 by itself **regressed S1 by +57 %** because the FP4 path lacks `flashinfer` decode kernels and triggers cuTLASS BF16 fallback. Therefore we will pursue Medusa **on top of GPTQ**, where the underlying decode kernels are already SM120-tuned. If we ever recover the NVFP4 path, the Medusa wiring will be reusable.

## 5. Proposed staging — four phases

The work is broken into four reviewable phases. **Each phase has its own pre-merge approval gate.** No source code is touched until you approve at least Phase R1.

### Phase R1 — Plumbing spike (random heads, no training)

Goal: prove tree-verify + GLA-fork run end-to-end on SALA without crashing or regressing accuracy.

Files (estimated):
- New: `python/sglang/srt/speculative/medusa_info.py` (data classes for tree topology, mirrors `eagle_info.py`)
- New: `python/sglang/srt/speculative/medusa_worker.py` (one-shot worker subclassing `base_spec_worker.BaseSpecWorker`; reuses sglang's tree-mask infra from `eagle_utils.py`)
- New: `python/sglang/srt/models/minicpm_medusa.py` (MiniCPM-SALA with K Medusa heads; MLP + classifier per the article: `p_t^(k) = softmax(W₂^(k)·(SiLU(W₁^(k)·h_t) + h_t))`; `W₁` zero-init)
- Modified: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` — expose a "branch_id" parameter when reading/writing GLA state buffers
- Modified: `python/sglang/srt/layers/attention/fla/{chunk.py,fused_recurrent.py}` — accept and respect a per-token branch index in the recurrent state path
- Modified: `python/sglang/srt/server_args.py` — register `MEDUSA` enum value (alongside existing `EAGLE`/`EAGLE3`/`NEXTN`/`STANDALONE`/`NGRAM`)
- Modified: `benchmark/soar/demo_sala/prepare_env.sh` — add `SOAR_SPEC_MEDUSA=1` opt-in (default 0; preserves byte-equivalent v22 baseline)

Validation:
- Local correctness diff: same input prompt, same seed, served once with `SOAR_SPEC_MEDUSA=0` and once with `=1` and **`accept_threshold=1.0` (only argmax)** → token sequences must be byte-identical for ≥10 representative prompts (mcq + niah + cwe + qa).
- One smoketest on fcloud: server boots, returns valid completions for a single `/generate`. Measure verify-step latency.
- **Abort gate**: if accuracy diverges from baseline at `accept_threshold=1.0`, stop. The GLA-fork bug is then in our implementation.

Effort estimate: 2–3 days code + 1 fcloud round (~1 h).

### Phase R2 — GLA state-fork correctness

Goal: extend R1 from `K=1` deterministic verify to actual tree branching and confirm verify overhead matches the champion's 0.39 ms ballpark.

Tasks:
- Implement branch-id aware GLA state buffer: shape `(batch, branch, heads, ...)`. Before each verify step, broadcast each parent state across its sibling branches.
- Wire the existing sglang `tree_mask` builder (in `srt/speculative/eagle_utils.py`) into the Medusa worker.
- Validation harness: a CPU reference that walks both the recurrent path and the verify tree path on a small prompt set; assert hidden-state equality at every accepted token.

Validation:
- 50-sample mcq + niah + cwe diff vs baseline at `accept_threshold=1.0`: must be byte-identical.
- 150-sample full eval at default sampling temp (`accept_threshold` floats per main-model probability): norm accuracy must stay within ±0.5 pt of today's 96.83 %.
- Speed: K=1 verify overhead per decode step measured; target ≤1 ms.

**Abort gate**: if K=1 lossless mode (`accept_threshold=1.0`) cannot be made byte-identical to baseline on at least 50 prompts, the GLA-fork is structurally wrong; do not advance to R3.

Effort estimate: 3–5 days code + 2 fcloud rounds.

### Phase R3 — Train Medusa heads on eval-aligned distribution

Goal: bring accept_rate into the 0.5–0.7 band.

Inputs:
- Training corpus: stratified sampling of the 5 task types in our `perf_public_set.jsonl` (qa/mcq/cwe/fwe/niah). Champion explicitly calls out that **distribution-weighted training data improves accept rate** vs random sampling.
- Heads: K∈{1,2,3}; one MLP per head as in §4.
- Loss: standard Medusa cross-entropy on next-k token, with main-model frozen.
- Compute: head weights ≈ (`hidden_size`)² × 2 + (`hidden_size`·vocab) × K. For SALA hidden_size≈4096 vocab≈73440 K=2 → ~2 × 32 MB MLP + 2 × 600 MB classifier = **~1.2 GB**, which **breaks the 2 GB submission budget** if naive. Mitigation: **share the main model's `lm_head` as the classifier** (the article does this implicitly — only `W₁`, `W₂` are head-specific MLPs, the final classifier is the existing LM head). With shared `lm_head` we are at ~2 × 32 MB ≈ 64 MB total → fits.

Validation:
- Head-only loss curves on a held-out slice.
- Re-eval on fcloud: norm accuracy ≥ today's 96.83 %, accept_rate ≥ 0.45.

Effort estimate: training ~1 day on a single H100/RTX 6000; pipeline + curation 2 days; integration 1 day.

### Phase R4 — Bench + variance probe + submission package

- Run accuracy + speed (S1, S8, Smax) twice for variance.
- Update `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` with measured speedup.
- Build the submission tarball; verify ≤2 GB; verify `prepare_model.sh` still completes within budget with heads loading.
- Two-run consensus rule: ship only if both runs show C ≥ 0.96 and S1 improvement ≥ 10 %.

## 6. Files & lines that will change (R1–R2 estimate)

```
python/sglang/srt/speculative/medusa_info.py        +200 (new)
python/sglang/srt/speculative/medusa_worker.py      +400 (new)
python/sglang/srt/models/minicpm_medusa.py          +250 (new)
python/sglang/srt/layers/attention/fla/chunk.py     +50  (branch_id arg)
python/sglang/srt/layers/attention/fla/fused_recurrent.py  +30
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py  +80
python/sglang/srt/server_args.py                    +5
python/sglang/srt/speculative/eagle_utils.py        +20  (re-export tree_mask helper)
benchmark/soar/demo_sala/prepare_env.sh             +20  (opt-in env)
benchmark/soar/demo_sala/preprocess_model.py        +10  (head weights pass-through)
```

R3 adds a `tools/train_medusa_heads.py` (~300 lines) and the trained `.safetensors`. R4 only adds packaging glue.

## 7. Test commands

After Phase R1:
```
# fcloud
cd /root/submission_sim
source prepare_env.sh
export SOAR_SPEC_MEDUSA=1
python3 -m sglang.launch_server --model-path "$MODEL_PATH" "${SGLANG_SERVER_ARGS[@]}"
# expect: server boot OK, single /generate returns the same first token as SOAR_SPEC_MEDUSA=0
```

After Phase R2:
```
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
# diff predictions.jsonl vs baseline at temperature=0; accuracy norm Δ within ±0.5pt
```

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GLA-fork bug subtle, silent accuracy loss | High | Lossless `accept_threshold=1.0` byte-identity gate before any speed claim |
| `torch.compile` graph bloat (verify forward ≠ decode forward) → boot time blow-up | Medium | Disable torch.compile under `SOAR_SPEC_MEDUSA=1` initially; revisit after R3 |
| Head training collapses to base distribution (low accept_rate) | Medium | Use champion's eval-distribution-weighted curation; K=1 first, K=2 only after K=1 ≥0.5 |
| Submission size overflows 2 GB | Low | Share `lm_head` as classifier; budget already verified at ≤100 MB heads |
| Fcloud SM120 doesn't expose enough memory for verify+main concurrent | Low | Verify forward and main forward share weights; only state buffers grow |

## 9. Rollback plan

All work is gated by `SOAR_SPEC_MEDUSA=0` (default). Rollback = unset env var. New files are additions; modified files use early-return when the env is off.

## 10. Estimated total effort

| Phase | Effort | Fcloud rounds | Cumulative |
|---|---|---|---|
| R1 plumbing | 2–3 days | 1 | 3 d |
| R2 GLA-fork | 3–5 days | 2 | 8 d |
| R3 heads training | 4 days | 1 | 12 d |
| R4 bench + package | 1 day | 1 | 13 d |

## 11. Decision gates

You will be asked to approve **before** each of: R1 spike, R2 fork, R3 training run, R4 packaging. Each phase produces its own bilingual CHANGE_NNNN report.

## 12. Next concrete action (awaiting your "go")

Approve Phase R1 only. I will:
1. Implement R1 file deltas in `mixed_minicpm_cudagraph` branch.
2. Run a local syntax / unit-test pass.
3. After fcloud is healthy: smoketest on fcloud (single `/generate`, verify-step latency).
4. Write `CHANGE_0153_medusa_phase_r1.{en,zh}.md`, commit + push.

**Awaiting explicit approval to begin R1.** Until then this proposal stands as documentation only; no code changes will be made.

## References

- Champion article: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- Medusa paper: Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" — ICML 2024 (https://arxiv.org/abs/2401.10774)
- Companion survey: [RESEARCH_speculative_decoding_survey_001.en.md](RESEARCH_speculative_decoding_survey_001.en.md)
- Prior failed EAGLE3 spike: [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) Test 22-acc
- Prior triage of EAGLE3 / GLA blocker: [CHANGE_0072_three_path_optimization_research.en.md](CHANGE_0072_three_path_optimization_research.en.md)
