# Post-v18 Baseline Decision & M2 Mixed-Precision KV Cache Plan

**Date**: 2026-04-26 10:29
**Author**: agent (Claude)
**Status**: Awaiting code execution; user-approved decisions captured
**Supersedes**: prior verbal proposals only

---

## Context: latest official results (3 most recent submissions)

| Submission | acc_ori | C | S1 | S8 | Smax | final_score |
|---|---|---|---|---|---|---|
| pre-v18 (current leaderboard) | 79.24 | **1.0** | 596.25 | **1066.35** | **2746.62** | **39.25** |
| v18 | 80.13 | **1.0** | **586.64** | 1087.90 | 2857.64 | 39.01 |
| v19 | 78.07 | 0.92 | 591.04 | 1113.65 | 2917.74 | 35.37 |

**Leaderboard**: team-beta No.20 at 39.25; gap to #5 (Slightwind 79.52) = **2.03×** improvement needed.

---

## (1) Local-vs-official S8/Smax inversion analysis

### Observation
| Tier | fcloud (local) | official |
|------|----------------|----------|
| S1 | v18 wins | v18 wins ✓ aligned |
| S8 | v18 wins | **pre-v18 wins** ✗ inverted |
| Smax | v18 wins | **pre-v18 wins** ✗ inverted |

### Interpretation
v18's "improvements" over pre-v18 are tuned for **short-context** workload (our local
`speed_s8.jsonl` / `speed_smax.jsonl` have shorter inputs than the official set). On
official's long-context S8 (1066s) and Smax (2746s), pre-v18 is genuinely faster.

### Likely root cause
The aggressive scheduler args added in v18 era — `--prefill-max-req=1`,
`--sched-cons=1.0`, `--enable-fused-qk-norm-rope`, torch.compile max-bs=8, chunk=32K —
help short-batch S1 but **fragment long-context prefill**.

The 2026-04-20 config sweep (catalog file) found best-on-old-data was
`prefill-max-req=4, sched-cons=0.8`, which matches pre-v18.

### Implication
- Keep v18's **source-code** changes (fused qk-norm-rope, torch.compile) — net positive
- v18's **`SGLANG_SERVER_ARGS`** may be over-tuned for our local benchmark — needs A/B test
- This becomes a **separate M3 track**: rebuild long-context-faithful local speed set so
  we stop optimizing for the wrong target. Not blocking M2.

---

## (2) Repository revert decision: **Option C (surgical file-level revert)**

### User decision
**Approved: Option C** — surgical revert of v19-only source changes via `git checkout` of
specific files from v18 commit, preserving any neutral v19 changes (e.g., FLA kernels
which our isolation tests showed were beneficial when paired with v18 minicpm.py).

### Files to identify and classify
Need to:
1. Identify v18 commit hash (commit that produced `minicpm_sala_submit_v18.tar.gz`)
2. Run `git diff v18..HEAD --name-only` to enumerate v19 changes
3. Classify each file:
   - **REVERT**: changed by v19 in a way our isolation test proved harmful
     (definitely: `srt/models/minicpm.py`, possibly: `srt/models/minicpm_eagle3.py`,
     `srt/speculative/eagle_worker.py` — these caused the 3000s hang at sample 138)
   - **KEEP**: v19 changes our isolation test proved neutral or beneficial
     (FLA kernel files — v19-c-FLA test confirmed v19 FLA was beneficial; reverting them
     dropped acc from 78.04 → 76.29)
   - **TBD**: any change not directly tested in isolation; needs case-by-case judgment

### Execution plan
```bash
git fetch minicpm-src
git log --oneline <v18_tag>..HEAD                 # list commits added since v18
git diff <v18_commit>..HEAD --name-only           # list files changed
# For each file in REVERT list:
git checkout <v18_commit> -- <path>
git commit -m "Revert v19 changes in <files>; baseline back to v18"
git push minicpm-src mixed_minicpm_cudagraph
```

### Validation after revert
1. Smoke-build sgl-kernel (incremental, ccache preserved)
2. fcloud accuracy run — must reproduce v18 baseline (≥80% acc, no hang at sample 138)
3. fcloud speed S1/S8/Smax — must match v18 reference (Test 12: 121.71/44.09/35.86)

### Branch hygiene
Working branch stays `mixed_minicpm_cudagraph`; revert commits are on top so history
is clean and any v19 piece can be re-cherry-picked later if salvageable.

---

## (3) Iteration M2 — Mixed-precision FP8/NVFP4 KV cache

### User decision
**Approved with phased start**: begin with **M2.0 ablation only**, decide further phases
based on M2.0 results.

### Strategic rationale
Week 5 champion (智算一队 semifinal) recipe — directly applicable to MiniCPM-SALA:
- Pure NVFP4 KV → ~75% acc (theirs) → would fail our C ≥ 0.92 floor
- Pure FP8 KV → ~80% acc → matches our v18 baseline
- **Mixed (FP8 first/last layers + NVFP4 middle layers) → ~80% acc with FP4 bandwidth**

Why this attacks our weakness specifically:
- Our worst official tier is **Smax = 2857s** (vs S1 = 587s)
- Official speed dataset: 25% inputs in 32K-128K, 26% in 128K-256K, 17% in 256K-512K
- Long-context decode is **memory-bandwidth-bound**, not compute-bound
- Halving KV-cache reads halves the dominant bottleneck

### Clarification: prior NVFP4 testing (answers user question)

| Test | What it tested | KV-cache dtype | Weight dtype | Activation dtype | Result | Decision |
|------|---------------|----------------|--------------|-------------------|--------|----------|
| **Test 21 (2026-04-16)** | "NVFP4 W4A4" via modelopt | **FP8** (unchanged) | **NVFP4 (W4)** | **NVFP4 (A4)** | acc ≈ 12% catastrophic; infinite `<think>` loops; 5× 3000s timeouts | Stopped FP4 **weight** quantization path |
| **Test 21 conclusion** | — | — | — | — | "FP4 too aggressive for reasoning" | Closed only the **W4A4** path |
| **(no prior test)** | NVFP4 KV cache | **NVFP4** | W4 (Marlin GPTQ) | BF16 | **NEVER TESTED** | M2 will fill this gap |

**Key correction**: We have **never** tested NVFP4 KV cache. The "75%" number from the
champion's blog refers to their pure-NVFP4-KV-cache experiment, not ours. Ours was a
different axis (weights+activations both FP4, KV stayed FP8). The catalog's
"NVFP4 Status: NOT VIABLE" was scoped to weight quantization, not KV cache — wording
should be updated.

### M2.0 ablation scope (answers "do we need new files for M2.0?")

**Goal**: produce a per-layer FP4-sensitivity CSV → identifies "sensitive" layers
(must stay FP8) vs "robust" layers (can drop to FP4) → drives M2.1+ design.

### CRITICAL CORRECTION: ablation count is 8, NOT 32

MiniCPM-SALA's 32 transformer layers are split by `config.mixer_types[layer_id]`:
- **8 layers with `mixer_type == "minicpm4"` (standard attention)** — have a paged KV
  cache → **subject to FP4 KV ablation**
- **24 layers with `mixer_type == "lightning"` (SimpleGLA)** — use recurrent state,
  **no KV cache** → **NOT applicable to this ablation**

The ablation runs **8 times**, once per standard attention layer. Need to extract the
exact 8 layer indices from `config.json` of `MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8`
on fcloud at run time — these are the only layers we touch.

**Implication for the champion's "first/last layers FP8, middle FP4" rule**: their
architecture differs from ours. We have only 8 standard-attention layers spread among
32 total, so "first/last" must mean "first/last among the 8 standard ones", not
"first/last among 32". The ablation tells us empirically which of our 8 are sensitive
— don't assume the champion's pattern transfers.

**Scope of M2.0** — designed to be **lowest-effort possible**:

#### Option M2.0-A: SGLang code-free ablation (preferred if feasible)
- Reuse existing `--kv-cache-dtype` flag and existing FP4 KV pool implementation **if
  any already exists in upstream SGLang**.
- We need to first survey SGLang's codebase for any existing FP4 KV cache support
  (some upstream PRs may have landed it for Blackwell; if so, M2.0 needs zero new code).
- If found: build a per-layer dtype-override env var hook (small, ~30 lines, single file)
  that lets us flip individual layer KV to FP4 while leaving rest at FP8.
- Run the harness **8 times** (once per `mixer_type=="minicpm4"` layer flipped to FP4).

**Files modified for M2.0-A** (if upstream FP4 KV exists):
1. `python/sglang/srt/layers/radix_attention.py` (or similar) — add per-layer dtype hook
   reading `SGLANG_KV_LAYER_DTYPE_OVERRIDE` env var
2. `benchmark/soar/demo_sala/m20_ablation_runner.py` (NEW) — orchestrator script that
   sets the env var, restarts server, runs accuracy eval, records result
3. `benchmark/soar/demo_sala/m20_ablation_results.csv` (NEW, output) — per-layer results

**No new memory pool. No new attention backend. No new kernels.**

#### Option M2.0-B: synthetic ablation (fallback if no upstream FP4 KV)
- If SGLang has no FP4 KV pool yet, simulate FP4 effects via **fake quantization** on
  KV in BF16 storage: round to FP4 grid then dequant back to BF16. No memory savings,
  but gives accuracy signal identical to true FP4.
- Implementation: a single hook in attention forward that conditionally applies
  `(x / scale).round().clamp(-6,6) * scale` to K and V tensors when env-flag set for
  that layer.

**Files modified for M2.0-B**:
1. `python/sglang/srt/layers/attention/minicpm_flashinfer.py` (or whichever SDPA impl) —
   add ~20-line fake-quant hook gated by env var
2. `benchmark/soar/demo_sala/m20_ablation_runner.py` (NEW)
3. `benchmark/soar/demo_sala/m20_ablation_results.csv` (NEW)

**Still no new memory pool. No new full attention backend.**

#### Why M2.0 is decoupled from M2.1+
- M2.0 only needs to answer: **"which layers tolerate FP4 in KV?"**
- It does NOT need true FP4 storage — only true FP4 numerics
- Fake-quant in BF16 storage gives identical accuracy signal at ~zero implementation cost
- Once M2.0 says e.g. "of the 8 standard layers, indices [a,b] sensitive, [c,d,e,f,g,h]
  robust", we then commit to M2.1 (real NVFP4 pool) and M2.2 (mixed-dtype attention
  backend) with FP4 enabled only on the 6 robust layers (lightning layers continue to
  use SimpleGLA recurrent state regardless of M2)

### Iteration plan (post-M2.0 phases, conditional on results)

| Phase | Effort | Output | Files modified |
|-------|--------|--------|----------------|
| **M2.0** | 1 day (8 layers × ~5min eval = ~40min compute + setup) | per-layer FP4 sensitivity CSV; go/no-go decision | 1 SGLang file + 2 new benchmark scripts |
| M2.1 (if M2.0 passes) | 3-5 days | NVFP4 KV memory pool | NEW: `python/sglang/srt/mem_cache/nvfp4_kv_pool.py` |
| M2.2 | 3-5 days | mixed-dtype attention backend | MODIFY: `python/sglang/srt/layers/attention/minicpm_flashinfer.py` (+ possibly new fused dequant CUDA kernel) |
| M2.3 | 2-3 days | per-layer config wiring + final fcloud test | MODIFY: `prepare_env.sh` (+ env var plumbing) |

### Go/no-go thresholds for M2.0 → M2.1
- **GO** if: there exists a layer subset where flipping to FP4 keeps acc ≥ 78%
  (≥1pt margin over our C=0.92 floor)
- **NO-GO** if: any subset large enough to give ≥10% bandwidth saving drops acc <77%
- **NO-GO** if: ablation reveals all layers are sensitive (acc ≤77% even when flipping
  a single layer) — would mean MiniCPM-SALA is more KV-sensitive than the champion's
  model

---

## Recommended immediate next actions

1. **Today**: identify v18 commit hash; produce `git diff v18..HEAD --name-only` with
   classification table → user approval
2. **Day 1**: execute Option C surgical revert; smoke-test on fcloud (acc + speed)
3. **Day 1**: survey SGLang for upstream FP4 KV cache support → decide M2.0-A vs M2.0-B
4. **Day 2**: implement M2.0 hook + runner script
5. **Day 2**: run **8-layer** ablation on fcloud (one accuracy eval per `minicpm4`
   standard-attention layer; ~5 min each → ~40 min compute + restart overhead)
6. **Day 2-3**: analyze CSV → go/no-go for M2.1
7. **Day 3+**: if GO, draft `PROPOSAL_iteration_M2.1_nvfp4_kv_pool.{en,zh}.md`

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| Surgical revert misses a v19 file that contributed to regression | Run full accuracy after revert; if ≠ v18 baseline, expand revert scope |
| SGLang has no upstream FP4 KV cache → must implement from scratch in M2.1 | Use M2.0-B fake-quant approach to validate gain potential before committing engineering effort |
| All 8 standard-attention layers are FP4-sensitive (NO-GO outcome) | M2 dies cheap (only M2.0 burned); pivot to other paths in catalog |
| Fcloud quota / cost during 8-layer ablation | Each layer needs only accuracy eval (~5 min); 8 × (5 min eval + ~3 min server restart) ≈ 65 min total; very cheap |
| Only 8 layers means small bandwidth-saving headroom | Even all 8 going FP4 only halves the **standard-attention** KV reads. Lightning layers (24 of 32) continue to use SimpleGLA state regardless. Need to estimate what fraction of decode KV-bandwidth is attributable to the 8 standard layers vs SimpleGLA state I/O before committing to M2.1 (this is part of M2.0 deliverable). |

---

## Memory log update needed

After execution:
- `/memories/soar_2026_leaderboard.md` — already updated 2026-04-24 snapshot
- `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` — append v18 baseline confirmation row,
  M2.0 ablation rows (**8 entries** — one per standard-attention layer)
- `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` — fix "NVFP4 Status:
  NOT VIABLE" wording to scope-it to weight quantization only

---

## Open questions for user (none blocking)

1. Should the v19 FLA kernel changes (proven beneficial) be **explicitly preserved** in
   the v18-revert plan, or is the user OK with starting from "pure" v18 source even if
   it costs ~1.7pt acc?
2. For M2.0, prefer Option A (true FP4 KV if upstream available) or Option B (fake-quant
   simulation, faster to implement)?
