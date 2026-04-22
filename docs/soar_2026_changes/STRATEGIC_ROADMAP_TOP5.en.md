# Strategic Roadmap to Top 5 / Top 3 — SOAR 2026 (MiniCPM-SALA)

**Date**: 2026-04-23  
**Current standing**: team-beta rank #21, score **39.62**  
**Gap to #5**: ≥ 79.55 / 39.62 ≈ **2.01× performance score needed**  
**Working baseline**: GPTQ + FP8 KV (e5m2) + dense + torch.compile(max-bs=8) + mixed-chunk + v18 scheduling (chunk=32K, prefill-max-req=1, sched-cons=1.0, running=24)

---

## 1. Evidence summary from Tests 29–33 (new fcloud SM120)

| Finding | Evidence | Implication |
|---|---|---|
| **Accuracy floor ≈77–79% on new fcloud** | Tests 29/30/32/33: no single knob moves the mean; mcq flaps 40–96% | Chasing accuracy via config is futile; need structural change |
| **Per-task variance ±20–40 pts** | Test 29 mcq=96 vs Test 30 mcq=53 with one flag flip | 150-sample concurrency=32 eval is intrinsically noisy |
| **torch.compile ≈0% speed gain, +3 min boot** | Test 33 vs 29: 3016s vs 3005s duration; boot 36s vs 219s | Boot cost is real in ≤5h quant+eval budget |
| **KV dtype (e5m2 vs e4m3) swings per-task, not overall** | Test 30: cwe+25, fwe+14, mcq−43, overall −0.77 | Precision isn't the dominant variance source |
| **Scheduling (v18 vs v19) doesn't fix accuracy** | Test 32 v18-sched gave 75.73% (C=0) | Scheduling aggression is not the source of failures |
| **Scoring is multiplicative** | `Final = (S1×0.4 + S8×0.3 + Smax×0.3) × C` | 10% speed gain = +10 score points; 1 C-tier (0.96→1.0) = +4.2% |

### Current best results recorded

| Metric | Best run | Config | Value |
|---|---|---|---|
| Accuracy | Test 20 | GPTQ + FP8 KV + dense + torch.compile(bs=8) + mixed-chunk + max-running-req=24 | acc_ori=**80.64%**, C=1.0 |
| S1 | Test 25A-spd | prefill-max-req=4, sched-cons=0.8, chunk=32K | **110.58s** |
| S8 | Test 24-spd | dense-calibrated (acc-broken) | **40.45s** |
| Smax | Test 20-spd | mixed-chunk + max-running-req=24 | **34.15s** |
| On current fcloud SM120 | Test 29 | CHANGE_0130 reverted baseline | acc=78.73%, C=0.96 |

---

## 2. Accuracy vs Speed — Which should be the next goal?

**Verdict: speed is the primary lever. Accuracy is conditional.**

### Why speed wins

1. **Diminishing returns on accuracy**  
   C=0.96 → C=1.0 = multiplicative bonus of only 1.042×. Our current performance score is already multiplied by 0.96; moving to 1.0 gives +4.2%. A 10% speed improvement gives +10% directly — **2.4× larger impact**.

2. **Gap analysis is multiplicative**  
   From 39.62 → 79.55 (top 5) = need **2.01× more score**. Even if we raise C from 0.96 → 1.0 (1.042×), we'd only reach ~41.3. To reach 79.55 we must **roughly double performance score**. That only comes from faster kernels or speculative decoding.

3. **Accuracy sits at a model-capability wall**  
   Tests 29–33 show no config knob reliably lifts acc above ~79%. The 150-sample eval with concurrency=32 has ~±2–3 pt noise floor from batch-interleaving + thinking-chain length nondeterminism. Reaching C=1.0 consistently (norm≥99%) requires acc_ori ≥ 79.2% — we're already right at the edge.

### When to pivot to accuracy as primary

Accuracy becomes the primary goal **only if**:
- **v18 official resubmission** (user's current pending test) lands at acc_ori < 77%, which would give C=0 (elimination). In that case, we must stabilize accuracy before optimizing speed.
- Or if we find a specific structural bug (e.g., mcq runaway chains) that can be fixed with a targeted code change (not config).

---

## 3. Three-iteration roadmap to Top 5

### Iteration A — Marlin GPTQ SM120 decode tile specialization (HIGH impact)

**Target**: +5–15% decode TPS (S1 and S8 reduce by 5–15%)  
**Why**: Current decode TPS=439 (Test 29). SM120 BF16/FP16 peak = 148 TFLOPS. At W4A16 small-M (batch 1–8) the Marlin GEMM underutilizes tensor cores. Extended tile table (CHANGE_0125) compiles new shapes but scorer rarely picks them. Need dispatch logic that *actively prefers* small-M SM120 tiles at decode time.

**Action plan**:
1. Profile Marlin invocations during decode on SM120 via `ncu` / PyTorch profiler to identify tile-selection misses
2. Add explicit (M, N, K, num_threads) entries in `gptq_marlin.cu` tile table for M∈{1, 2, 4, 8} × N∈{4096, 7168, 13824, …} (MiniCPM-SALA MLP/QKV shapes)
3. Modify scorer (`heuristic.cc` or equivalent) to give SM120 decode tiles a scoring bonus when M ≤ 16
4. Rebuild wheel (~4h first / ~3min incremental), run speed benchmark to confirm

**Risk**: kernel build time; accuracy neutral (tile choice = same math). Rollback = revert commit.  
**Effort**: 1–2 iterations. Reference: vLLM M100 / TensorRT-LLM SM120 W4A16 kernel dispatch.  
**Docs target**: `CHANGE_0140_sm120_decode_marlin_tiles.{en,zh}.md`

---

### Iteration B — Runtime tuning + drop torch.compile for boot savings (MEDIUM impact)

**Target**: save 3 min per server restart (matters in 5h official budget if they restart mid-eval), no runtime regression  
**Why**: Test 33 proved torch.compile gives ~0% speed on this workload but adds 3 min warmup. Under official 5h ceiling (quantize + eval), every minute counts. CUDA graphs (automatic via sglang) cover decode paths already.

**Action plan**:
1. Remove `--enable-torch-compile --torch-compile-max-bs 8` from prepare_env.sh
2. Verify `--cuda-graph-max-bs` (or equivalent) is covering decode batches
3. Re-measure speed — confirm no regression
4. Apply after Iteration A to avoid tangling variables

**Risk**: very low.  
**Effort**: 1 test run.  
**Bundle**: into v21 submission combined with Iteration A.

---

### Iteration C — Speculative decoding (HIGH upside, HIGH risk)

**Target**: +30–100% decode throughput on repetitive tasks  
**Why**: Prior EAGLE3 attempt (CHANGE_0090) failed because draft was untrained (accept_rate=0.26). A working speculative path would provide the multiplicative speedup needed to reach #5.

**Two options**:

- **C1 — n-gram speculative decoding (LOW effort)**  
  Built into sglang. No training required. Matches token patterns from prompt/prior generation. Best on NIAH/FWE which have repetitive structure.  
  Expected: +10–30% on NIAH/FWE specifically.  
  Effort: 1 iteration (flag + eval).

- **C2 — Trained EAGLE draft (HIGH effort, HIGH reward)**  
  Train a 1-layer EAGLE draft head on provided calibration data (~1–2h GPU time on SM120). Target accept_rate ≥ 0.6.  
  Expected: +50–100% decode TPS.  
  Effort: multi-iteration, requires training infra and correctness safeguards.

**Risk**: accept_rate can tank → slower than baseline; accuracy must stay > 97% normalized.  
**Sequencing**: C1 first as a cheap probe; C2 only if we still have headroom after A+B.

---

## 4. Recommended sequencing (today)

1. **Test 34a — reproduce Test 20 baseline** on new fcloud SM120 (existing GPTQ model). Confirm we can still hit ~80% acc + C=0.96/1.0 reliably. Also run a speed benchmark to establish current-fcloud baseline numbers (S1/S8/Smax).
2. **Test 34b — fresh on-site re-quantization** via `prepare_model.sh` → produces a new GPTQ model calibrated against current CUDA/Triton/kernel stack on SM120. Compare accuracy + speed vs 34a.
3. **Decision gate**:  
   - If 34a or 34b ≥ 79% acc_ori → lock that as v20 submission candidate, proceed to **Iteration A** (kernel work).  
   - If both < 78% → accuracy is unstable; investigate mcq runaway structurally (sample capping via custom `eos_token_id` tuning, or decode-side `logit_bias` for termination tokens).
4. **Await v18 official resubmit result** (running in parallel) → calibrates local-vs-official gap. If v18 lands at C=0.96 or C=1.0, we know our local runs are reliable and can submit v20 confidently.

---

## 5. Decision triggers

| Trigger | Action |
|---|---|
| v18 official lands C=1.0 (≥99% norm) | Resubmit v20 = Test 34a config; start Iteration A immediately |
| v18 official lands C=0.96 | Same as above, but also schedule Iteration B (boot savings) |
| v18 official lands C=0.92 | Investigate what changed since original v18 (server code drift); pause aggressive speed work until stable |
| v18 official lands C=0 | **Accuracy becomes primary goal**. Investigate structural mcq fix. |
| Iteration A delivers ≥10% speed | Submit v21 immediately. Expect rank jump toward #10–#15. |
| Iteration A+B+C1 combined ≥20% speed with C≥0.96 | Realistic path to Top 10; stretch goal Top 5 requires Iteration C2. |

---

## 6. Rollback safety

Every iteration commits to `mixed_minicpm_cudagraph` branch. To roll back any change:
```bash
git log --oneline -20
git revert <commit>
# or:
git reset --hard <last_known_good>
git push minicpm-src mixed_minicpm_cudagraph --force-with-lease
```
For kernel changes, the pre-built wheel under `/root/submission_sim/sgl_kernel-*.whl` is the recovery artifact.
