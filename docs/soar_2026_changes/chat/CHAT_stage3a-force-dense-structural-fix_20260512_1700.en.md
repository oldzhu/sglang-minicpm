# CHAT LOG: Stage 3a Force-Dense Structural Fix Validation
**Topic**: `stage3a-force-dense-structural-fix`  
**Date**: 2026-05-12  
**Participants**: User (oldzhu), Agent (GitHub Copilot)  
**fcloud Instance**: ai-e7e98a7c52

---

## 1. Session Opens — Collecting Accuracy Test Results

**Context**: Previous session launched accuracy eval with `SOAR_BACKEND_KEEP_FORCE_DENSE=1`
(forced `--force-dense-minicpm` active) and timed out at 145/150 requests. The eval
had 2 HTTP timeout failures at requests 127-128 (long NIAH tasks). fcloud was left running.

**Agent action**: Found the output directory `/root/data/outputs/20260512_083652/` empty
(the eval process crashed at 3600s before writing `summary.json`). Detected eval process
had already exited. Re-ran directly with 2-hour timeout:

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/data && python3 eval_model_001.py --api_base http://127.0.0.1:30000 \
   --model_path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
   --data_path /root/data/perf_public_set.jsonl --concurrency 32 2>&1' \
  --timeout 7200
```

**Accuracy result**:
```
Average Score: 77.87%
Total Duration: 3338.76s
Total Tokens: In=8,644,406  Out=1,235,925

Per-task:
  cwe:  86.00%  (avg_out=24,453)
  fwe:  100.00% (avg_out=7,287)
  mcq:  46.67%  (avg_out=8,997)  ← runaway thinking
  niah: 100.00% (avg_out=352)
  qa:   56.67%  (avg_out=108)
```

---

## 2. Speed Tests (Force-Dense Active)

Server still running with `SOAR_BACKEND_KEEP_FORCE_DENSE=1`.

**Results**:
| Variant | Duration | vs Stage3a no-force-dense |
|---------|----------|--------------------------|
| S1 | 202.70s | 202.96s (−0.1%) |
| S8 | 61.60s | 61.65s (−0.1%) |
| Smax | 43.29s | 43.40s (−0.3%) |

**Conclusion**: Zero measurable speed impact from adding `--force-dense-minicpm`.

---

## 3. fcloud Paused

```bash
python3 scripts/fcloud/fcloud_workflow.py pause-instance
# → HTTP 200 "任务已暂停"
```
(First attempt: HTTP 504, retry succeeded.)

---

## 4. Analysis and Decisions

### 4.1 Why Force-Dense Has No Speed Impact

`--force-dense-minicpm` controls only pool type selection at init time:
- False: `MiniCPMHybridReqToTokenPool` (allocates K1/K2 tables)
- True: `HybridReqToTokenPool` (no K1/K2)

Neither pool type affects the forward pass computation or KV cache access patterns
under flashinfer backend. The GPU compute kernel is identical either way.

### 4.2 Structural Fix Decision

**Outcome**: Make `SOAR_BACKEND_KEEP_FORCE_DENSE=1` the default in `prepare_env.sh`.

Rationale:
1. Zero speed/accuracy cost (verified)
2. Eliminates entire K1/K2 crash class — CHANGE_0158/0161 patches become unreachable
3. Cleaner than patch-level fixes; no runtime branches in hot path
4. Matches Stage 2 pass-through behavior (which already uses force-dense)

### 4.3 Accuracy Analysis

| Config | Accuracy | Norm | C |
|--------|----------|------|---|
| Stage 2 (no MEDUSA) | 80.11% | ~100% | 1.0 |
| Stage 3a (CHANGE_0161, no force-dense) | 76.04% | ~95% | 0 |
| **Stage 3a (CHANGE_0162, force-dense)** | **77.87%** | **~97.34%** | **0.92** |

C=0.92 penalty is due to mcq=46.67% (norm ≈ 97.34% < 98% → C=0.92 not C=1.0).
The mcq runaway thinking (avg_out=8,997 tokens) is a pre-existing Stage 3a issue
affecting both force-dense and no-force-dense configurations.

**Stage 3a is NOT a submission candidate** regardless of force-dense:
- S1=202.70s vs Stage 2 baseline 118.28s (+72% slower) — zero-init heads rejected every draft
- C=0.92 due to mcq runaway — 8% performance penalty

---

## 5. Changes Applied

### `benchmark/soar/demo_sala/prepare_env.sh`
- Added `export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"` default
- Placed immediately after `export SOAR_BACKEND_VARIANT="${SOAR_BACKEND_VARIANT:-flashinfer}"`
- No logic changes; existing conditional already handles this correctly

### `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`
- Added **Stage3a-force-dense** row to main table
- Records: commit 50e9466d0, accuracy 77.87% (norm 97.34%, C=0.92), S1/S8/Smax

### New docs created:
- `CHANGE_0162_force_dense_default_structural_fix.en.md`
- `CHANGE_0162_force_dense_default_structural_fix.zh.md`

---

## 6. Next Steps (Open)

### Immediate: Stage 3b Trained Medusa Heads

Stage 3a established the infrastructure. Stage 3b requires:
1. **Training data collection**: run base model on eval prompts, capture hidden states + labels
2. **Head training**: 1 ResBlock MedusaHead (hidden=4096 → vocab=122753), freeze base, 5 epochs
3. **Wiring**: replace zero-init draft in `MedusaWorker._forward_generate_k1` with head forward
4. **Expected gain**: accept_rate ~60-70% → spec_accept_length ~1.6-1.7 → ~20-30% speedup
5. **Size**: K=1 head ≈ 1 GB BF16 — within 2 GB submission limit

### mcq Accuracy Investigation

mcq drops ~10-15pt with MEDUSA (47% vs Stage 2's ~57%). Potential causes:
- MEDUSA verify overhead changes effective token budget
- GLA state interaction with thinking-format MCQ prompts
- max_tokens counting difference in spec decoding loop

Worth investigating before Stage 3b to ensure accuracy doesn't regress further
with real acceptance.

---

## 7. Cross-References

| Item | Link |
|------|------|
| CHANGE_0162 EN | [CHANGE_0162_force_dense_default_structural_fix.en.md](CHANGE_0162_force_dense_default_structural_fix.en.md) |
| CHANGE_0162 ZH | [CHANGE_0162_force_dense_default_structural_fix.zh.md](CHANGE_0162_force_dense_default_structural_fix.zh.md) |
| Test results | TEST_RESULTS_TRACKING.md row "Stage3a-force-dense" |
| Prior session chat | CHAT_stage3a-k1-crash-k1k2-fix_20260512_XXXX.en.md |
| CHANGE_0161 | CHANGE_0161_medusa_stale_k1k2_zero_out.en.md |
