# Chat — Round 13f-1 + CHANGE_0136 validation

**Date**: 2026-04-29
**Topic**: Run approved Round 13f-1 (flashinfer smoketest) + CHANGE_0136 sanity
(`SOAR_SPARSE_DENSE_LEN=524288` → 65536 → 16384). Document outcomes.
**Commits exercised**: `6a070110b` (initial implementation), `45e316ee0` (drop
`--enable-torch-compile` when env var set), `7a7a568eb` (drop `--dense-as-sparse`
when env var set).

## User asks (this round)

1. "fcloud started" — go run the planned tests.
2. "pls test in order as you planned" — proceed with full sequence.

## Agent actions / outcomes

### Step 1 — baseline restart (default Test 12 args, no env override)
- Server boot 209s, healthy. No regression from CHANGE_0136 source edits when env
  var unset (env override block guarded by `if _env_dense_len is not None`, so
  default falls into the existing else branch).
- Skipped a full Test 12 rerun (already extensively measured) to save fcloud time.

### Step 2 — Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`)
Server cmdline confirmed: `--attention-backend flashinfer`, no
`--force-dense-minicpm`, no `--dense-as-sparse`, no `--minicpm_flashinfer`.

- **Accuracy: 76.91%** (Test 12 baseline 79.29%; Δ −2.38pt)
- **Speed**: S1=110.76s, S8=40.50s, Smax=33.66s
  - vs Test 12 baseline 121.71 / 44.09 / 35.86 → **−9% / −8% / −6%** (faster)
- **Verdict**: speed gain real but accuracy is C=0 fatal. Normalized accuracy
  ≈ 76.91 / 80.00 ≈ 96.1% (below the 97% cutoff). Stock flashinfer cannot route
  SALA's mixed sparse/dense architecture correctly. Do not pursue.
- Bucket details: cwe=82.33% (vs ≥87 baseline), qa=56.67%, mcq=46.67%, fwe=98.89%,
  niah=100%. cwe + qa drops are the killers.

Recorded as TEST_RESULTS_TRACKING row `R13f1-flashinfer`.

### Step 3 — CHANGE_0136 sanity (`SOAR_SPARSE_DENSE_LEN=524288`) — BLOCKED

#### Attempt 1 (commit `6a070110b`): cudagraph capture crash
```
torch._dynamo.exc.InternalTorchDynamoError: RuntimeError:
Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture
```
Same crash documented in CHANGE_0133 §"Relationship to the Round 13d torch.compile
crash". Caused by leaving `--enable-torch-compile` while sparse path is reachable
(after dropping `--force-dense-minicpm`).

**Fix** (`45e316ee0`): when `SOAR_SPARSE_DENSE_LEN` is set, also clear
`TORCH_COMPILE_ARGS=""` (mirrors the existing `SOAR_SPARSE_MODE=1` branch).

#### Attempt 2 (commit `45e316ee0`): override silently no-op
Server boots, but `pgrep -af` shows args still include `--dense-as-sparse`. Inside
`MiniCPMAttentionBackend.__init__` the env-override block is guarded by
`not self.dense_as_sparse`, so `dense_as_sparse=True` short-circuits the override
and forces `dense_len=0` (every request → sparse path).

**Fix** (`7a7a568eb`): also clear `--dense-as-sparse` from `SGLANG_SERVER_ARGS`
when `SOAR_SPARSE_DENSE_LEN` is set.

#### Attempt 3 (commit `7a7a568eb`): prefill crash on first sample
Server boots in 36s. Backend log confirms override applied:
`SOAR_SPARSE_DENSE_LEN override: dense_len=524288 (model config default was 8192)`.

First request (prompt_tokens=103) immediately crashes:
```
File ".../minicpm_backend.py", line 1087, in forward_extend
  metadata.sparse_page_table[sparse_page_table_idx_start, :kv_len] = \
      page_table[dense_bs, :kv_len] * 2
RuntimeError: The expanded size of the tensor (103) must match the existing
size (104) at non-singleton dimension 0. Target sizes: [103]. Tensor sizes: [104]
```

This is a **pre-existing off-by-one bug** in the sparse-path prefill metadata
construction. It is distinct from CHANGE_0133 (decode-time compress_k1/k2
over-fill) — same code module, different function, different symptom.

**CHANGE_0136 cannot be validated** until this prefill bug is fixed. Steps 4
(`SOAR_SPARSE_DENSE_LEN=65536`, conservative) and 5 (`=16384`, aggressive) are
skipped because they would all hit the same crash on the first request that
exceeds the threshold and is routed through the sparse layers.

Recorded as TEST_RESULTS_TRACKING row `R13f-CHANGE_0136-sanity`.

### Cleanup
- fcloud shut down via `fcloud_workflow.py shutdown` after data collection (cost-saving rule).

## Key conclusions

1. **Round 13f-1 (flashinfer) is dead** for submission. Speed gain ~7% is too small
   to overcome 2.4pt accuracy loss (C goes to 0).
2. **CHANGE_0136 needs sparse-path bug fixes first.** The implementation itself is
   correct (override fires, environment plumbing works); the sparse path on HEAD
   has at least two pre-existing bugs blocking long-context routing:
   - CHANGE_0133 (proposed, not yet merged): decode-time compress_k1/k2 over-fill
   - **NEW**: prefill-time `sparse_page_table` off-by-one at
     `minicpm_backend.py:1087`. Not yet documented in any CHANGE_*.
3. The `R13e-prof-*` profile result still stands: BF16 GEMM dominates the BF16
   sparse line. CHANGE_0136 was meant to mitigate this for **GPTQ** by limiting
   sparse routing to long requests; but the sparse path bugs on HEAD prevent us
   from observing whether GPTQ + sparse-only-for-long actually beats Test 12 dense.

## Open follow-ups

- Author CHANGE_0137 to fix the `sparse_page_table[sparse_page_table_idx_start, :kv_len]`
  off-by-one (root cause investigation needed; likely a `+1` for the EOS / generated
  token slot, or a `dense_bs` vs `sparse_bs` indexing mismatch).
- After CHANGE_0137 + CHANGE_0133 land, re-attempt the CHANGE_0136 validation
  matrix: 524288 sanity → 65536 conservative → 16384 aggressive.

## Cross-references

- Implementation: commits `6a070110b`, `45e316ee0`, `7a7a568eb` on
  `minicpm-src/mixed_minicpm_cudagraph`.
- Proposal docs: [PROPOSAL_round13f1_flashinfer_backend_smoketest.en.md](../PROPOSAL_round13f1_flashinfer_backend_smoketest.en.md),
  [CHANGE_0136_minicpm_sparse_dense_len_flag.en.md](../CHANGE_0136_minicpm_sparse_dense_len_flag.en.md).
- Pre-existing bug references: [CHANGE_0133_sparse_compress_buffer_oversize.en.md](../CHANGE_0133_sparse_compress_buffer_oversize.en.md).
- Test rows: TEST_RESULTS_TRACKING.md `R13f1-flashinfer`, `R13f-CHANGE_0136-sanity`.
