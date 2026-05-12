# CHANGE_0157: Re-enable CUDA Graph for MEDUSA — Run TARGET_VERIFY Eagerly

**Date**: 2026-05-13  
**Commit**: `dc7300710`  
**Branch**: `mixed_minicpm_cudagraph`  
**Status**: Implemented, pending fcloud validation  
**Rule-compliance**: ✅ No scoring-rule violations  

---

## 1. Background and Motivation

Stage 3a Medusa (K=1 verify, commit `94f6ff6c6`) was validated with
`SOAR_SPEC_MEDUSA_EAGER=1` set in `prepare_env.sh`, which adds
`--disable-cuda-graph` to the server launch args and strips torch-compile.
This workaround was necessary because with `SOAR_SPEC_MEDUSA_EAGER=0`
(default production config) the server crashed during the first TARGET_VERIFY
forward with:

```
AttributeError: 'NgramVerifyInput' object has no attribute 'kv_indptr'
```

Goal of this change: **remove the `SOAR_SPEC_MEDUSA_EAGER=1` workaround** so
that Stage 3a runs under the standard production config
(GPTQ + FP8 KV + dense mode + cuda-graph + torch-compile).

---

## 2. Root-Cause Analysis

### 2.1 MEDUSA cuda-graph capture mode

`CudaGraphRunner.__init__()` (line ~282) sets:

```python
self.capture_forward_mode = ForwardMode.DECODE   # default
if (spec_algorithm.is_eagle()
        or spec_algorithm.is_standalone()
        or spec_algorithm.is_ngram()):
    self.capture_forward_mode = ForwardMode.TARGET_VERIFY
    self.num_tokens_per_bs = speculative_num_draft_tokens
```

Because `MEDUSA` is not in that guard, the MEDUSA graph runner captures
`DECODE` graphs only.

### 2.2 TARGET_VERIFY forward with DECODE graph

In `medusa_worker._forward_verify_k1()`:

1. `batch.spec_algorithm = NGRAM` (reuse NGRAM infra)
2. `batch.forward_mode = TARGET_VERIFY`
3. `batch.spec_info = NgramVerifyInput(draft_token_num=1, ...)`
4. → `target_worker.forward_batch_generation(model_worker_batch, is_verify=True)`
5. → `model_runner.forward(forward_batch)` → `_forward_raw()`

In `_forward_raw()`:

```python
mode_check = forward_batch.forward_mode.is_cuda_graph  # True for TARGET_VERIFY
can_run_graph = bool(
    mode_check()               # True
    and self.graph_runner      # exists
    and self.graph_runner.can_run(forward_batch)  # True (no mode check!)
)
# → graph_runner.replay() called
```

`can_run()` had **no check on `forward_mode`**, so it returned `True`.

`replay_prepare()` then called:

```python
attn_backend.init_forward_metadata_replay_cuda_graph(
    bs, ...,
    self.capture_forward_mode,   # DECODE
    forward_batch.spec_info,     # NgramVerifyInput
    ...
)
```

Inside the DECODE replay path:

```python
# indices_updater_decode.call_begin_forward()
else:
    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices  # CRASH
```

`NgramVerifyInput` does not have `kv_indptr` or `kv_indices` attributes — those
only exist on `EagleVerifyInput`.

### 2.3 Why NGRAM/EAGLE are unaffected

For NGRAM/EAGLE, `capture_forward_mode = TARGET_VERIFY`.  The cuda graph is
captured with TARGET_VERIFY metadata (prefill path with `generate_attn_arg_prefill`).
`replay_prepare()` uses the prefill path, which calls `generate_attn_arg_prefill()`
on the spec_info — no `kv_indptr` access.

---

## 3. Fix

### 3.1 `CudaGraphRunner.can_run()` — return False for MEDUSA + TARGET_VERIFY

**File**: `python/sglang/srt/model_executor/cuda_graph_runner.py`  
(and `benchmark/soar/demo_sala/sglang/python/…/cuda_graph_runner.py`)

Added after `is_ngram_supported`:

```python
# For MEDUSA: cuda graphs are captured with capture_forward_mode=DECODE.
# When the verify step runs (forward_mode=TARGET_VERIFY), the DECODE graph
# cannot handle NgramVerifyInput (it lacks kv_indptr/kv_indices attributes
# that the decode indices updater expects).  Return False here so that
# TARGET_VERIFY falls through to eager forward_extend(), which calls
# attn_backend.init_forward_metadata() → generate_attn_arg_prefill()
# via prefill_wrappers_verify — the correct code path.
# DECODE steps are unaffected (still use cuda graph as normal).
is_medusa_verify_ok = not (
    self.model_runner.spec_algorithm.is_medusa()
    and forward_batch.forward_mode.is_target_verify()
)

return (
    is_bs_supported
    and is_encoder_lens_supported
    and is_tbo_supported
    and capture_hidden_mode_matches
    and is_ngram_supported
    and is_medusa_verify_ok      # ← NEW
)
```

**Effect**: For MEDUSA + TARGET_VERIFY, `can_run() = False` → `_forward_raw()`
falls through to the eager dispatch. DECODE steps are unchanged.

### 3.2 `ModelRunner._forward_raw()` — route TARGET_VERIFY to `forward_extend()`

**File**: `python/sglang/srt/model_executor/model_runner.py`  
(and `benchmark/soar/demo_sala/sglang/python/…/model_runner.py`)

Changed the extend dispatch branch from:

```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
    ret = self.forward_extend(...)
```

to:

```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) or (
    forward_batch.forward_mode.is_target_verify()
    and self.spec_algorithm.is_medusa()
):
    # TARGET_VERIFY + MEDUSA: cuda graph is not captured for verify
    # (capture_forward_mode=DECODE for MEDUSA); forward_extend() handles
    # TARGET_VERIFY correctly via attn_backend.init_forward_metadata()
    # → prefill_wrappers_verify → generate_attn_arg_prefill().
    ret = self.forward_extend(...)
```

**Why `forward_extend()` is correct for TARGET_VERIFY**:

`forward_extend()` calls `self.attn_backend.init_forward_metadata(forward_batch)`.
In `FlashInferAttnBackend.init_forward_metadata()`, the `is_target_verify()` branch:

```python
elif forward_batch.forward_mode.is_target_verify():
    self.indices_updater_prefill.update(
        ...,
        prefill_wrappers=self.prefill_wrappers_verify,   # ← correct wrapper
        spec_info=forward_batch.spec_info,
    )
```

This calls `spec_info.generate_attn_arg_prefill()` on `NgramVerifyInput`, which
works correctly and sets up the verify attention without touching `kv_indptr`.

**NGRAM/EAGLE not affected**: They always use cuda graph (can_run() = True),
so they never fall through to this dispatch branch.

---

## 4. Call-flow Summary (after fix)

| Step | forward_mode | can_run() | path |
|------|-------------|-----------|------|
| Normal DECODE | DECODE | True | cuda graph replay (DECODE) |
| Prefill/EXTEND | EXTEND | False (not `is_cuda_graph`) | eager `forward_extend()` |
| Stage 3a VERIFY | TARGET_VERIFY | **False** (new: MEDUSA guard) | eager `forward_extend()` → `init_forward_metadata(TARGET_VERIFY)` → `prefill_wrappers_verify` → `generate_attn_arg_prefill()` |

---

## 5. Files Changed

| File | Lines changed | Note |
|------|--------------|-------|
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | +14 | `can_run()` MEDUSA guard |
| `python/sglang/srt/model_executor/model_runner.py` | +9 | `_forward_raw()` TARGET_VERIFY dispatch |
| `benchmark/soar/demo_sala/sglang/python/…/cuda_graph_runner.py` | +14 | submission copy sync |
| `benchmark/soar/demo_sala/sglang/python/…/model_runner.py` | +9 | submission copy sync |

---

## 6. Validation Commands

### 6.1 Server launch (default config, no SOAR_SPEC_MEDUSA_EAGER override)

```bash
# On fcloud — prepare_env.sh must have SOAR_SPEC_MEDUSA_EAGER=0 (default)
cd /root/submission_sim
source prepare_env.sh
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${SGLANG_SERVER_ARGS[@]}"
```

Expected: server boots successfully, cuda-graph capture completes (~14 min with
torch-compile).  No AttributeError during inference.

### 6.2 Quick accuracy check (MCQ subset)

```bash
cd /root/data
python3 eval_model_001.py \
  --url http://localhost:30000 \
  --data_path /root/data/perf_public_set.jsonl \
  --task_type mcq \
  --max_questions 20
```

Expected: ≥ 60% MCQ accuracy (≥ 12/20), consistent with Stage 3a GLA-fix
baseline (13/20 = 65% with EAGER=1).

### 6.3 Full accuracy + speed benchmarks

```bash
# Accuracy
python3 scripts/fcloud/fcloud_workflow.py accuracy

# Speed benchmarks
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

Expected:
- Normalized accuracy ≥ 99% (same as Stage 2 cgraph baseline)  
- S1/S8/Smax times roughly comparable to Stage 2 cgraph baseline
  (S1=118.28s, S8=43.87s, Smax=35.75s)
- (Verify runs eagerly, so per-token overhead is slightly higher than baseline
  but DECODE uses cuda graph as before)

---

## 7. Result Summary (to be filled after fcloud test)

| Config | Commit | MCQ(20) | Norm Acc | S1 (s) | S8 (s) | Smax (s) |
|--------|--------|---------|----------|---------|---------|----------|
| Stage2 cgraph baseline | 46553947b | — | 80.11% | 118.28 | 43.87 | 35.75 |
| Stage3a GLA fix (EAGER=1) | 94f6ff6c6 | 65% | — | — | — | — |
| **Stage3a cgraph (EAGER=0)** | **dc7300710** | TBD | TBD | TBD | TBD | TBD |

---

## 8. Rollback Instructions

To revert to CHANGE_0155 state (Stage 3a with EAGER=1):

```bash
# Option A: git revert
git revert dc7300710

# Option B: re-set SOAR_SPEC_MEDUSA_EAGER=1 in prepare_env.sh (quick workaround)
# In benchmark/soar/demo_sala/prepare_env.sh, change:
#   SOAR_SPEC_MEDUSA_EAGER=0   →   SOAR_SPEC_MEDUSA_EAGER=1
```

---

## 9. Next Steps

1. **Fcloud validation**: Start instance, sync, restart server with default
   `SOAR_SPEC_MEDUSA_EAGER=0`, run accuracy + speed benchmarks.
2. **Stage 3b**: Train real Medusa heads (K=1, W1≠0) for actual draft acceptance
   speedup.  Partial-accept snapshot/restore path in `medusa_worker.py` is already
   scaffolded.
3. **K>1 extension**: After trained heads are validated, extend to K=2+ for
   higher acceptance rates.
