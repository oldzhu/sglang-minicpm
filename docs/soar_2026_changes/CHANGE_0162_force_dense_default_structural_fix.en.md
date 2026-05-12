# CHANGE_0162 — Force-Dense Default: Structural Fix for K1/K2 Pool Crash Class

**Type**: Structural / Configuration  
**Status**: VALIDATED — 2026-05-12  
**Priority**: High (blocks MEDUSA crash class, zero cost)  
**Related**: CHANGE_0158, CHANGE_0159, CHANGE_0161 (patch-level fixes for same root cause)

---

## 1. Background and Motivation

MEDUSA Stage 3a (K=1 zero-init speculative decoding) was crashing with
`available_size > max_total_num_tokens` during speed benchmarks. The root cause
traced to `req_to_sparse_k1_token` / `req_to_sparse_k2_token` arrays holding
stale non-zero slot IDs from a previous longer request. CHANGE_0161 fixed this
at the patch level (zero-out K1/K2 rows in `cache_finished_req` after freeing).

However, a deeper architectural question arose: **why are K1/K2 tables allocated
at all when using the flashinfer backend, which never uses them?**

### Pool Type Selection Path

```python
# python/sglang/srt/model_config.py:238
def has_sparse_attention(self):
    return getattr(self.hf_config, "has_sparse_attention", False) \
           if not self.force_dense_minicpm else False

# python/sglang/srt/model_runner_kv_cache_mixin.py:369
if self.minicpm_hybrid_config is not None:
    if self.model_config.has_sparse_attention:
        self.req_to_token_pool = MiniCPMHybridReqToTokenPool(...)  # has K1/K2 tables
    else:
        self.req_to_token_pool = HybridReqToTokenPool(...)          # no K1/K2 tables
```

When `--force-dense-minicpm` is **absent**:
- `force_dense_minicpm=False` → `has_sparse_attention=True` (from hf_config)
- Pool type: `MiniCPMHybridReqToTokenPool` — allocates `req_to_sparse_k1_token` + `req_to_sparse_k2_token`
- K1/K2 tables **are allocated** even though flashinfer never writes them

When `--force-dense-minicpm` is **present**:
- `force_dense_minicpm=True` → `has_sparse_attention=False`
- Pool type: `HybridReqToTokenPool` — no K1/K2 tables allocated
- `cache_finished_req`'s `isinstance(pool, MiniCPMReqToTokenPool | MiniCPMHybridReqToTokenPool)` → `False`
- K1/K2 branch **never entered** → crash class structurally impossible

### Why force-dense was absent

When `SOAR_BACKEND_VARIANT=flashinfer` was adopted in v20 (Round 13f), the
`prepare_env.sh` flashinfer branch cleared `FORCE_DENSE_ARG=""` by default.
The rationale was: "stock flashinfer doesn't use the minicpm backend, so
force-dense is irrelevant." The side effect on pool type selection
(`has_sparse_attention` → pool allocation → K1/K2) was overlooked.

---

## 2. Rule-Compliance Statement

- No model weights changed
- No accuracy-affecting code changed
- Only `prepare_env.sh` default env var changed (`SOAR_BACKEND_KEEP_FORCE_DENSE=1`)
- The `--force-dense-minicpm` flag was already used in the Stage 2 pass-through
  submission and production baseline — this restores that behaviour
- Rollback: set `SOAR_BACKEND_KEEP_FORCE_DENSE=0`

---

## 3. Implementation Plan (before change)

1. Add `export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"` to `prepare_env.sh`
2. The existing conditional in the flashinfer branch already handles this:
   ```bash
   if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
       # Keep FORCE_DENSE_ARG as set above (" --force-dense-minicpm")
       :
   else
       FORCE_DENSE_ARG=""
   fi
   ```
   With default `1`, `FORCE_DENSE_ARG` is preserved.

---

## 4. Actual Code Changes

### `benchmark/soar/demo_sala/prepare_env.sh`

Added `SOAR_BACKEND_KEEP_FORCE_DENSE` default export (18 new lines with comment block):

```bash
# CHANGE_0162 (2026-05-12): default SOAR_BACKEND_KEEP_FORCE_DENSE=1 so that
# --force-dense-minicpm is always active with the flashinfer backend.
# Structural fix: --force-dense-minicpm → model_config.has_sparse_attention=False
# → HybridReqToTokenPool (no req_to_sparse_k1_token) → K1/K2 crash class
# structurally impossible. Verified 2026-05-12 (Stage3a-force-dense test):
#   - Zero crashes on accuracy (150 req) + speed (S1/S8/Smax) runs
#   - Zero speed impact vs no-force-dense (202.70/61.60/43.29s vs 202.96/61.65/43.40s)
#   - Accuracy: 77.87% (norm≈97.34%, C=0.92) — within Stage 3a noise band
# Previously this was off by default when SOAR_BACKEND_VARIANT=flashinfer,
# causing MiniCPMHybridReqToTokenPool to be allocated even though flashinfer
# never uses K1/K2 sparse tables. Set SOAR_BACKEND_KEEP_FORCE_DENSE=0 to
# roll back to old behaviour for A/B testing.
export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"
```

The existing flashinfer branch logic (introduced in Round 13f-2/13f-3) already
handles this correctly — no other code changes required.

---

## 5. Validation Commands

```bash
# 1. Verify prepare_env.sh emits --force-dense-minicpm in SGLANG_SERVER_ARGS
source benchmark/soar/demo_sala/prepare_env.sh 2>&1 | grep "SGLANG_SERVER_ARGS"
# Expected: contains "--force-dense-minicpm"

# 2. Verify server boots with force_dense_minicpm=True
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec 'grep -i "force_dense" /tmp/sglang_server.log | head -3'

# 3. Accuracy test
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 4. Speed tests
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# 5. Rollback test
SOAR_BACKEND_KEEP_FORCE_DENSE=0 source benchmark/soar/demo_sala/prepare_env.sh 2>&1 | grep "SGLANG_SERVER_ARGS"
# Expected: does NOT contain "--force-dense-minicpm" (old flashinfer behaviour)
```

---

## 6. Result Summary

| Metric | Stage3a (no force-dense) | Stage3a-force-dense | Change |
|--------|--------------------------|---------------------|--------|
| Crashes | 0 (after CHANGE_0161) | **0** | — |
| K1/K2 allocated | Yes (MiniCPMHybridReqToTokenPool) | **No** (HybridReqToTokenPool) | Structural |
| S1 | 202.96s | **202.70s** | −0.1% (noise) |
| S8 | 61.65s | **61.60s** | −0.1% (noise) |
| Smax | 43.40s | **43.29s** | −0.3% (noise) |
| Accuracy (orig) | 76.04% | **77.87%** | +1.83pt (noise band) |
| Accuracy (norm) | ~95.05% | **~97.34%** | C=0 → C=0.92 |
| C | 0 | **0.92** | Improved |

**Key finding**: Making `--force-dense-minicpm` the default when using flashinfer
backend has **zero measurable speed impact** and **eliminates the entire K1/K2
crash class** by preventing the tables from being allocated in the first place.

> Note: Stage 3a speeds (202/61/43s) are still slower than the Stage 2 baseline
> (118/43/35s). This is expected — K=1 zero-init Medusa heads have accept_rate=0
> (every draft rejected). Stage 3b (trained heads) is required for speedup.

---

## 7. Relationship to CHANGE_0158/0159/0161

| Fix | Type | When active |
|-----|------|-------------|
| CHANGE_0158 | Patch: `ne(0)` filter for K1/K2 free | `MiniCPMHybridReqToTokenPool` present |
| CHANGE_0159 | Patch: `ne(0)` filter for main KV free | Always |
| CHANGE_0161 | Patch: zero-out stale K1/K2 rows after free | `MiniCPMHybridReqToTokenPool` present |
| **CHANGE_0162** | **Structural: prevent K1/K2 allocation entirely** | flashinfer backend |

With CHANGE_0162 active, CHANGE_0158 and CHANGE_0161 are dead code (the
`isinstance(pool, MiniCPMReqToTokenPool | MiniCPMHybridReqToTokenPool)` branch
never fires). CHANGE_0159 (main KV zero-filter) remains active and useful.

---

## 8. Rollback Instructions

```bash
# Option 1: env override (test only, not persistent)
export SOAR_BACKEND_KEEP_FORCE_DENSE=0

# Option 2: revert the default in prepare_env.sh
# Change: export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"
# To:     export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-0}"
```

---

## 9. Next Steps

1. **Stage 3b: Trained Medusa heads** — required to achieve actual speedup vs
   Stage 2 baseline. With K=1 zero-init heads, accept_rate=0 and Stage 3a is
   ~70% slower than Stage 2. Trained heads targeting accept_rate≥0.60 would give
   `spec_accept_length≈1.60` → ~20-30% speedup vs non-spec.
2. **mcq accuracy**: Both Stage 3a configurations show mcq≈43-47%, down from
   Stage 2's ~57%. Root cause under investigation — likely interaction between
   MEDUSA verify overhead and max_tokens counting on long thinking chains.
3. **Official submission**: Stage 3a with CHANGE_0162 is still not a viable
   submission candidate (speeds slower than Stage 2 + C=0.92). Submit Stage 2
   pass-through (commit `46553947b`) or proceed to Stage 3b.
