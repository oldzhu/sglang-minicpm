# CHANGE_0125: SM120 Marlin GEMM Tile Instantiations

## Background

Profiling (CHANGE_0120) showed GEMM = 85.3% of prefill time. The Marlin GPTQ kernel in `sgl-kernel` has SM120-specific auto-config that scores 5 thread configs by occupancy, fill ratio, and wave coverage. However, the **top-priority configs were never usable** because their kernel template specializations were not instantiated in `get_marlin_kernel()`.

## Rule Compliance

- **Accuracy**: Zero impact — same Marlin kernel code, different tile scheduling parameters
- **Submission size**: Negligible binary size increase (kernel specializations add ~KB)
- **Reproducibility**: Deterministic auto-config scoring selects tiles

## Missing Instantiations (Before)

| SM120 Priority | Config (K,N,T) | K_blocks | N_blocks | M1 (decode) | M234 (prefill) |
|----------------|----------------|----------|----------|-------------|----------------|
| 1 | `{128,256,256}` | 8 | 16 | **MISSING** | **MISSING** |
| 2 | `{64,256,256}` | 4 | 16 | **MISSING** | Available |
| 3 | `{128,128,256}` | 8 | 8 | Available | **MISSING** |
| 4 | `{64,128,128}` | 4 | 8 | Available | Available |
| 5 | `{128,64,128}` | 8 | 4 | Available | Available |

Decode fell to priority #3. Prefill fell to priority #2. The top-priority `{128,256,256}` was unreachable.

## Changes

**File**: `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`

### 1. Added kernel template instantiations

Added 5 new lines to `COMMON_GET_IF` macro:
```cpp
// Before (6 lines):
COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)
COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)
COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)
COMMON_GET_IF_M234(W_TYPE, 16, 4, 256)
COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)
COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)

// After (11 lines):
COMMON_GET_IF_M1(W_TYPE, 16, 8, 256)   // NEW: SM120 priority #1
COMMON_GET_IF_M1(W_TYPE, 16, 4, 256)   // NEW: SM120 priority #2 for decode
COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)
COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)
COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)
COMMON_GET_IF_M234(W_TYPE, 16, 8, 256) // NEW: SM120 priority #1 for prefill
COMMON_GET_IF_M234(W_TYPE, 16, 4, 256)
COMMON_GET_IF_M234(W_TYPE, 8, 8, 256)  // NEW: SM120 priority #3 for prefill
COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)
COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)
```

### 2. Enhanced SM120 auto-config logging

Changed `log_sm120_exec_config_once` to log each unique `(N, K, thread_m_blocks)` combination (up to 20 entries) instead of only the first config ever. This shows which tile config is selected for each linear layer type in the model.

## Validation

1. **Build**: Rebuild `sgl-kernel` on fcloud (CUDA compilation with new template specializations)
2. **Server logs**: Check `[sgl-kernel] SM120 Marlin auto-config:` lines — should show `thread_n=256 thread_k=128` for compatible layers
3. **Speed benchmark**: Compare S1/S8/Smax against Test 25 baseline (S1=120.58s)
4. **Accuracy**: Quick accuracy check to confirm no regression
5. **Re-profile**: Torch profiler to verify GEMM time reduction

## Rollback

Revert commit 338989afe:
```bash
git revert 338989afe
```
