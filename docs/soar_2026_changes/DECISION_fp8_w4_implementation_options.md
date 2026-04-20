# DECISION DOCUMENT: Path 1 Implementation Options

**Date**: 2026-04-21  
**User Concerns**: Dependency conflicts, submission size (2GB limit), install time  
**Status**: Evaluating options — awaiting decision

---

## Context: Your Current Setup

From `prepare_env.sh`:
- Pre-built wheels collected locally: flash_attn, gptqmodel, transformers, torchao, sgl_kernel
- Careful dependency management via `--force-reinstall --no-deps` to avoid cascading breaks
- Submission package = collection of wheels + sglang source
- Size limit: **2GB strict**
- Install method: `uv pip install` with isolated dependency tree

---

## Option A: Use TRT-LLM PyTorch Wheel (Original Recommendation)

### Implementation
```bash
pip install tensorrt-llm  # or download wheel locally
torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(...)
```

### Pros ✅
- **Shortest timeline**: 1-2 days, minimal code changes
- **Battle-tested**: NVIDIA's production kernel
- **Easiest validation**: Just import and test

### Cons ❌
- **Wheel size**: ~300-500 MB (unacceptable for 2GB budget when combined with all other wheels)
- **Dependency hell**: TRT-LLM depends on:
  - CUDA runtime libraries
  - PyTorch (pins specific version)
  - CUTLASS (if built from source)
  - May conflict with your existing `transformers==4.57.1`, `flash-attn==2.8.3+cu128sm120` pinning
- **Install time**: First install ~5-10 min (download + unpack + validation), but every reinstall costs time
- **Submission package**: 
  - TRT-LLM wheel would be ~30% of your 2GB budget alone
  - Combined with all other wheels likely exceeds 2GB
  - Or forced to drop other components
- **Maintenance burden**: 
  - TRT-LLM API changes between versions
  - Your submission becomes dependent on NVIDIA's release schedule
  - If TRT-LLM wheel breaks on official eval system, submission fails

### Risk Assessment
- **Dependency conflict**: ⚠️ **High** — TRT-LLM likely pulls in conflicting versions
- **Size**: ⚠️ **Blocker** — 300-500 MB incompatible with 2GB limit + existing wheels
- **Robustness**: ⚠️ **Medium** — External dependency on eval system

### Verdict
❌ **NOT RECOMMENDED** — Violates your submission size constraint + introduces dependency risk similar to flash-attn issues you've already solved.

---

## Option B: Integrate CUTLASS Kernel into sgl-kernel (RECOMMENDED)

### Implementation
1. **sgl-kernel changes**:
   - Add CUTLASS 3.x as git submodule (in `sgl-kernel/third_party/cutlass/`)
   - Create `sgl-kernel/csrc/gemm/blackwell_fp8_w4/blockwise_gemm.cu` (C++ wrapper)
   - CMakeLists.txt: Generate kernel from CUTLASS CuTe DSL at build time
   - Export Python binding (pybind11)

2. **Integration in sglang**:
   - In `linear.py`: Call sgl-kernel's new FP8 GEMM directly
   - Scale conversion: GPTQ group-wise → per-row (in weight loading)

3. **Building**:
   ```bash
   cd sgl-kernel
   export CXX=g++ CC=gcc CCACHE_DIR=/root/.ccache
   make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
   # Kernel generated + compiled, all in sgl_kernel-*.whl
   ```

### Pros ✅
- **Single wheel**: Integrates into existing `sgl_kernel-*.whl` (no new dependency)
- **No size overhead**: CUTLASS source is <10MB, kernel compiled once, bundled in wheel
- **Your control**: No external dep, everything in your submission
- **No dependency conflicts**: CUTLASS is header-only, no runtime dep
- **Submission-safe**: One wheel, deterministic build, no external services
- **Future-proof**: CUTLASS stable, maintained by NVIDIA, well-documented

### Cons ❌
- **Build time**: First build ~1-2 hours (kernel generation + compilation)
- **Incremental builds**: Kernel recompilation might be needed if headers change (~10-15 min)
- **Implementation effort**: **Medium** (5-7 days)
  - CUTLASS integration into CMakeLists
  - CuTe DSL customization for W4A8 format
  - Python binding boilerplate
  - Tuning for MiniCPM shapes
- **Complexity**: Higher than Option A, but lower than writing custom kernel

### Risk Assessment
- **Dependency conflict**: ✅ **None** — CUTLASS is header-only
- **Size**: ✅ **Safe** — Wheel stays in your budget
- **Build complexity**: ⚠️ **Medium** — CUTLASS CMake can be finicky, but well-documented
- **Robustness**: ✅ **High** — Deterministic, self-contained build

### Build Timeline
- **First build** (on fcloud): ~1.5-2 hours
- **Subsequent incremental builds** (after file changes): ~10-15 min
- **Post-submission**: No rebuild needed, wheel is final

### Verdict
✅ **RECOMMENDED** — Solves all your concerns:
- No new external dependencies
- Fits in submission budget
- Self-contained, reproducible build
- No risk of dependency hell

---

## Option C: Write Custom SM120 FP8 Kernel

### Implementation
Write from scratch in `sgl-kernel/csrc/gemm/sm120_fp8_w4/`.

### Pros ✅
- **Full control**: Optimize specifically for MiniCPM shapes
- **No CUTLASS dependency**: Smaller build time (potentially)

### Cons ❌
- **Very high effort**: **4-6 weeks** (kernel dev is slow)
- **High risk**: 
  - Likely 10-20% slower than CUTLASS (unoptimized)
  - Accuracy bugs in FP8 dequant + MMA
  - Occupancy/register pressure issues
  - SM120-specific MMA scheduling unfamiliar territory
- **Maintenance**: Debugging custom CUDA is time-consuming

### Risk Assessment
- **Technical risk**: ⚠️ **Very High** — CUDA kernel dev is hard
- **Performance risk**: ⚠️ **High** — May not meet expected 5-8% speedup
- **Timeline**: ❌ **Unacceptable** — 4-6 weeks too slow

### Verdict
❌ **NOT RECOMMENDED** — Too slow, too risky, diminishing returns vs Option B.

---

## Option D: Use Existing Pre-Compiled Binary (Alternative)

Could we:
- Check if NVIDIA publishes pre-compiled SM120 FP8 kernels (unlikely)
- Use vLLM's SM90 FP8 kernel as template (might work)
- Compile TRT-LLM once off-fcloud, save binary, submit binary (complex, risky)

### Verdict
❌ **Not viable** — No pre-compiled binaries exist, compilation must happen on submission system.

---

## Option E: Defer to Path 2 (Speculative Decoding + Focus on Ngram)

### Rationale
- **Ngram speculative decoding** = pure Python, no kernel, no deps
- Expected: **1.5-2.5× decode speedup** (S1 target directly)
- **Zero dependency risk**: Just add server args
- **Faster to test**: 1 day instead of 2-3 weeks

### If chosen:
1. Test ngram with `--speculative-algorithm NGRAM --speculative-num-draft-tokens 12`
2. Validate acceptance rate on eval workload
3. If good (>60% acceptance): Commit + submit
4. If poor (<50%): Fall back to Path 1 later

### Risk
- **Ngram is workload-dependent**: Cold cache, no gain initially
- Requires eval workload to be repetitive (new 32K-512K token dataset may not be)

### Verdict
⚠️ **Alternative path** — Lower risk, faster to test, but uncertain gain on new workload.

---

## Recommendation Matrix

| Criterion | Option A | Option B | Option C | Option E |
|-----------|----------|----------|----------|----------|
| **Size (2GB)** | ❌ Blocker | ✅ Safe | ✅ Safe | ✅ Safe |
| **Dependency conflicts** | ❌ High risk | ✅ None | ✅ None | ✅ None |
| **Effort** | ⚠️ 1-2 days | 📊 5-7 days | ❌ 4-6 weeks | ✅ 1-2 days |
| **Expected gain** | 📊 5-8% | 📊 5-8% | ⚠️ 2-4% | 📊 20-50% |
| **Risk** | ❌ High | ✅ Low | ❌ Very High | ⚠️ Medium |
| **Robustness** | ⚠️ External dep | ✅ Self-contained | ✅ Self-contained | ✅ Pure Python |
| **Maintenance** | ⚠️ NVIDIA dep | ✅ CUTLASS stable | ❌ Custom CUDA | ✅ sglang native |

---

## Final Recommendation

### **Short-term (Best Option): Option B + Option E (Parallel Tracks)**

**Track 1**: Option B (CUTLASS integration into sgl-kernel)
- Start: 2026-04-22
- Timeline: 5-7 days implementation + testing
- Parallel: Incremental builds don't block development
- Result: Permanent 5-8% speedup, fits submission

**Track 2**: Option E (Ngram speculative decoding quick test)
- Start: 2026-04-22 (parallel)
- Timeline: 1-2 days to test
- Low risk, might yield 20-50% if workload is repetitive
- If fails: No harm, revert

**Combined**: If both succeed, cumulative gains could be **25-65%** (multiplicative).

### **Why NOT Option A**
1. **Submission package bloat**: 300-500 MB is unacceptable at 2GB budget
2. **Dependency hell**: You've already solved this with your careful wheel management
3. **External risk**: TRT-LLM APIs change, official eval env may not have it

### **Why Option B > Option C**
- CUTLASS is mature, proven, well-optimized
- Custom kernel is a trap (slow dev, high maintenance, uncertain results)

---

## Decision Required

Which path do you prefer?

1. **Option B + Option E** (Recommended): Invest 1 week total, dual-track approach
2. **Option E only** (Quick): Test ngram immediately, defer CUTLASS
3. **Option B only**: Commit to CUTLASS kernel integration
4. **Other**: Propose alternative

Let me know and I'll start Phase 2 immediately.
