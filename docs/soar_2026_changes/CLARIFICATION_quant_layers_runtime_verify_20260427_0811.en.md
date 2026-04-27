# Clarification — Quantization layers in MiniCPM-SALA submission and how to verify them at runtime

**Date**: 2026-04-27 08:11
**Context**: User asked whether the submission already implements W4A8 (since we use GPTQ INT4 + Marlin + FP8 KV), whether the lightning recurrent state becomes FP8 when `--kv-cache-dtype fp8_e5m2` is set, whether GPTQ INT4 is the same as NVFP4, and how to verify these claims at runtime (not only via code review).
**Status**: Discussion / reference. No code change. Continues the analysis recorded in `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md`.

---

## 1. The misconception triangle (and why each part is wrong)

| Claim | Reality |
|---|---|
| "GPTQ INT4 weights + Marlin kernel = W4A8" | ❌ Marlin dequantizes INT4 → BF16 inside the kernel and runs **BF16 × BF16** on tensor cores. Activation precision is **BF16**, not 8-bit. |
| "`--kv-cache-dtype fp8_e5m2` makes attention compute in FP8" | ❌ It only changes how K/V are **stored** between layers. They are cast back to BF16 the moment they're read for attention math. |
| "`--kv-cache-dtype fp8_e5m2` also covers the lightning recurrent state" | ❌ Lightning state is allocated by a different pool (`MambaPool`) and stays BF16. The flag does not touch it. |
| "GPTQ INT4 ≈ NVFP4" | ❌ Two completely different number formats and two different hardware execution paths. |

The rest of this document spells out each of these and gives you **runtime verification recipes** so you don't have to take it on trust from code review alone.

---

## 2. What we actually ship today (data-flow view)

### 2.1 GEMM (qkv_proj, o_proj, MLP up/down/gate, lightning Q/K/V/O)

```
Input activation (BF16)  ──┐
                           │
                           ▼
                   ┌──────────────────────────────┐
                   │ Marlin GEMM kernel           │
GPTQ INT4 weight ─►│  1. Load INT4 weight tile    │── BF16 output activation
                   │  2. Dequant to BF16 (INT4 →  │
                   │     BF16 via per-group scale)│
                   │  3. Compute BF16 × BF16 on   │
                   │     BF16 tensor core (148 TF)│
                   └──────────────────────────────┘
```

- Weight **storage** : INT4 (4 bits/value) — saves DRAM bandwidth + footprint.
- Weight **compute precision after dequant** : BF16.
- Activation **storage and compute precision** : BF16.
- Tensor-core path : **BF16** at 148 TFLOPS on SM120.
- **No FP8 tensor core (296 TFLOPS) is touched.**

This is what GPTQ + Marlin always does, by design. To get into the FP8 tensor-core path, you need a different kernel (W4A8 / mxfp8 / nvfp4-MMA), not just a different config flag.

### 2.2 Standard-attention KV cache (the 8 `minicpm4` layers)

Code anchors in `python/sglang/srt/mem_cache/memory_pool.py`:
- Line 668-670: `if dtype is FP8: store_dtype = torch.uint8 else store_dtype = dtype`
- Line 1019-1021: on read, `cache_k = cache_k.view(self.store_dtype)` then **cast back** before returning to attention math.

Concretely:

```
Attention layer outputs k, v in BF16
                │
                ▼
    cast / pack to FP8 e5m2 (storage_dtype = uint8 view)
                │
                ▼
   stored in MHATokenToKVPool
                │
                ▼  (at next attention step that reads it)
   load FP8 bytes
                │
                ▼
   cast back to BF16
                │
                ▼
   FlashInfer / FA backend computes softmax(QKᵀ)V in BF16
```

So `--kv-cache-dtype fp8_e5m2` is a **bandwidth + memory** optimization. The compute math is BF16. It's still a valuable optimization (KV reads dominate decode), just not a "compute precision" optimization.

### 2.3 Lightning recurrent state (the 24 `lightning` layers)

Code anchors:
- `python/sglang/srt/mem_cache/memory_pool.py:128` — `class MambaPool` (separate from `MHATokenToKVPool`).
- The pool is created based on `model_config` linear-attn parameters, not from `--kv-cache-dtype`.
- Allocation dtype is BF16 (or FP32 for accumulation) regardless of the KV flag.

`fused_recurrent_simple_gla` and `chunk_simple_gla` both read/write this state in BF16.

So today:
- 24 lightning layers × per-request × `(num_kv_heads, d, d)` BF16 state lives in `MambaPool`.
- Decoding one token reads + writes this BF16 state for every lightning layer.
- This is **completely unaffected** by `--kv-cache-dtype fp8_e5m2`.

### 2.4 Putting it all together — current data layout

| Component | Storage dtype | Compute dtype | Hardware unit | Controlled by |
|---|---|---|---|---|
| Most weights (MLP, lightning Q/K/V/O, std-attn QKV non-sparse) | INT4 (GPTQ) | BF16 (post-dequant) | BF16 tensor core 148 TF | preprocess_model.py + Marlin |
| 8 std-attn QKV weights (sparse_qkv_w8) | INT8 (GPTQ) | BF16 (post-dequant) | BF16 tensor core | `SOAR_GPTQ_SPARSE_QKV_BITS=8` |
| All input/output activations | BF16 | BF16 | BF16 tensor core | model dtype |
| Std-attn KV cache | **FP8 e5m2 (storage)** | BF16 (cast on read) | BF16 attention math | `--kv-cache-dtype fp8_e5m2` |
| Lightning recurrent state | **BF16** | BF16 | Triton kernel BF16 | not currently a flag |
| FlashInfer / FA backend (8 layers) | n/a | BF16 | BF16 | backend default |
| Triton SimpleGLA kernels (24 layers) | n/a | BF16 | BF16 | kernel default |

### 2.5 What "real W4A8 + FP4-KV + FP8-state" would look like

| Component | Storage dtype | Compute dtype | Hardware unit |
|---|---|---|---|
| Most weights | INT4 (GPTQ) | **FP8** (post-dequant) | **FP8 QMMA 296 TF** |
| Activations | **FP8 e4m3** | FP8 | FP8 QMMA |
| Std-attn KV cache (mixed) | FP8 / **FP4 E2M1** for inner layers | BF16 cast (or FP4 native if QMMA path) | BF16 / FP4 attention |
| Lightning recurrent state | **FP8** | FP8 (or BF16 accum) | FP8 in Triton |

These three are **three separately-engineered changes**, none of which are automatic from current flags.

---

## 3. GPTQ INT4 ≠ NVFP4

Two completely different number systems and two different SM120 hardware paths.

| Format | Encoding | Hardware execution | Used by us for |
|---|---|---|---|
| **GPTQ INT4** | Uniform integer: `value = scale × (q − zero_point)`, group_size=128, per-group BF16 scale | Marlin custom dequant → BF16 tensor core (148 TF) | Today's weights |
| **NVFP4 (E2M1)** | Floating-point: 1 sign + 2 exp + 1 mantissa, with shared block-scale (typically per-16-element block) | **QMMA tensor core directly on FP4 inputs** at 593 TFLOPS on SM120 | Test 21 (failed: ~12 % accuracy) |

A model can be "4-bit" in either format, but:
- INT4 is uniform-grid integer; saturates near 0 and clusters far values.
- FP4 is logarithmic; better dynamic range, worse near-zero precision.
- Most LLM weights are mean-zero and Laplacian-like → INT4 with per-group scale fits well; FP4 sometimes needs QAT.

Conclusion: yes, our current quantization is **GPTQ INT4**, not NVFP4. They are not interchangeable. (The NVFP4 attempt was a separate branch and produced catastrophic accuracy regression on Test 21.)

---

## 4. Runtime verification recipes (the heart of this document)

You asked: "besides code review, any runtime way to verify what was claimed?" — yes, here are concrete checks. Each one operates on the running fcloud server (or a local minimal repro) and produces an unambiguous yes/no answer.

### V1 — Check tensor dtypes inside the running model

After server starts, run a small Python probe in the same process (e.g., via an SGLang-internal hook, or by attaching with `py-spy dump --pid <pid>` and inspecting object dtypes).

The simplest path: add a one-shot debug print before serving any request. Edit `python/sglang/srt/models/minicpm.py` `__init__` of one Std-attn and one Lightning layer (DO NOT submit this — local-only):

```python
# inside MiniCPMAttention.__init__ (std-attn)
print(f"[VERIFY] layer={layer_id} type=std qkv_proj.weight.dtype={self.qkv_proj.qweight.dtype} (expect torch.int32 packed)")
print(f"[VERIFY] layer={layer_id} type=std qkv_proj.scales.dtype={self.qkv_proj.scales.dtype} (expect bfloat16)")

# inside MiniCPMLightningMixer.__init__
print(f"[VERIFY] layer={layer_id} type=lightning qkv_proj.qweight.dtype={self.qkv_proj.qweight.dtype}")
```

Expected output:
```
[VERIFY] layer=0 type=lightning qkv_proj.qweight.dtype=torch.int32 (4-bit packed)
[VERIFY] layer=2 type=std qkv_proj.scales.dtype=torch.bfloat16
```

The `int32` packed weight (Marlin's storage layout) + BF16 scales prove **weight is INT4 stored, BF16 dequantized**. There is no FP8 scale anywhere in the GEMM path.

### V2 — Confirm the KV pool storage dtype (FP8) and the cast-back

After server start, list the KV pool object (this is in `model_runner.token_to_kv_pool`):

```python
# scratch script run inside the same Python session
import sglang
from sglang.srt.managers.scheduler import ... # however you reach the runner
pool = engine.runner.token_to_kv_pool
print(type(pool).__name__)            # MHATokenToKVPool
print("dtype     =", pool.dtype)      # the *logical* dtype; expect torch.bfloat16
print("store_dtype=", pool.store_dtype)  # expect torch.uint8 (FP8 stored as raw bytes)
print("k_buffer dtype=", pool.k_buffer[0].dtype)  # torch.uint8
```

If `dtype == bfloat16` and `store_dtype == uint8`, that **proves** the pool stores FP8 packed but reports BF16 to the attention backend → the cast-back happens on every read. This is exactly what the code at `memory_pool.py:1019-1021` does.

### V3 — Confirm the lightning recurrent state dtype

Same idea but for the Mamba pool:

```python
mamba_pool = engine.runner.mamba_pool   # or wherever the lightning state lives
print(type(mamba_pool).__name__)        # MambaPool
for i, st in enumerate(mamba_pool.state_buffers[:3]):
    print(f"layer {i} state dtype = {st.dtype}, shape = {tuple(st.shape)}")
```

Expected output:
```
layer 0 state dtype = torch.bfloat16, shape = (max_bs, num_kv_heads, d, d)
```

If you see `torch.bfloat16` here, **it confirms the lightning state is BF16 today**, regardless of the KV flag. To make it FP8, the pool dtype itself has to change — and that requires kernel changes too (proposal #3).

### V4 — Confirm tensor-core unit actually used (FP8 vs BF16)

This is the question your runtime check most directly addresses: "is FP8 tensor core actually firing?"

Use **Nsight Compute** on a single decode iteration:

```bash
# on fcloud, with the server idle except for one in-flight request
nsys profile --trace=cuda,nvtx -o /tmp/trace_decode \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python /root/data/eval_model_001.py --num_samples 1 ...

# OR use ncu for per-kernel SM-unit info
ncu --target-processes all --section ComputeWorkloadAnalysis \
    --launch-skip 200 --launch-count 30 \
    -o /tmp/ncu_marlin python /root/data/eval_model_001.py --num_samples 1 ...
```

Then open the `.ncu-rep` and look at the GEMM kernel rows. The relevant counter is:

- `sm__inst_executed_pipe_tensor_op_hmma` — **BF16/FP16 tensor-core ops (HMMA)**
- `sm__inst_executed_pipe_tensor_op_qmma` — **FP8 tensor-core ops (QMMA)** on Blackwell

If you only see `hmma` (BF16) firing for the Marlin kernels and `qmma` is **zero**, that's the runtime proof we are not using FP8 tensor cores at all. After implementing W4A8, you should see `qmma` jump up.

For our current submission, expectation is: `hmma` >> 0, `qmma` ≈ 0.

### V5 — Quick sanity check via memory bandwidth

Decode is memory-bound. If "FP8 KV" is really only storage-side, then disabling it (going to BF16 KV) should:
- **Roughly double** KV read traffic (from ~1 byte/elt to 2 bytes/elt).
- **Reduce** decode TPS by some predictable amount (NOT halve it, because weights also dominate; expect ~10-20 % loss at long context).
- **Not change** GEMM latency at all (because GEMM activations stay BF16 in both modes).

Run two side-by-side benchmarks:

```bash
# baseline (FP8 KV)
SGLANG_SERVER_ARGS=(... --kv-cache-dtype fp8_e5m2 ...)
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# variant (BF16 KV)
SGLANG_SERVER_ARGS=(... --kv-cache-dtype auto ...)
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

If the difference is in the ~10-20 % range, that confirms FP8 KV is bandwidth-only. If it were a compute-precision change you'd see a much larger swing (and Marlin GEMM would also speed up, which it won't). We've actually run this comparison before — see Test 13 (BF16 KV, Smax slower) vs Test 12 (FP8 KV).

### V6 — Quick sanity check for lightning-state dtype

If lightning state is really BF16, doubling `max-running-requests` should ~double the L2 footprint of MambaPool. If it is silently FP8, the doubling is half as much.

Practical instrument: read `nvidia-smi --query-gpu=memory.used` immediately after server start at two `--max-running-requests` values:

```
--max-running-requests 12 → memory.used = X MB
--max-running-requests 24 → memory.used = Y MB
```

The delta `Y - X` should equal `12 × (24 layers × bs_per_token × num_kv_heads × d × d × bytes_per_elt)`. With BF16 → 2 bytes/elt, this should match the BF16 expectation. If it's half, lightning state would somehow be FP8 (it isn't, but this is a runtime-only confirmation).

### V7 — Use SGLang's built-in profiling hooks

SGLang integrates PyTorch profiler. Enable via:

```bash
curl -X POST http://127.0.0.1:30000/start_profile
# do one inference
curl -X POST http://127.0.0.1:30000/stop_profile
```

The trace JSON includes per-kernel info. Filter for kernel names:
- `marlin_gemm_*` — should show BF16 input dtype.
- `flash_attention_v2_kernel_*` — should show BF16 input dtype.
- `simple_gla_*` (Triton autogen names) — should show BF16 input dtype.

This is the cheapest way to inspect actual kernel signatures end-to-end.

---

## 5. Summary

| Question | Answer | How to verify at runtime |
|---|---|---|
| Are we already W4A8? | No. Weights are INT4-stored but compute is BF16 × BF16 on BF16 tensor cores. | V1 + V4 (ncu shows `hmma`, no `qmma`) |
| Does FP8 KV cache imply FP8 attention compute? | No. Storage-only; cast back to BF16 on read. | V2 (`pool.dtype=bf16, store_dtype=uint8`) |
| Does FP8 KV cache also cover lightning state? | No. Lightning state is in `MambaPool`, BF16. | V3 (`mamba_pool.state_buffers[i].dtype=bf16`) |
| Is GPTQ INT4 the same as NVFP4? | No. Different encoding, different hardware path. NVFP4 uses QMMA on FP4 inputs natively. | V4 (look for `qmma_e4m3` or `qmma_e2m1` in ncu output) |
| Is "lightning state FP8" a free side-effect of any current flag? | No. Requires kernel modification (proposal #3). | V3 confirms the starting point |

---

## 6. Open questions (before next proposal)

1. Run V2 + V3 + V4 once on the v18 baseline to lock down the "today" picture as a printed reference? (Cost: ~10 minutes of fcloud time.)
2. Of #1 / #3 / #4 / #6 in the prior research note, which to draft as a formal proposal next?
3. For #1 (W4A8): GPTQ Marlin extension vs upstream `mxfp8 / nvfp4` path? (Latter integrates more cleanly with SGLang's existing infra but loses our GPTQ-INT4 calibration.)
4. For #3 (lightning state FP8): per-head static scale or per-token dynamic? Static is fast but riskier.
