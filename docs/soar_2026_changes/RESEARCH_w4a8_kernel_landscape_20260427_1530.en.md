# Research note: W4A8 kernel landscape, Phase 0 microbench plan, W4A16 vs W4A8 cost analysis (2026-04-27 15:30)

This note consolidates the Q&A from session 2026-04-27 afternoon for review and tracking. Companion: `RESEARCH_w4a8_kernel_landscape_20260427_1530.zh.md`.

---

## 1. Latest champion configuration (intelligence note)

Per WeChat post https://mp.weixin.qq.com/s/w1g3njB24rxLCiCLxWFD7Q (latest weekly champion writeup):

- **Quantization**: W4A16 GPTQ (same dtype family as our v18 baseline)
- **KV cache**: mixed **NVFP4 + FP8** KV cache
- **Other components**: not disclosed in post

Implication for our roadmap: champion is **NOT using W4A8** today. The "real W4A8" path is still unproven on this hardware/competition. Our stronger near-term lever may be **NVFP4 KV cache** (already prototyped in `ANALYSIS_nvfp4_offline_quant_*` docs) rather than W4A8 kernel work. Treat W4A8 as a research bet, not a guaranteed win.

---

## 2. Existing W4A8 kernel inventory (verified by code search)

### 2.1 Inside `sgl-kernel/` (already vendored, build-ready)

| Kernel | File | Weight | Activation | MMA | Arch guard | Use case |
|---|---|---|---|---|---|---|
| `qserve_w4a8_per_group_gemm` | `csrc/gemm/qserve_w4a8_per_group_gemm.cu` | INT4 packed, group=128 | **INT8** symmetric | I8 IMMA via inline PTX | `__CUDA_ARCH__ ≥ 800` (SM80+) | **dense GEMM**, MIT QServe |
| `qserve_w4a8_per_chn_gemm` | `csrc/gemm/qserve_w4a8_per_chn_gemm.cu` | INT4, per-channel | INT8 | I8 IMMA | SM80+ | dense GEMM |
| `cutlass_w4a8_moe_mm` | `csrc/moe/cutlass_moe/w4a8/*` | INT4 packed | **FP8 e4m3** | FP8 QMMA via Hopper TMA | **SM90 only** (`is_hopper()` gate) | **MoE grouped GEMM only**, NOT dense |

### 2.2 Inside vllm `csrc/quantization/machete/`

- README confirms: `compute_type = a.dtype` where `a` is BF16/FP16. **Machete is W4A16, not W4A8.**
- Hopper-optimized successor to Marlin for the same problem space we already cover with our SM120 Marlin tiles (CHANGE_0125).
- A few downstream forks have added FP8 activation, but it is NOT in vllm mainline.

### 2.3 Conclusion on existing kernel availability for SM120 dense W4A8

| Path | Existing kernel? | SM120 ready? | Effort to integrate |
|---|---|---|---|
| **Dense W4-INT8** | YES — QServe `qserve_w4a8_per_group_gemm` (already in our sgl-kernel) | YES — SM80+ inline PTX path | **Low**: Python wiring in `gptq.py` + weight repack + per-token INT8 quantizer |
| **Dense W4-FP8** | **NO** anywhere we found (sgl-kernel MoE-only on SM90; vllm Machete is W4A16) | N/A | **High**: write/port a new kernel; possibly weeks |
| **Custom CUTLASS mixed-input** | starting from scratch | depends | Highest |

This **flips the priority** stated earlier. Updated recommendation:

- **Dense W4-FP8 on SM120 = research project.** Combined with point #1 above (no champion is doing it), Option A is unattractive as the first iteration.
- **Dense W4-INT8 (QServe) = pragmatic first attempt** if and only if Phase 0 microbench shows SM120 INT8 IMMA throughput is competitive with FP8.

---

## 3. Phase 0 microbench: SM120 INT8 IMMA vs FP8 QMMA vs BF16

### 3.1 What we need to measure

The official `SM120_RTX_PRO_HARDWARE.md` lists FP8=296 TF, BF16=148 TF, FP4=593 TF, but **does NOT list INT8**. We need to confirm whether SM120 INT8 IMMA runs at the FP8 tier (~296 TF) or is throttled to the BF16 tier (~148 TF). This decides whether QServe W4-INT8 is worth integrating.

### 3.2 Simple PyTorch-only microbench (no cutlass_profiler build)

PyTorch already wraps the relevant tensor-core paths:
- `torch.matmul(bf16, bf16)` → BF16 tensor cores
- `torch._scaled_mm(fp8, fp8, ...)` → **FP8 QMMA** (the simple FP8 test you asked about)
- `torch._int_mm(int8, int8)` → INT8 IMMA

This makes a **single ~5-minute fcloud script** sufficient. No cutlass build, no cuBLAS calls.

### 3.3 Script (will be written to `/root/bench_int8_vs_fp8_sm120.py`)

```python
import torch, time
torch.manual_seed(0)
DEV = "cuda"

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# Representative MiniCPM shapes: qkv_proj/o_proj at hidden=4096; gate_up/down at MLP=14336
shapes = [
    (4096, 4096, 4096),   # qkv-like
    (4096, 14336, 4096),  # gate_up-like
    (4096, 4096, 14336),  # down_proj-like
]
for M, N, K in shapes:
    a_bf16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    a_fp8  = a_bf16.to(torch.float8_e4m3fn)
    b_fp8  = b_bf16.to(torch.float8_e4m3fn)
    a_i8   = (a_bf16 * 100).clamp(-128, 127).to(torch.int8)
    b_i8   = (b_bf16 * 100).clamp(-128, 127).to(torch.int8)
    sa = torch.tensor(1.0, device=DEV); sb = torch.tensor(1.0, device=DEV)

    t_bf16 = bench(lambda: torch.matmul(a_bf16, b_bf16))
    t_fp8  = bench(lambda: torch._scaled_mm(
        a_fp8, b_fp8.t().contiguous().t(),
        scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    try:
        t_i8 = bench(lambda: torch._int_mm(a_i8, b_i8))
        i8_str = f"INT8={t_i8*1e3:.3f}ms ({2*M*N*K/t_i8/1e12:.1f} TFLOPS)"
    except Exception as e:
        i8_str = f"INT8=N/A ({e})"

    flops = 2 * M * N * K
    print(f"[{M}x{N}x{K}]  "
          f"BF16={t_bf16*1e3:.3f}ms ({flops/t_bf16/1e12:.1f} TF)  "
          f"FP8={t_fp8*1e3:.3f}ms ({flops/t_fp8/1e12:.1f} TF)  "
          f"{i8_str}")
```

### 3.4 Run command (when fcloud is up)

```bash
# After: python3 scripts/fcloud/fcloud_workflow.py setup
python3 /root/bench_int8_vs_fp8_sm120.py 2>&1 | tee /root/phase0_int8_vs_fp8_sm120.log
# Collect log back to local for tracking
```

Results will be appended to **a new follow-up document `PHASE0_INT8_vs_FP8_SM120_<timestamp>.md`** (not into this research note) so we can keep raw measurement separate from research narrative.

### 3.5 Decision rule

| Result | Action |
|---|---|
| INT8 ≥ ~250 TF (≈ FP8 tier) | Green-light QServe W4-INT8 integration as next iteration |
| INT8 ≈ ~148 TF (BF16 tier) | INT8 has no compute win on SM120; abandon W4-INT8 path |
| FP8 < ~250 TF | Spec contradiction; investigate before any 8-bit path |

---

## 4. W4A16 (current Marlin) vs W4A8 — beyond TFLOPS

### 4.1 Where W4A16 already wins (current baseline)

Marlin W4A16 already has:
- **2× weight bandwidth** vs BF16 weights (INT4 storage = 0.5 byte/elem, BF16 = 2 byte/elem → 4× actually, but counting against FP8 storage which is 1 byte/elem it's 2× weight bandwidth). At decode the workload is **weight-bandwidth-bound**, so this is the dominant lever and Marlin already has it.
- BF16 MMA at 148 TF on SM120 → far above the actual sustained rate at decode (decode is bandwidth-bound, not compute-bound).
- BF16 activation precision → minimal accuracy loss.

### 4.2 Where W4A8 *could* add over W4A16

At the **per-layer microscopic level**, switching the activation from BF16 → 8-bit changes three things:

| Aspect | W4A16 (Marlin BF16 act) | W4A8 (INT8 or FP8 act) | Delta |
|---|---|---|---|
| Weight storage | INT4 (0.5 B/elem) | INT4 (0.5 B/elem) | **0** (same) |
| Activation memory traffic | BF16 (2 B/elem) | INT8 / FP8 (1 B/elem) | **−50% activation BW** |
| MMA throughput peak | BF16 = 148 TF | FP8 = 296 TF (or INT8 if equal) | **+100% peak compute** |
| Output / accumulator | BF16 | BF16 | 0 |
| Per-token overhead | none | per-token activation quantization (one mul + cast) | small **add** |

### 4.3 Quantitative expectation by workload

Decode (S1) and prefill (S8/Smax) load tensor cores very differently. Estimates:

#### Decode bs=1 (S1, our baseline 121.71s)
- **Memory-bandwidth bound on weights**, not on activations or compute.
- Activation BW reduction: tiny (activation is just a single token = O(hidden_size) bytes, dwarfed by O(hidden×hidden) weight bytes).
- Compute peak doubling: irrelevant (MMA is not the bottleneck).
- **Realistic S1 gain: ≤ 5%**, possibly negative if activation quantization adds latency.
- Risk: per-token quantization adds a small kernel launch + memory pass per layer per token. At bs=1 every cycle counts.

#### Prefill / large M (S8 ~44s, Smax ~36s)
- **Compute and SMEM bound**, not pure weight BW.
- Activation BW reduction: moderate (activations grow with sequence length and matter at long context).
- Compute peak doubling: matters here; tensor cores stay closer to peak.
- **Realistic S8/Smax gain: 10–20% if kernel achieves ~70% of FP8/INT8 peak.**
- Risk: depends entirely on how well the new kernel is tuned. QServe was tuned for SM80; SM120 may need re-tuning.

### 4.4 Honest gain expectation table

| Tier | Baseline | Optimistic W4A8 | Pessimistic W4A8 | Expected most-likely |
|---|---|---|---|---|
| S1 | 121.71s | ~115s (−5%) | ~125s (+3%) | ≈ baseline ± 3% |
| S8 | 44.09s | ~36s (−18%) | ~42s (−5%) | ~40s (−9%) |
| Smax | 35.86s | ~29s (−19%) | ~34s (−5%) | ~32s (−11%) |

These are *theoretical estimates assuming the kernel works correctly on SM120 and accuracy stays > 99% normalized.* Actual numbers likely worse at first (untuned kernel).

### 4.5 Strategic verdict

- W4A8 is primarily a **prefill / long-context** optimization, not a decode optimization.
- The official speed dataset is reported to have more long-context samples than our local set, so prefill optimization should help officially even if local Smax doesn't move much.
- BUT: the champion using W4A16 + NVFP4 KV proves **we don't need W4A8 to be competitive.** NVFP4 KV cache is likely a higher-leverage optimization on this hardware.

---

## 5. Updated open questions and recommended sequence

1. **Phase 0 microbench (cheap, ~5 min on fcloud)** — run before any kernel work. **Awaiting your fcloud start.**
2. If Phase 0 passes for INT8: **Option B = QServe W4-INT8 integration** (low-effort, kernel already in repo).
3. If Phase 0 fails for INT8 AND the team has bandwidth for kernel research: defer W4A8, **prioritize NVFP4 KV cache** (matches champion combo).
4. Option A (custom W4-FP8 dense kernel) only if both (2) and (3) are exhausted.

---

## 6. Tracking

- Created: 2026-04-27 15:30
- Author: agent
- Companion ZH: `RESEARCH_w4a8_kernel_landscape_20260427_1530.zh.md`
- Triggers next document: `PHASE0_INT8_vs_FP8_SM120_<timestamp>.md` (created after fcloud microbench run)
