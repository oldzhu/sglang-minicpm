# MiniCPM-SALA Inference Flow Analysis
## SOAR 2026 — Kernel Optimization Roadmap

**Date**: 2026-05-25 | **Model**: MiniCPM-SALA-90B (GPTQ INT4, dense mode)  
**Baseline Config**: GPTQ (sparse_qkv_w8) + FP8 KV cache + `--force-dense-minicpm`

---

## 1. Model Architecture Summary

### 1.1 Global Dimensions

| Parameter | Value |
|-----------|-------|
| `num_hidden_layers` | **32** |
| `hidden_size` (D) | **4096** |
| `num_attention_heads` (Q heads) | **32** |
| `num_key_value_heads` (KV heads) | **8** (GQA ratio = 4:1) |
| `head_dim` | **128** (= 4096/32) |
| `intermediate_size` (MLP hidden) | **14336** |
| `vocab_size` | **150528** |

### 1.2 Layer Type Distribution

> **CRITICAL**: `--force-dense-minicpm` does **NOT** convert lightning layers to
> standard attention. It only sets `model_config.has_sparse_attention=False` and
> `model_config.sparse_layer_ids=[]`, which affects attention routing inside
> `MiniCPMAttention` (sparse→dense FA3). The module CLASS is determined by
> `config.mixer_types[layer_id]` at init time, independent of `--force-dense-minicpm`.
> The 24 lightning layers always use `MiniCPMLightningMixer` (GLA recurrent).

| Mixer Type | Count | Module Class | KV Cache? | Attention (current) |
|------------|-------|-------------|-----------|---------------------|
| `minicpm4` | **8 layers** | `MiniCPMAttention` | ✅ Paged KV | Dense FA3 (force_dense) |
| `lightning` | **24 layers** | `MiniCPMLightningMixer` | ❌ Recurrent state | GLA chunk/recurrent |

**Lightning layer sub-config** (`config.lightning_*`):
| Parameter | Value |
|-----------|-------|
| `lightning_nh` (heads) | **16** |
| `lightning_nkv` (KV heads) | **16** |
| `lightning_head_dim` | **64** |

**Key**: Lightning layers use **recurrent state** (no KV cache), while sparse layers use **paged KV cache**. Lightning layers have different head dimensions (64 vs 128) and different quantization paths.

### 1.3 Quantization Layout

All linear layers use **GPTQ INT4** (4-bit, `group_size=128`, symmetric, `desc_act=False`). Weight storage per layer:

| Projection | Shape (in×out) | Storage (INT4) |
|------------|-------------------|----------------|
| **Sparse: QKV** | 4096 × (32×128 + 2×8×128) = 4096 × 6144 | ~3.1 MB |
| **Sparse: O** | 4096 × 4096 | ~2.0 MB |
| **Sparse: Gate-Up** | 4096 × (2×14336) = 4096 × 28672 | ~14.3 MB |
| **Sparse: Down** | 14336 × 4096 | ~14.3 MB |
| **Lightning: QKV** | 4096 × (16×64 + 2×16×64) = 4096 × 3072 | ~1.5 MB |
| **Lightning: O** | 1024 × 4096 | ~0.5 MB |
| **Lightning: Z (gate)** | 4096 × 1024 (if enabled) | ~0.5 MB |

---

## 2. Per-Layer Forward Pass Trace

### 2.1 Sparse Attention Layer (`minicpm4`, 8 layers)

The `MiniCPMDecoderLayer.forward()` executes:

```
Input: hidden_states [B×T, 4096], residual, positions, forward_batch

Step 1: INPUT LAYERNORM + RESIDUAL
├─ hidden_states, residual = input_layernorm(hidden_states, residual)
│  └─ Op: RMSNorm(4096), element-wise, ~32K FLOPs/token
│  └─ In-place residual add: residual += hidden_states (fused)

Step 2: QKV PROJECTION (MATMUL — HEAVY)
├─ qkv, _ = qkv_proj(hidden_states)
│  └─ Op: Linear(4096 → 6144), GPTQ INT4
│  └─ Kernel: GPTQ Marlin (default, 148 TFLOPS) or W4A8 FP8 fused (if enabled)
│  └─ FLOPs: 2 × 4096 × 6144 ≈ 50M FLOPs/token
│  └─ W4A8 eligible: YES ✅

Step 3: SPLIT Q/K/V
├─ q, k, v = qkv.split([4096, 1024, 1024], dim=-1)
│  └─ Op: Split tensor (view/reshape, zero FLOPs)

Step 4: ROPE (if attn_use_rope=True)
├─ q, k = rotary_emb(positions, q, k)
│  └─ Op: RoPE (YaRN variant), ~2 × 4096 × 128 ≈ 1M FLOPs/token
│  └─ Can use fused_qk_norm_rope kernel if enabled

Step 5: ATTENTION COMPUTE (HEAVY)
├─ attn_output = self_attn(q, k, v, forward_batch)
│  └─ RadixAttention dispatches to MiniCPMBackend:
│
│  [PREFILL path]:
│  ├─ Dense (seq_len < dense_len threshold): FlashAttention v3
│  │  └─ FLOPs: O(T² × head_dim), memory-bound for long T
│  │
│  ├─ Sparse (seq_len ≥ dense_len): MiniCPM sparse attention
│  │  └─ TopK selection + compressed K1/K2 + sparse_kernel_extension
│  │  └─ Currently FORCE_DENSE = 1, so this path is SKIPPED
│  │
│  [DECODE path]:
│  └─ FlashAttention v3 decode (1 token × KV cache)
│     └─ FLOPs: O(T_cache × head_dim) per head = ~128K FLOPs

Step 6: OUTPUT GATE (if use_output_gate)
├─ o_gate_output = sigmoid(o_gate(hidden_states))
│  └─ Op: Linear(4096→4096) + sigmoid, GPTQ INT4
│  └─ NOT W4A8 eligible
├─ attn_output *= o_gate_output

Step 7: O PROJECTION (MATMUL — HEAVY)
├─ output, _ = o_proj(attn_output)
│  └─ Op: Linear(4096 → 4096), GPTQ INT4
│  └─ Kernel: GPTQ Marlin or W4A8 FP8 fused
│  └─ FLOPs: 2 × 4096 × 4096 ≈ 33M FLOPs/token
│  └─ W4A8 eligible: YES ✅

Step 8: RESIDUAL SCALE
├─ hidden_states *= residual_scale  (depth-dependent, ~0.18)

Step 9: POST-ATTN LAYERNORM + RESIDUAL
├─ hidden_states, residual = post_attention_layernorm(hidden_states, residual)
│  └─ Op: RMSNorm(4096)

Step 10: MLP GATE-UP PROJECTION (MATMUL — HEAVIEST)
├─ gate_up, _ = gate_up_proj(hidden_states)
│  └─ Op: Linear(4096 → 28672), GPTQ INT4
│  └─ FLOPs: 2 × 4096 × 28672 ≈ 235M FLOPs/token  ← LARGEST SINGLE OP
│  └─ W4A8 eligible: YES ✅

Step 11: SILU GATE + MULTIPLY
├─ x = SiluAndMul(gate_up)
│  └─ Op: Split → siLU(gate) × up, fused
│  └─ FLOPs: ~29K FLOPs/token (negligible vs matmuls)

Step 12: MLP DOWN PROJECTION (MATMUL — HEAVY)
├─ x, _ = down_proj(x)
│  └─ Op: Linear(14336 → 4096), GPTQ INT4
│  └─ FLOPs: 2 × 14336 × 4096 ≈ 117M FLOPs/token
│  └─ W4A8 eligible: YES ✅

Step 13: RESIDUAL SCALE + OUTPUT
├─ hidden_states *= residual_scale
└─ Return (hidden_states, residual)
```

**Sparse layer FLOPs per token (decode)**:
| Op | FLOPs | % Total |
|----|-------|---------|
| Gate-Up matmul | 235M | 54% |
| Down matmul | 117M | 27% |
| QKV matmul | 50M | 11% |
| O matmul | 33M | 8% |
| **Total** | **~435M** | 100% |

### 2.2 Lightning Attention Layer (`lightning`, 24 layers)

```
Input: hidden_states [B×T, 4096], residual, positions, forward_batch

Step 1: INPUT LAYERNORM + RESIDUAL
├─ Same as sparse layer (RMSNorm)

Step 2: QKV PROJECTION (MATMUL)
├─ qkv, _ = qkv_proj(hidden_states)
│  └─ Op: Linear(4096 → 3072), GPTQ INT4
│  └─ lightning_nh=16, head_dim=64, kv_heads=16
│  └─ Q: 16×64=1024, K: 16×64=1024, V: 16×64=1024
│  └─ FLOPs: 2 × 4096 × 3072 ≈ 25M FLOPs/token
│  └─ W4A8 eligible: ❌ NO (stays on Marlin BF16)

Step 3: Q/K NORM + ROPE (fused if enabled)
├─ q, k, v = _apply_qk_norm_rope(qkv, positions)
│  ├─ q_norm: RMSNorm(64) per head
│  ├─ k_norm: RMSNorm(64) per head
│  └─ RoPE: rotary embedding
│  └─ Can use fused_qk_norm_rope CUDA kernel (enabled)

Step 4: RESHAPE FOR BACKEND
├─ Reshape to 4D: (B×T, heads, head_dim) → (1, B×T, heads, head_dim)

Step 5: LINEAR ATTENTION (GLA KERNEL - HEAVY)
├─ o = linear_attn_backend.forward(q, k, v, forward_batch, layer_id)
│
│  [PREFILL/EXTEND]:
│  └─ chunk_simple_gla(q, k, v, ...)
│     └─ FLA kernel: chunked GLA with chunk_size=64
│     └─ FLOPs: O(T × heads × head_dim²)
│
│  [DECODE]:
│  └─ fused_recurrent_simple_gla(q, k, v, ...)
│     └─ FLA kernel: recurrent GLA update
│     └─ FLOPs: O(heads × head_dim²) per token ≈ 16×64×64 ≈ 65K
│     └─ Much cheaper than sparse attention decode!

Step 6: OUTPUT EPILOGUE
├─ o = _apply_output_epilogue(o, hidden_states)
│  ├─ o_norm: RMSNorm(1024) if enabled
│  ├─ Gate: z = sigmoid(z_proj(hidden_states)), o *= z  (if enabled)
│  │  └─ z_proj: Linear(4096→1024), NOT W4A8 eligible
│  └─ o_proj: Linear(1024 → 4096), NOT W4A8 eligible
│
Step 7-9: MLP (same as sparse layer)
├─ gate_up_proj + SiLU + down_proj
└─ Same dimensions: 4096→28672→4096
   └─ W4A8 eligible: YES ✅ (MLP linears ARE tagged)
```

**Lightning layer FLOPs per token (decode)**:
| Op | FLOPs | % Total |
|----|-------|---------|
| Gate-Up matmul | 235M | 62% |
| Down matmul | 117M | 31% |
| QKV matmul | 25M | 7% |
| GLA compute | ~65K | <1% |
| O proj | 8M | 2% |
| **Total** | **~385M** | 100% |

---

## 3. W4A8 Fused Kernel Integration

### 3.1 Call Sites

The W4A8 fused kernel (`torch.ops.w4a8_fused.w4a8_fp8_fused_gemm`) is called from a **SINGLE** location:

**File**: `python/sglang/srt/layers/quantization/gptq.py`  
**Method**: `GPTQMarlinLinearMethod.apply()` (line ~951)

> **KEY**: This method intercepts EVERY GPTQ linear forward pass — QKV, O,
> Gate-Up, Down — across all 32 layers. Whether the fused kernel is actually
> used depends on `layer._soar_w4a8_real_active`, which is only set for
> `MiniCPMAttention` linears (8 layers) and `MiniCPMMLP` linears (all 32 layers).

**Activation flow**:
```
GPTQMarlinLinearMethod.apply(layer, x)
├─ if layer._soar_w4a8_real_active AND M >= 64:
│  ├─ Pad x to M%128==0
│  ├─ Convert x to FP8 e4m3
│  ├─ torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(
│  │    layer._w4a8_qweight,    # INT4 weights (original GPTQ format)
│  │    layer._w4a8_qzeros,     # INT4 zero points
│  │    layer._w4a8_scales,     # BF16 scales
│  │    x_fp8,                  # FP8 input
│  │    out_features, in_features, group_size)
│  └─ Unpad result, convert to input dtype
│
├─ elif layer._soar_w4a8_active:
│  └─ cutlass_w8a8_block_fp8_linear (old W8A8, doubled HBM)
│
└─ else:
   └─ Standard GPTQ Marlin (Marlin repack + cuBLAS GEMM)
```

### 3.2 Eligible Layers

> **Why only 8 layers for QKV/O?** `MiniCPMLightningMixer` does NOT set
> `_soar_w4a8_eligible` on its linears (see `minicpm.py` lines ~330-350).
> `--force-dense-minicpm` only affects attention routing, not module class.
> The 24 lightning layers always use `MiniCPMLightningMixer` regardless.
>
> **MLP linears** (gate_up_proj, down_proj) are tagged for ALL 32 layers
> because both layer types use the same `MiniCPMMLP` class.

| Layer Type | Projection | Shape | W4A8? | Status |
|------------|-----------|-------|-------|--------|
| **minicpm4 Attn (8 layers)** | QKV | 4096×6144 | ✅ | Kernel ready, M%128 issue |
| **minicpm4 Attn (8 layers)** | O | 4096×4096 | ✅ | Kernel ready, M%128 issue |
| **MLP (all 32 layers)** | Gate-Up | 4096×28672 | ✅ | Kernel ready, M%128 issue |
| **MLP (all 32 layers)** | Down | 14336×4096 | ✅ | Kernel ready, M%128 issue |
| Lightning Attn (24 layers) | QKV | 4096×3072 | ❌ | `MiniCPMLightningMixer` — not tagged |
| Lightning Attn (24 layers) | O | 1024×4096 | ❌ | `MiniCPMLightningMixer` — not tagged |
| Lightning Attn (24 layers) | Z (gate) | 4096×1024 | ❌ | Not tagged |

But MLP dimensions are SAME across all 32 layers, so the W4A8 kernel handles:
- **Gate-Up**: 4096×28672 — this is the single biggest op (54-62% of layer FLOPs)
- **Down**: 14336×4096
- **QKV (sparse only)**: 4096×6144
- **O (sparse only)**: 4096×4096

### 3.3 Current Blocker: M % 128 == 0

The fused FP8 MMA kernel requires `M % 128 == 0`:
- **Decode**: M=1-24 (batch sizes for CUDA graph) → kernel REFUSED, falls back to Marlin
- **Prefill**: M=prompt_length (typically 100-10000+) → kernel WORKS for most prefill batches

### 3.4 Padding Strategy

`gptq.py` already implements M-padding:
```python
M_pad = ((M_orig + 127) // 128) * 128  # Round up to 128
if M_pad != M_orig:
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, M_pad - M_orig))
```
But the kernel's `TORCH_CHECK(M % kTileM == 0)` REJECTS small M before padding can even be applied (the check is inside the kernel, and the kernel requires M ≥ 64).

---

## 4. Op Inventory & Optimization Targets

### 4.1 Matmul Ops (Ranked by FLOPs Impact)

| Rank | Op | FLOPs/token | % Total | W4A8? | Optimization |
|------|-----|-------------|---------|-------|--------------|
| **1** | **Gate-Up MLP** | 235M | 54-62% | ✅ | FP8 `mma.sync` 296 TFLOPS (current: BF16 148 TFLOPS) |
| **2** | **Down MLP** | 117M | 27-31% | ✅ | Same as above |
| 3 | QKV Sparse | 50M | 11% | ✅ | Small op, marginal gain |
| 4 | O Sparse | 33M | 8% | ✅ | Small op |
| 5 | QKV Lightning | 25M | 7% | ❌ | Marlin BF16 only |
| 6 | O Lightning | 8M | 2% | ❌ | Negligible |

**Key insight**: The Gate-Up MLP projection alone accounts for >50% of all FLOPs. Optimizing just this ONE op to FP8 `mma.sync` (296 TFLOPS) would give the largest speedup.

### 4.2 Attention Ops

| Op | Kernel | Prefill Cost | Decode Cost | Optimizable? |
|----|--------|-------------|-------------|--------------|
| Sparse attn (prefill) | FlashAttention v3 | O(T²d) | N/A | (already fast) |
| Sparse attn (decode) | FA3 decode | O(T_cache × d) | ~128K FLOPs | Memory-bound, not compute-bound |
| Lightning attn (prefill) | `chunk_simple_gla` (FLA) | O(Td²) | N/A | FLA kernel (C++/CUDA) |
| Lightning attn (decode) | `fused_recurrent_simple_gla` | N/A | ~65K FLOPs | Very cheap |

**Key insight**: Attention ops are **memory-bound** or very cheap. They are NOT the bottleneck for speed optimization at current stage.

### 4.3 Norm & Activation Ops

| Op | Cost | Optimizable? |
|----|------|-------------|
| RMSNorm (×2 per layer) | ~32K FLOPs each | Negligible, already fused |
| SiLU+Mul (SiluAndMul) | ~29K FLOPs | Fused, negligible |
| RoPE (Q & K) | ~1M FLOPs | `fused_qk_norm_rope` already enabled |
| Q/K Norm (lightning) | ~1K FLOPs | Fused in `fused_qk_norm_rope` |
| Residual scale | 1 multiply | Negligible |

### 4.4 Fused Kernels Already Active

| Kernel | Layer Type | Enabled? | Impact |
|--------|-----------|----------|--------|
| `fused_qk_norm_rope` | Both | ✅ Yes | Fuses Q/K norm + RoPE, reduces kernel launches |
| `lightning_fast_state_io` | Lightning | ✅ Yes | Fast recurrent state save/load |
| `lightning_fast_output_gate` | Lightning | ✅ Yes | Fused output gate + sigmoid |
| Marlin GPTQ | All linears | ✅ Yes (baseline) | 148 TFLOPS INT4 GEMM |
| W4A8 FP8 fused | Sparse attn + MLP | 🔧 WIP | 296 TFLOPS target |

---

## 5. Optimization Strategy Recommendation

### 5.1 Phase 1: Get W4A8 Fused Kernel Working (Current)

**Approach**: Fix M%128 tile restriction in `w4a8_fp8_qmma.cu`:

Option A: **Add boundary masking** — Process full 128×128 tiles with masking for partial M
Option B: **Add smaller tile dispatch** — Use m16n8k32 with padding for M < 128
Option C: **Hybrid dispatch** — Use fused kernel for M ≥ 128 (prefill), Marlin for M < 128 (decode)

**Estimated gain**: 296/148 = **2× on matmul ops** → ~1.5× overall speedup on decode, ~1.8× on prefill

### 5.2 Phase 2: Optimize Attention (If Needed)

- Lightning attention recurrent kernel is already very fast (<1% FLOPs)
- Sparse attention is memory-bound; FP8 KV cache already used
- FA3 is already optimal

### 5.3 Phase 3: Scheduling / Batching Optimizations

- Already using `--schedule-conservativeness 0.8`, chunked prefill
- CUDA graph capture covers bs 1-24

---

## 6. Appendix: Data Flow Diagram

```
                    ┌──────────────────────────────┐
                    │     Token Embedding           │
                    │  (scale_emb × embed_tokens)   │
                    └──────────┬───────────────────┘
                               │ [B×T, 4096] BF16
                    ┌──────────▼───────────────────┐
                    │   FOR layer_id = 0..31:       │
                    │                               │
                    │  ┌─ RMSNorm ─────────────────┐│
                    │  │                             ││
                    │  │  IF mixer_type == minicpm4: ││
                    │  │    W4A8: QKV [4096→6144]    ││
                    │  │    RoPE (fused qk_norm)     ││
                    │  │    FA3 Sparse Attn           ││
                    │  │    W4A8: O   [4096→4096]    ││
                    │  │                             ││
                    │  │  ELSE (lightning):           ││
                    │  │    QKV [4096→3072] Marlin    ││
                    │  │    Fused QK Norm + RoPE      ││
                    │  │    FLA chunk/recurrent GLA   ││
                    │  │    O proj + gate (Marlin)    ││
                    │  └─────────────────────────────┘│
                    │                                 │
                    │  ┌─ RMSNorm + residual ────────┐│
                    │  │   W4A8: Gate-Up [4096→28672]││ ← HEAVIEST (54%)
                    │  │   SiLU × Up (fused)          ││
                    │  │   W4A8: Down   [14336→4096] ││ ← 2ND HEAVIEST (27%)
                    │  │   × residual_scale           ││
                    │  └─────────────────────────────┘│
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   Final RMSNorm + LM Head     │
                    │   (if not tied embeddings)     │
                    └──────────────────────────────┘
```

---

## 7. Dense vs Sparse Attention — Kernel↔Formula Mapping

> **This section clarifies Q2/Q3**: In the 8 `minicpm4` layers, `--force-dense-minicpm`
> changes HOW attention is computed, but QKV/O projections are IDENTICAL in both modes.

### 7.1 Dense Mode (current with `--force-dense-minicpm`)

```
QKV → Split → RoPE → FA3 Dense Attention → O

Q = x @ W_Q  (4096 → 4096, 50M FLOPs)   ← W4A8 ✅
K = x @ W_K  (4096 → 1024)               ← combined in QKV
V = x @ W_V  (4096 → 1024)               ← combined in QKV

RoPE: q' = q·cos(θ·pos) + rotate_half(q)·sin(θ·pos)
            θ_i = base^{-2i/d}, base=10000

Attention (FlashAttention-3):
  S = Q @ K^T / √d           scores: (1, num_heads, cache_len)
  P = softmax(S)              row-wise softmax
  O = P @ V                   weighted sum: (1, num_heads, head_dim)
  
  FLOPs: 4 × num_heads × cache_len × head_dim
       = 4 × 32 × cache_len × 128
         cache_len=100K → 1.6G FLOPs (memory-bound, bottleneck is KV cache load)
```

**Kernel call chain**:
```
RadixAttention.forward()  [radix_attention.py:95]
└─ MiniCPMBackend.forward_decode()  [minicpm_backend.py:1161]
   └─ flashinfer.batch_decode_with_padded_kv_cache()  OR
   └─ flash_attn_3_varlen_func()                      [GPU kernel]
      └─ Formula: softmax(Q·K^T/√d)·V
```

### 7.2 Sparse Mode (without `--force-dense-minicpm`, when seq_len ≥ dense_len)

The sparse path adds THREE stages of pre-processing before sparse attention:

```
QKV → Split → RoPE → [Stage1: Compress Keys] → [Stage2: TopK] → [Stage3: Sparse FA] → O
```

**Stage 1: Key Compression** (`compress_k_to_scratch_kernel`, Triton kernel)
```
K1 = avg_pool(K, kernel_size=32, stride=16)   // K: (cache_len, heads, dim) → (cache_len/16, heads, dim)
K2 = avg_pool(K, kernel_size=128, stride=64)  // → (cache_len/64, heads, dim)
```

**Stage 2: TopK Selection** (`compressed_attention`, max_pool_1d_varlen)
```
S1 = Q @ K1^T / √d               // coarse scores: (1, num_heads, cache_len/16)
S2 = Q @ K2^T / √d               // fine scores: (1, num_heads, cache_len/64)
block_scores = max_pool_1d(S1)   // pool over windows
topk_blocks = topk(block_scores, k=sparse_topk)  // k=8 blocks
// Each block = sparse_block_size=32 tokens → 256 sparse tokens per head
```

**Stage 3: Sparse FlashAttention** (`sparse_kernel_extension` or `flashinfer`)
```
K_sparse = gather(K, topk_blocks)     // (1, num_heads, 256) sparse tokens
V_sparse = gather(V, topk_blocks)
S = Q @ K_sparse^T / √d               // sparse scores
P = softmax(S)
O = P @ V_sparse                      // sparse output

FLOPs: 4 × 32 × 256 × 128 ≈ 4.2M (vs 1.6G dense = 380× less compute!)
```

**Kernel call chain (sparse mode)**:
```
RadixAttention.forward()  [radix_attention.py:95]
└─ MiniCPMBackend.forward_decode()  [minicpm_backend.py:1161]
   ├─ get_topk_for_sparse()  [minicpm_backend.py:872]
   │  ├─ compress_k_to_scratch_kernel()        [Triton kernel]
   │  └─ compressed_attention()                 [minicpm_sparse_utils.py:498]
   │     └─ max_pooling_1d_varlen() + topk()   [infllmv2 kernel]
   └─ AttentionParams → flashinfer.batch_decode()  [GPU kernel]
```

**Comparison: Dense vs Sparse (per decode token, cache_len=100K)**:

| Component | Dense (FA3) | Sparse (TopK) | Ratio |
|-----------|------------|---------------|-------|
| **Attention formula** | softmax(Q·K^T/√d)·V | masked softmax on top-k blocks | — |
| **K,V tokens processed** | 100K (all) | ~256 (sparse topk) | 390× less |
| **Attention FLOPs** | 1.6G | 4.2M | 380× |
| **Extra pre-processing** | 0 | Key compression + TopK (~2M FLOPs) | — |
| **Memory access** | Full KV cache read | Sparse KV gather | ~100× less BW |
| **QKV/O projections** | IDENTICAL (50M + 33M) | IDENTICAL (50M + 33M) | 1× |

> **Key insight**: QKV/O projections are **unchanged** between dense and sparse mode.
> Same `MiniCPMAttention` class, same `qkv_proj`/`o_proj` calls, same dimensions.
> Only the attention COMPUTE step changes (which kernel is called for the Q·K·V operation).

---

## 8. Full Call Chain — Top to Bottom

```
MiniCPMForCausalLM.forward()           [minicpm.py:745]
└─ MiniCPMModel.forward()              [minicpm.py:688]
   ├─ embed_tokens(input_ids) × scale_emb   [minicpm.py:709]
   └─ for layer_id in 0..31:
      └─ MiniCPMDecoderLayer.forward() [minicpm.py:626]
         │
         ├─ input_layernorm(hidden_states, residual)  [RMSNorm, minicpm.py:636]
         │
         ├─ [IF mixer_type == "minicpm4"]: MiniCPMAttention.forward()  [minicpm.py:267]
         │  │
         │  ├─ qkv_proj(hidden_states)         [minicpm.py:267]
         │  │  └─ GPTQMarlinLinearMethod.apply()  [gptq.py:951]
         │  │     ├─ [IF SOAR_W4A8_REAL_FP8_GEMM=1] → torch.ops.w4a8_fused.w4a8_fp8_fused_gemm()
         │  │     └─ [ELSE] → gptq_marlin_repack() + marlin_gemm()
         │  │        Formula: qkv = W_qkv @ x + b     [D×3·D_out matmul]
         │  │
         │  ├─ q, k, v = qkv.split([4096,1024,1024])  [minicpm.py:268]
         │  │
         │  ├─ q, k = rotary_emb(positions, q, k)     [minicpm.py:270]
         │  │  └─ MRotaryEmbedding (or fused_qk_norm_rope CUDA kernel)
         │  │     Formula: q'_i = q_i·cos(θ·pos) + rotate_half(q_i)·sin(θ·pos)
         │  │
         │  ├─ attn_output = attn(q, k, v, forward_batch)  [minicpm.py:273]
         │  │  └─ RadixAttention.forward()    [radix_attention.py:95]
         │  │     └─ MiniCPMBackend.forward_extend/decode()  [minicpm_backend.py]
         │  │        [DENSE mode]:
         │  │        └─ flashinfer.batch_decode_with_padded_kv_cache()
         │  │           Formula: softmax(Q·K^T/√d)·V
         │  │        [SPARSE mode]:
         │  │        ├─ compress_k_to_scratch_kernel()   [Triton kernel]
         │  │        ├─ compressed_attention() → topk    [minicpm_sparse_utils.py]
         │  │        └─ flashinfer.batch_decode()         [GPU kernel]
         │  │
         │  └─ o_proj(attn_output)           [minicpm.py:279]
         │     └─ GPTQMarlinLinearMethod.apply()  [gptq.py:951]
         │        Formula: out = W_o @ attn_output + b   [D×D matmul]
         │
         ├─ [ELIF mixer_type == "lightning"]: MiniCPMLightningMixer.forward()  [minicpm.py:476]
         │  │
         │  ├─ qkv_proj(hidden_states)        [minicpm.py:488]
         │  ├─ q, k = q_norm(q), k_norm(k)    [minicpm.py:511]
         │  ├─ q, k = rotary_emb(pos, q, k)   [minicpm.py:512]
         │  ├─ SimpleGLAAttnBackend.forward()  [hybrid_linear_attn_backend.py:1724]
         │  │  [DECODE]: fused_recurrent_simple_gla(q,k,v,...)
         │  │     Formula: h_t = λ·h_{t-1} + k_t⊗v_t, o_t = q_t·h_t
         │  │  [PREFILL]: chunk_simple_gla(q,k,v,...)
         │  └─ o_proj(o) + z_proj(gate)       [minicpm.py:518-521]
         │
         ├─ hidden_states *= residual_scale   [minicpm.py:648]
         │
         ├─ post_attention_layernorm(h, residual)  [RMSNorm, minicpm.py:651]
         │
         ├─ mlp(hidden_states)                [MiniCPMMLP, minicpm.py:162]
         │  ├─ gate_up_proj(h)                [gptq.py:951]
         │  │  Formula: [gate|up] = W_gu @ h     [D×2·D_int matmul, 235M FLOPs]
         │  ├─ SiluAndMul(gate_up)            [activation.py]
         │  │  Formula: out = up ⊙ silu(gate)
         │  └─ down_proj(out)                 [gptq.py:951]
         │     Formula: out = W_d @ out           [D_int×D matmul, 117M FLOPs]
         │
         └─ hidden_states *= residual_scale   [minicpm.py:654]

└─ self.norm(hidden_states)                   [RMSNorm, minicpm.py:713]
└─ [IF lm_head]: lm_head(hidden_states)       [Linear]
```

---

## 9. FLOPs Calculation Methodology (Decode, M=1)

### 9.1 Matmul FLOPs

Standard formula: **FLOPs = 2 × M × K × N** (one multiply + one add per element)

| Projection | Shape (M×K×N) | FLOPs Formula | Result |
|-----------|---------------|---------------|--------|
| QKV Sparse | 1 × 4096 × 6144 | 2 × 1 × 4096 × 6144 | **50.3M** |
| QKV Lightning | 1 × 4096 × 3072 | 2 × 1 × 4096 × 3072 | **25.2M** |
| O Sparse | 1 × 4096 × 4096 | 2 × 1 × 4096 × 4096 | **33.6M** |
| Gate-Up | 1 × 4096 × 28672 | 2 × 1 × 4096 × 28672 | **234.9M** |
| Down | 1 × 14336 × 4096 | 2 × 1 × 14336 × 4096 | **117.4M** |

Note: GPTQ INT4 matmuls have same FLOPs count as FP16 — quantization reduces memory, not compute.

### 9.2 Attention FLOPs (decode, M=1)

**Dense FA3**: `4 × num_heads × cache_len × head_dim`

With cache_len=100K: 4 × 32 × 100,000 × 128 ≈ **1.64G FLOPs**
But this is **memory-bound** — actual wall-clock dominated by KV cache load (32 × 100,000 × 128 × 1 byte ≈ 400 MB). At 1398 GB/s bandwidth, theoretical minimum = 400MB/1398GB/s ≈ 0.3ms. FLOPs aren't the bottleneck.

**Sparse TopK**: `4 × num_heads × sparse_tokens × head_dim + key_compression`

With sparse_tokens=256: 4 × 32 × 256 × 128 ≈ **4.2M FLOPs**
Plus key compression: ~2M FLOPs. Total ≈ 6M FLOPs.

**GLA Recurrent**: `heads × (head_dim² + head_dim)` ≈ 16 × (64² + 64) ≈ **66K FLOPs**

### 9.3 Percentage Breakdown (Sparse Layer Decode)

| Op | Raw FLOPs | Adjusted* | Percentage |
|----|-----------|-----------|------------|
| Gate-Up matmul | 235M | 235M | **54%** |
| Down matmul | 117M | 117M | **27%** |
| QKV matmul | 50M | 50M | **11%** |
| O matmul | 34M | 34M | **8%** |
| Attention | ~0.1M–1.6G | ~0.3ms (BW-bound) | **<1% of time** |
| **Total matmul** | **436M** | — | **100%** |

*\*Adjusted: attention is counted at 0 for FLOPs percentage because it's memory-bound, not compute-bound. The matmuls dominate compute time.*

---

## 10. File Reference Index

| Component | File | Key Lines |
|-----------|------|-----------|
| Model definition | `python/sglang/srt/models/minicpm.py` | L132-L288 (attention/MLP), L525-L654 (decoder layer) |
| GPTQ quantization | `python/sglang/srt/layers/quantization/gptq.py` | L880-L970 (W4A8 setup), L951 (apply dispatch) |
| Dense attention backend | `python/sglang/srt/layers/attention/minicpm_backend.py` | L150-L210 (init), L942-L1050 (prefill), L1161-L1300 (decode) |
| Sparse attention utils | `python/sglang/srt/layers/attention/minicpm_sparse_utils.py` | L498 (compressed_attention) |
| Lightning attn backend | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | L1445-L1810 (SimpleGLAAttnBackend) |
| RadixAttention dispatch | `python/sglang/srt/layers/radix_attention.py` | L95-L133 |
| Model config | `python/sglang/srt/configs/model_config.py` | L107 (force_dense), L238 (has_sparse), L248 (sparse_ids) |
| Server args | `python/sglang/srt/server_args.py` | L542 (force_dense_minicpm) |
| Fused kernel | `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu` | L1-L176 (SM120 warp-level FP8 mma.sync) |
| prepare_env.sh | `benchmark/soar/demo_sala/prepare_env.sh` | All env vars and server args |
