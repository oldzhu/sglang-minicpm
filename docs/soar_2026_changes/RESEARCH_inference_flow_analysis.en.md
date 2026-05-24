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

| Mixer Type | Count | Class | KV Cache? | Attention Type |
|------------|-------|-------|-----------|----------------|
| `minicpm4` (sparse) | **8 layers** | `MiniCPMAttention` | ✅ Paged KV | Sparse/dense attention |
| `lightning` (linear) | **24 layers** | `MiniCPMLightningMixer` | ❌ Recurrent state | Linear (GLA) attention |

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

| Layer Type | Projection | Shape | W4A8? | Status |
|------------|-----------|-------|-------|--------|
| **Sparse Attn** | QKV | 4096×6144 | ✅ | Kernel ready, M%128 issue |
| **Sparse Attn** | O | 4096×4096 | ✅ | Kernel ready, M%128 issue |
| **Sparse MLP** | Gate-Up | 4096×28672 | ✅ | Kernel ready, M%128 issue |
| **Sparse MLP** | Down | 14336×4096 | ✅ | Kernel ready, M%128 issue |
| Lightning Attn | QKV | 4096×3072 | ❌ | Not tagged |
| Lightning Attn | O | 1024×4096 | ❌ | Not tagged |
| Lightning Attn | Z (gate) | 4096×1024 | ❌ | Not tagged |
| Lightning MLP | Gate-Up | 4096×28672 | ✅ | Same as sparse |
| Lightning MLP | Down | 14336×4096 | ✅ | Same as sparse |

**Total W4A8-eligible projections**: 8 (sparse attn) + 2 (MLP × 24 lightning) + 2 (MLP × 8 sparse) = **4 per attention layer** × 8 sparse + **2 per MLP** × 32 all layers = 32 + 64 = 96 projections total.

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
