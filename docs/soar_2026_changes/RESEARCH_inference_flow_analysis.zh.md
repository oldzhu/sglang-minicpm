# MiniCPM-SALA 推理流程分析
## SOAR 2026 — 内核优化路线图

**日期**: 2026-05-25 | **模型**: MiniCPM-SALA-90B (GPTQ INT4, dense mode)  
**基线配置**: GPTQ (sparse_qkv_w8) + FP8 KV cache + `--force-dense-minicpm`

---

## 1. 模型架构总览

### 1.1 全局参数

| 参数 | 值 |
|------|-----|
| `num_hidden_layers`（总层数） | **32** |
| `hidden_size`（隐藏维度 D） | **4096** |
| `num_attention_heads`（Q 头数） | **32** |
| `num_key_value_heads`（KV 头数） | **8**（GQA 比率 = 4:1） |
| `head_dim`（每头维度） | **128**（= 4096/32） |
| `intermediate_size`（MLP 中间层） | **14336** |
| `vocab_size`（词表大小） | **150528** |

### 1.2 层类型分布

> **关键澄清**: `--force-dense-minicpm` **不会**把 lightning 层转换为标准 attention。
> 它仅设置 `model_config.has_sparse_attention=False` 和 `sparse_layer_ids=[]`，
> 影响的是 `MiniCPMAttention` 内部的 attention 路由（sparse→dense FA3）。
> 层的 MODULE CLASS 由 `config.mixer_types[layer_id]` 在初始化时决定，
> 完全独立于 `--force-dense-minicpm`。
> 24 个 lightning 层始终使用 `MiniCPMLightningMixer`（GLA 循环注意力）。

| Mixer 类型 | 数量 | 模块类 | KV Cache? | 当前 Attention 类型 |
|------------|------|--------|-----------|---------------------|
| `minicpm4` | **8 层** | `MiniCPMAttention` | ✅ Paged KV | Dense FA3（因 force_dense） |
| `lightning` | **24 层** | `MiniCPMLightningMixer` | ❌ 循环状态 | GLA chunk/recurrent |

### 1.3 Lightning 层子配置

| 参数 | 值 |
|------|-----|
| `lightning_nh`（heads） | **16** |
| `lightning_nkv`（KV heads） | **16** |
| `lightning_head_dim` | **64** |

---

## 2. 关键澄清：QKV/O 的 "sparse only" 问题

### 2.1 `--force-dense-minicpm` 做了什么？

`--force-dense-minicpm` 仅影响以下两处（见 `model_config.py` line 238-248）：
```python
@property
def has_sparse_attention(self):
    return getattr(self.hf_config, "has_sparse_attention", False) \
           if not self.force_dense_minicpm else False  # ← 强制返回 False

@property
def sparse_layer_ids(self):
    return getattr(self.hf_config, "sparse_layer_ids", []) \
           if not self.force_dense_minicpm else []   # ← 强制返回 []
```

它**不改变** `MiniCPMDecoderLayer.__init__` 中的模块类选择逻辑：
```python
if self.mixer_type == "minicpm4":
    self.self_attn = MiniCPMAttention(...)        # ← 8 层
elif self.mixer_type in ["lightning", ...]:
    self.self_attn = MiniCPMLightningMixer(...)    # ← 24 层
```

### 2.2 为什么 QKV/O 仅 sparse 层可用 W4A8？

- `MiniCPMAttention.__init__` 设置 `self.qkv_proj._soar_w4a8_eligible = True`（line 235-236）
- `MiniCPMLightningMixer.__init__` **不设置**此标记（line ~330-350）
- `--force-dense-minicpm` 不改变模块类 → 24 个 lightning 层仍使用 `MiniCPMLightningMixer` → QKV/O 不标记

### 2.3 Sparse vs Lightning — QKV/O 维度差异

| | Sparse Attn（8 层） | Lightning Attn（24 层） |
|---|---|---|
| **模块类** | `MiniCPMAttention` | `MiniCPMLightningMixer` |
| **Q heads × dim** | 32 × 128 = **4096** | 16 × **64** = **1024** |
| **KV heads × dim** | 8 × 128 = **1024** | 16 × 64 = **1024** |
| **QKV proj 形状** | 4096 → **6144** | 4096 → **3072** |
| **O proj 形状** | **4096** → 4096 | **1024** → 4096 |
| **W4A8 可用?** | ✅ Yes | ❌ No（未标记） |
| **Attention 内核** | FA3（dense） | FLA GLA chunk/recurrent |
| **KV Cache?** | ✅ Paged KV（FP8 e5m2） | ❌ 循环状态 |

### 2.4 MLP 线形层不受影响

MLP 线形层（gate_up_proj, down_proj）在所有 32 层使用相同的 `MiniCPMMLP` 类，
该类在 line 151-152 设置 `_soar_w4a8_eligible = True`。所以 MLP 的 W4A8 覆盖全部 32 层。

---

## 3. 逐层前向传播与操作清单

### 3.1 Sparse Attention 层前向传播

```
输入: hidden_states [B×T, 4096]

├─ RMSNorm (32K FLOPs)
├─ QKV: Linear(4096→6144) = 50M FLOPs  ← W4A8 ✅
├─ RoPE (~1M FLOPs, fused)
├─ Attention: FA3 dense decode (~128K FLOPs, 内存受限)
├─ O: Linear(4096→4096) = 33M FLOPs     ← W4A8 ✅
├─ Gate-Up: Linear(4096→28672) = 235M   ← W4A8 ✅  **最大操作**
├─ SiLU × Up (融合)
├─ Down: Linear(14336→4096) = 117M      ← W4A8 ✅
└─ × residual_scale
```

**Sparse 层 FLOPs 占比（decode）**: Gate-Up 54% > Down 27% > QKV 11% > O 8%

### 3.2 Lightning Attention 层前向传播

```
输入: hidden_states [B×T, 4096]

├─ RMSNorm
├─ QKV: Linear(4096→3072) = 25M FLOPs   ← Marlin BF16 only ❌
├─ QK Norm + RoPE (fused)
├─ GLA recurrent decode (~65K FLOPs)     ← 极快
├─ O proj + gate
├─ Gate-Up: Linear(4096→28672) = 235M   ← W4A8 ✅  **最大操作**
├─ SiLU × Up (融合)
├─ Down: Linear(14336→4096) = 117M      ← W4A8 ✅
└─ × residual_scale
```

**Lightning 层 FLOPs 占比（decode）**: Gate-Up 62% > Down 31% > QKV 7% > O 2%

---

## 4. 优化目标优先级

| 排名 | 操作 | FLOPs/token | 覆盖范围 | W4A8? | 预估收益 |
|------|------|-------------|----------|-------|----------|
| **1** | Gate-Up MLP | 235M | 全部 32 层 | ✅ | 2× 矩阵乘法加速 |
| **2** | Down MLP | 117M | 全部 32 层 | ✅ | 2× 矩阵乘法加速 |
| 3 | QKV (sparse) | 50M | 8 层 | ✅ | 小 |
| 4 | O (sparse) | 33M | 8 层 | ✅ | 小 |

### 4.1 当前已启用的融合内核

| 内核 | 作用 | 状态 |
|------|------|------|
| `fused_qk_norm_rope` | Q/K Norm + RoPE 融合 | ✅ 已启用 |
| `lightning_fast_state_io` | 循环状态快速 I/O | ✅ 已启用 |
| `lightning_fast_output_gate` | 输出门融合 | ✅ 已启用 |
| GPTQ Marlin | INT4 GEMM（148 TFLOPS） | ✅ 基线 |
| W4A8 FP8 fused | INT4+FP8 GEMM（296 TFLOPS） | 🔧 M%128 限制待修复 |

### 4.2 下一步：修复 W4A8 融合内核

当前阻止融合内核工作的根因：kernel 的 `kTileM=128` 要求 `M % 128 == 0`，
但 decode 阶段 batch size 为 1-24。`gptq.py` 已有 M-padding 逻辑（填充到 128 的倍数），
但 kernel 内部还有 `M >= 64` 检查。

修复方案：
- 方案 A: 在 kernel 中添加边界掩码，支持部分 tile
- 方案 B: 使用更小的 MMA tile（m16n8k32），M≥16 即可
- 方案 C: M≥128 用融合 kernel（prefill），M<128 回退 Marlin（decode）
