# CHANGE_0120: 性能分析结果与内核分解分析

## 概述

**日期**: 2026-04-20  
**提交**: 08fd86023 (CHANGE_0120 配置)  
**配置**: GPTQ + FP8 KV + 稠密模式 + torch.compile(max-bs=8) + mixed-chunk + prefill-max-req=4 + sched-cons=0.8 + chunk=65536  
**fcloud实例**: 223.167.85.181  
**目的**: 分析前向传播内核分解，确定优化目标

---

## 分析方法

### 使用的命令

**步骤1: 启动分析（按阶段分离，每阶段3步）**
```python
# 通过fcloud上的HTTP API
requests.post("http://localhost:30000/start_profile", json={
    "output_dir": "/tmp/minicpm_profile",
    "num_steps": 3,
    "profile_by_stage": True,
    "activities": ["CPU", "GPU"],
    "with_stack": True,
    "record_shapes": True
})
```

**步骤2: 发送推理请求**
- 短上下文运行：5个MCQ样本（平均127 token输入）
- 长上下文运行：1个NIAH（30K token）+ 1个QA（25K token），并发=1

**步骤3: 分析跟踪数据**
```bash
python3 /tmp/analyze_profile.py /tmp/minicpm_profile
```

### 输出文件
| 文件 | 大小 | 上下文 | 描述 |
|------|------|--------|------|
| `1776651754-TP-0-EXTEND.trace.json.gz` | 598K | 短（127 tok） | 短上下文prefill |
| `1776651754-TP-0-DECODE.trace.json.gz` | 155K | 短（127 tok） | 短上下文decode |
| `1776652067-TP-0-EXTEND.trace.json.gz` | 838K | 长（25K-30K tok） | **长上下文prefill** |
| `1776652067-TP-0-DECODE.trace.json.gz` | 153K | 长（25K-30K tok） | 长上下文decode |

---

## 结果：长上下文Prefill（25K-30K tokens）— 关键跟踪数据

**GPU内核总时间: 4989.6 ms**（3次prefill前向传播）

### 分类分解

| 类别 | 时间 (ms) | 占比 | 描述 |
|------|----------|------|------|
| **GEMM (Marlin/GPTQ)** | **4257.8** | **85.3%** | GPTQ W4A16 反量化+GEMM |
| **FLA/SimpleGLA** | **620.1** | **12.4%** | FLA块内核 + FlashInfer注意力 |
| 其他 | 87.6 | 1.8% | 类型转换、sigmoid、索引 |
| RMSNorm | 23.4 | 0.5% | 融合QK-norm-RoPE |

### 前10内核（长上下文Prefill）

| 排名 | 占比 | 时间(ms) | 调用次数 | 内核 | 类别 |
|------|------|---------|---------|------|------|
| 1 | 80.8% | 4029.7 | 2056x | `Marlin<bf16, ..., 128, 4, 4, 8>` | GPTQ GEMM（主） |
| 2 | 8.8% | 439.8 | 8x | `BatchPrefillWithRaggedKVCacheKernel` | FlashInfer注意力（8标准层） |
| 3 | 3.2% | 161.9 | 32x | `cutlass_tensorop_bf16_gemm_relu` | Gate GEMM（MLP） |
| 4 | 1.5% | 72.5 | 32x | `act_and_mul_kernel<bf16, silu>` | SiLU激活 |
| 5 | 1.3% | 66.2 | 248x | `Marlin<bf16>`（变体） | GPTQ GEMM（8位层） |
| 6 | 1.0% | 49.3 | 64x | `FusedAddRMSNormKernel` | 融合残差+归一化 |
| 7 | 0.6% | 28.0 | 24x | `chunk_fwd_kernel_o` | FLA块输出内核 |
| 8 | 0.5% | 23.4 | 24x | `fusedQKNormRopeKernel` | 融合QK-norm + RoPE |
| 9 | 0.4% | 22.0 | 24x | `chunk_fwd_kernel_h` | FLA块隐藏状态内核 |

---

## 结果：Decode（每步单token）

**GPU内核总时间: 18.5 ms**（3次decode步骤）

### 分类分解

| 类别 | 时间 (ms) | 占比 | 描述 |
|------|----------|------|------|
| **GEMM (Marlin/GPTQ)** | **11.7** | **63.5%** | GPTQ W4A16 反量化+GEMM |
| **其他（torch.compile融合）** | **5.4** | **29.4%** | 融合GEMM+激活、RMSNorm、状态I/O |
| **FLA/SimpleGLA** | **0.9** | **4.9%** | fused_recurrent_fwd + FlashInfer分页KV |
| RMSNorm | 0.3 | 1.5% | fusedQKNormRopeKernel |
| 激活 (SiLU) | 0.1 | 0.5% | triton融合SiLU |

### Decode中torch.compile融合内核详情

| 占比 | 时间(ms) | 调用次数 | 推测功能 | 内核名 |
|------|---------|---------|----------|--------|
| 10.5% | 1.9 | 72x | GEMM+RMSNorm+sigmoid（输出门） | `triton_red_fused__to_copy_add_mean_mm_mul_pow_rsqrt_sigmoid_t_1` |
| 7.3% | 1.3 | 3x | GEMM+permute（lm_head?） | `triton_red_fused_div_mm_permute_0` |
| 3.3% | 0.6 | 23x | GEMM+sigmoid | `triton_red_fused_mm_mul_sigmoid_t_0` |
| 2.2% | 0.4 | 194x | RMSNorm（融合） | `triton_red_fused__to_copy_add_*` |
| 1.7% | 0.3 | 74x | 索引（状态加载） | `index_elementwise_kernel` |
| 1.3% | 0.2 | 75x | 索引（状态存储） | `index_put_kernel` |

---

## 决策分析

### 数据告诉我们什么

**最大瓶颈是GEMM（Marlin/GPTQ），占prefill时间的85%。**

对于新竞赛数据集（68%的输入为32K-512K token），prefill时间占比更大（更多GEMM）。这意味着：

1. **FLA内核优化影响有限** — 仅占prefill的12.4%。即使FLA内核提速50%，总体prefill也只提速~6%。

2. **Marlin GEMM优化影响最大** — 占prefill的85.3%。提速10%即可获得8.5%的总体提速。

3. **FP8权重量化可能带来变革** — 从W4A16 Marlin切换到FP8 W8A16：
   - 使用原生FP8张量核而非反量化+FP16 GEMM
   - FP8张量核吞吐量约为FP16的2倍
   - 但权重大小翻倍（8位 vs 4位），需要更多内存带宽
   - 最终效果取决于这些序列长度下GEMM是计算瓶颈还是内存带宽瓶颈

### 修正后的优先级

| 优先级 | 路径 | 预期影响 | 理由 |
|--------|------|---------|------|
| **1** | **Marlin GEMM分析与调优** | **5-15%** | 占prefill 85.3%的时间 |
| **2** | **FP8权重量化 (W8A16)** | **10-30%** | 用原生FP8张量核替代W4反量化 |
| **3** | **FlashInfer注意力优化** | **5-8%** | 占prefill 8.8%（8标准注意力层） |
| **4** | **FLA块内核优化** | **1-3%** | 降级，仅占prefill 1% |
| **5** | **状态连续性A1** | **<1%** | prefill中可忽略 |
| **6** | **K4融合RMSNorm** | **<0.5%** | RMSNorm已可忽略 |

### 为什么FLA优化被降级

之前的优化目录假设"24个SimpleGLA层 = 前向传播时间的75%"。**分析数据否定了这一假设。** 在prefill阶段：
- SimpleGLA层总贡献~12.4%（FLA块 + fused_recurrent + 状态I/O）
- 其余~87.6%是GEMM + 注意力 + 其他
- 每个SimpleGLA层的FLA内核每步仅~2ms，而GEMM每步~50ms

"75%前向传播时间"的估计是针对decode为主的工作负载。新的长上下文数据集以prefill为主，**GEMM才是真正的瓶颈。**

---

## 建议的下一步

1. **分析Marlin GEMM在Blackwell SM120上的占用率** — 自动配置是否选择了最优瓦片尺寸？
2. **调研FP8权重量化 (W8A16)** — sglang原生支持`--quantization fp8`
3. **提交v19** — 当前配置先提交

---

## 附录：分析脚本

分析脚本位于：
- `scripts/fcloud/analyze_profile.py` — 主分类分析
- `scripts/fcloud/analyze_profile_other.py` — "其他"类别详细分解
