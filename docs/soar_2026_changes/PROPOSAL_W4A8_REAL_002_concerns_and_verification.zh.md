# PROPOSAL: 真 W4A8 — 解答 FP8 精度与旧路径回归的顾虑

**日期**: 2026-05-15 | **状态**: PROPOSAL → 验证中  
**前序**: `PROPOSAL_W4A8_REAL_001`（2026-04-27 提出，未实施）  
**关联**: `CHANGE_W4A8_001_iteration_002`（旧 W8A8 FP8 路径，提交 `7ce21c3f5`，已废弃）

## 背景

在决定投入真 W4A8（INT4 存储 + FP8 MMA）内核开发前，用户提出两项顾虑：

1. **FP8 精度风险**：FP8 (e4m3) 尾数位少于 BF16/FP16 —— 可能将 MiniCPM-SALA 的准确率拉到 97% norm 阈值以下（C=0）。
2. **旧路径的速度倒退**：之前的 W4A8 尝试（提交 `7ce21c3f5`）在 S1 上退步了 +118%（121.71→265.32s）。即使真 W4A8 能做到 25% 加速，绝对速度可能仍不如基线。

本文档从第一性原理和现有实证数据出发，对这两个顾虑给出分析。

---

## 顾虑 1：FP8 激活值的精度风险

### 问题

> "W4A8 意味着使用 FP8 激活值。FP8 的位数比 BF16/FP16 少。这不会损害准确率吗？"

### FP8 精度对比

| 数据类型 | 尾数位 | 指数位 | 动态范围 | 用途 |
|---|---:|---:|---|---|
| BF16 | 7 | 8 | ~1e-38 到 3.4e38 | 当前基线（Marlin W4A16） |
| FP8 e4m3 | 3 | 4 | ~1.5e-5 到 448 | 建议的激活数据类型 |

FP8 e4m3 只有 3 个尾数位，比 BF16 的 7 个少 4 位。

### 为什么风险可控

**1. 逐 token 缩放补偿精度。** 每个 token 的激活向量分配独立的 FP8 缩放因子（`amax × 2^(exponent) → scale`）。这是 vLLM、TRT-LLM 及 SGLang 自有 `cutlass_w8a8_fp8` 内核的标准做法。逐 token 缩放将动态范围有效扩展至 FP8 静态范围之外，对于典型 transformer 激活分布，量化误差很小。

**2. 仅 GEMM 输入为 FP8 —— 其余全部保持 BF16。** 残差路径、注意力 softmax、layer norm、SimpleGLA 递归状态均保持 BF16。仅线性层的矩阵乘法（`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）使用 FP8 激活值。

**3. 我们自己的实证数据是正向的。** 旧 W4A8 测试（提交 `7ce21c3f5`）在**权重和激活两端**都用了 FP8（W8A8 FP8 blockwise GEMM）。准确率为 **79.20% vs 基线 79.29%**（Δ −0.09pt，完全在噪声范围内）。真 W4A8 使用 **INT4 权重（与基线相同）+ FP8 激活值** —— 仅激活端从 BF16→FP8。精度风险**低于**已证明中性的 FP8×FP8 情况。

| 测试 | 权重 | 激活值 | GEMM | 准确率 | 相对 Test 12 的 Δ |
|---|---:|---:|---|---|---|
| Test 12（基线） | INT4（0.5 B/elem） | BF16 | Marlin BF16 MMA | 79.29% | — |
| 旧 W4A8（7ce21c3f5） | **FP8**（1.0 B/elem） | **FP8** | cutlass FP8 blockwise | 79.20% | −0.09pt |
| **真 W4A8（建议）** | INT4（0.5 B/elem） | **FP8** | 混合输入 FP8 QMMA | **待测** | 预期 ≤±0.5pt |

**4. 缓解计划。**
- **Phase 0 微基准测试**：在真实 MiniCPM 隐藏状态上做 FP8 激活量化 CPU 端测试；测量逐 token MSE 与 BF16 参考值的比较。
- **门控**：若逐 token MSE 超过 5e-3（保守阈值；旧 FP8×FP8 测试在往返容忍 5e-2），中止并回退至 Option B（INT8 激活）。
- 提交前在 fcloud 上做**完整准确率运行** —— C=1.0 否则放弃。

---

## 顾虑 2：旧 W4A8 速度倒退预示真 W4A8 也会慢

### 问题

> "旧 W4A8 路径（7ce21c3f5）的 S1 +118%、S8 +56%、Smax +30%。即使真 W4A8 比基线快 25%，旧的退步远大于此。真 W4A8 会最终比基线慢吗？"

### 旧路径为什么慢 —— 以及真 W4A8 为什么不同

旧路径是 **W8A8 FP8 blockwise**，不是 W4A8。关键缺陷是**加载时的权重上转型**：

```
旧路径（W8A8，7ce21c3f5）：
  磁盘上的 GPTQ 权重： INT4（0.5 字节/元素）
  → 反量化为 BF16
  → 重新量化为 FP8
  GPU HBM 中的权重：   FP8（1.0 字节/元素）  ← 相对 INT4 翻倍！
  激活值：             FP8（1.0 字节/元素）
  GEMM：               FP8 × FP8 cutlass blockwise @ 296 TF

真 W4A8（建议）：
  磁盘上的 GPTQ 权重： INT4（0.5 字节/元素）
  GPU HBM 中的权重：   INT4（0.5 字节/元素）  ← 与基线相同！
  激活值：             FP8（1.0 字节/元素）   ← 比 BF16 少 2×！
  GEMM：               混合输入 INT4→FP8 反量化 + FP8 QMMA @ 296 TF
```

| 属性 | 基线（W4A16 Marlin） | 旧路径（W8A8 FP8） | 真 W4A8 |
|---|---|---|---|
| 权重 HBM 存储 | 0.5 B/elem | 1.0 B/elem (**2×**) | **0.5 B/elem**（= 基线） |
| 激活 HBM 流量 | 2.0 B/elem（BF16） | 1.0 B/elem | **1.0 B/elem**（0.5× 基线） |
| 计算（SM120 TFLOPS） | 148（BF16） | 296 | **296**（2× 基线） |

**真 W4A8 在每一个维度上均等于或优于基线。** 不存在其可能更慢的物理机制。

### 基于物理的逐级预测

| 级别 | 主导瓶颈 | FP8 激活的效果 | FP8 QMMA 的效果 | 相对基线的净 Δ |
|---|---|---|---|---|
| **S1**（decode bs=1, M=1） | 权重 HBM 带宽 | −50% 激活流量（M=1 时影响小） | 无（计算能力未被利用） | **−5 至 −10%** |
| **S8**（混合 prefill/decode） | 混合带宽 + 计算 | −50% 激活流量 | prefill 部分 2× GEMM | **−10 至 −20%** |
| **Smax**（prefill 为主） | GEMM 计算（据 R13e profiling 占 kernel 时间 67-83%） | −50% 激活流量 | **2× GEMM 计算**施加在主导 kernel 上 | **−20 至 −30%** |

### 旧路径倒退为何不能作为预测依据

旧 S1 倒退（+118%）**完全可由权重带宽惩罚解释**：在 decode bs=1 时，每个 token 都要从 HBM 读取权重。权重大小翻倍（INT4→FP8）即 HBM 读取时间翻倍。此效应以 10-20 倍的优势压倒所有其他因素。

真 W4A8 **没有此惩罚**，因为 INT4 权重存储从始至终保留。旧倒退对真 W4A8 的性能没有任何预测力 —— 它仅确认了"INT4 存储是必需的"。

---

## 验证计划：在最新 HEAD 上复测旧 W4A8 路径

### 验证目的

1. **确认旧路径在最新 HEAD 上仍能工作**（v24 基线：Tier1 long-context、force-dense、flashinfer、torch_compile_max_bs=24）。
2. **测量当前配置下的倒退幅度**，为真 W4A8 设计提供依据。2026-04 测得的 265s S1 是在旧 Test 12 配置上（chunk=32K、prefill-max-req=1、sched-cons=1.0、torch_compile_max_bs=8）。当前配置（chunk=65K、prefill-max-req=4、sched-cons=0.8、torch_compile_max_bs=24）可能通过更好的调度部分掩盖权重带宽惩罚。
3. **健全性检查环境变量门控**（`SOAR_W4A8_FP8_GEMM=1`）在当前代码路径上仍有效。
4. **为优化目录提供免费的基线数据点**。

### 测试步骤（无源码编辑）

1. `start-instance` — 恢复暂停的 fcloud 任务
2. `sync` — 拉取最新提交 `8ce03caaf`（mcq cap 已关闭，与 v24 基线逐字节等价）
3. 覆盖环境变量：在 restart-server 中 `source prepare_env.sh` 前设置 `SOAR_W4A8_FP8_GEMM=1`
4. `restart-server` → `wait-server`（预期 ~3-5 分钟，因加载时 FP8 转换会更长）
5. 烟幕测试（单个 Paris 补全）—— 确认服务存活
6. `speed --variant s1` — 主要信号（受影响最大的级别）
7. `speed --variant s8` — 次要
8. `speed --variant smax` — 第三
9. **跳过准确率** —— iteration_002 已验证中性（79.20%，C=1.0）。FP8 路径未改变。
10. `pause-instance`

### 预计时间：约 45 分钟

### 决策矩阵

| S1 结果 | 解读 | 下一步 |
|---|---|---|
| ≥ 1.5× v24 基线（~165s+） | 旧倒退基本重现；当前配置无法改善 | 进入真 W4A8 内核开发 |
| 1.0–1.5× v24 基线（~110-165s） | 当前 Tier1 + torch_compile_max_bs=24 有助于掩盖惩罚 | 调查哪个标志缩小了差距；可能比内核开发更便宜 |
| < 1.0× v24 基线（<110s） | FP8 blockwise 在当前配置上已超越 Marlin | 重大意外 —— 先复现确认再行动 |

---

## 后续步骤（验证之后）

若旧路径倒退确认（预期结果）：
1. **构建真 W4A8 内核**，按照 `PROPOSAL_W4A8_REAL_001` Option A（Machete 风格混合输入 FP8 QMMA）。
2. **Phase 0 微基准测试**：真实隐藏状态上的 FP8 激活量化 MSE。
3. **fcloud 完整准确率运行**，以 C=1.0 为门槛。
4. **并行轨道**：准确率稳定性（mcq 修复、更大 cap 或重复检测器）。

若旧路径倒退未重现（意外结果）：
1. 重新检查 W4A8 管线 —— 可能有东西悄然损坏。
2. 在投入内核开发前，重新评估旧 FP8 blockwise 路径是否实际可用（可能性很小，但值得检查）。

---

## 参考资料

- `PROPOSAL_W4A8_REAL_001.en.md` — 真 W4A8 内核设计（三个候选路径）
- `CHANGE_W4A8_001_iteration_002.en.md` — 旧 W8A8 FP8 路径验证结果（已废弃）
- `SM120_RTX_PRO_HARDWARE.md` — GPU 规格：FP8 QMMA = 296 TF、BF16 = 148 TF
- `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` — 真 W4A8 列为最高优先级事项
- `TEST_RESULTS_TRACKING.md` — 所有历史准确率/速度数据
- Profile 数据：`R13e-prof-32k/64k/128k` — GEMM 占长上下文 prefill 的 kernel 时间 67-83%
