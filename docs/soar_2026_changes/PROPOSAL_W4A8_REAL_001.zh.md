# 提案：真正的 W4A8 — INT4 存储 + FP8 计算 kernel

**状态：提案（待批准）。** 未获批准前不做任何代码改动。

## 背景与动机

CHANGE_W4A8_001（commit `7ce21c3f5`）**命名错误**。该实现实际做的是 **W8A8 FP8 blockwise** —— 加载期把 GPTQ INT4 权重反量化到 BF16，再量化为 **FP8（8 位）存储**，运行时跑 cutlass FP8 × FP8 GEMM。这把权重内存占用相对 Marlin INT4 基线（W4A16）翻了一倍。

fcloud（SM120 RTX PRO 6000）结果：
- S1 +118%、S8 +56%、Smax +30% vs v18 基线（W4A16 Marlin）。

这一回退是在内存带宽受限的 decode 工作负载下放弃 INT4 2× 权重带宽优势的**必然结果**。它**并不**反驳 W4A8 假设。

**真正的 W4A8** 同时保持 INT4 权重存储 *并* 使用 FP8/INT8 张量核心做 MMA，从而拿到双重收益：
- 2× 权重带宽（INT4 vs FP8/BF16 存储）
- 2× 算力（SM120 上 FP8 QMMA / INT8 IMMA = 296 TF vs BF16 = 148 TF）

相对 Marlin W4A16 的理论 S1 改进：
- 与 Marlin 同等权重带宽（保留 INT4 存储）
- FP8/INT8 激活 = 2× 更小 → 减少激活内存流量
- 在持续大 M 上 MMA 更快（对 prefill/S8/Smax 比 decode 影响更大）
- 实际预期收益：**S1（decode）5–15%**、**S8/Smax（prefill 混合）10–25%**

## 规则合规检查（SOAR 约束）

- **现场量化、≤ 5h**：权重反量化/再量化在每次提交加载时进行，激活量化在每次前向中进行。两者都是确定性的、快速的，远在预算内。
- **2GB 提交**：kernel 编译进 sgl-kernel wheel；不增加模型工件大小。
- **Apache 2.0 / 可复现 / 可解释**：下文所有候选 kernel 都是宽松许可。
- **不依赖禁用技巧**：不滥用 prefix cache，不修改评估脚本。

## 精度/稳定性风险

| 路径 | 激活精度 | 精度风险 |
|---|---|---|
| QQQ 风格 W4-INT8 Marlin | INT8（每 token 对称） | **中等**：INT8 激活在 vllm/QQQ 中已大量验证；每 token 对称量化表现良好。风险约等于 FP8 e5m2 KV（已在基线中）。 |
| W4A8-Machete（FP8 e4m3 激活） | FP8 e4m3（每 token 缩放） | **低-中等**：FP8 e4m3 仅 3 位尾数，但每 token 缩放使其对 transformer 激活竞争力接近 BF16。 |
| 自定义 CUTLASS 混合输入 | 可配置 | 视选择而定 |

合格判据：本地公共集**归一化精度 ≥ 99%（C=1.0）**，与 v18 基线一致。若候选 kernel 把 C 降到 1.0 以下，则放弃该具体 kernel，试下一个。

## 三个候选路径

### 选项 A：QQQ 风格 W4-INT8 Marlin（推荐先试）

- **来源**：https://github.com/IST-DASLab/marlin（W4A8 fork）和 https://github.com/HandH1998/QQQ
- **做什么**：Marlin 风格 INT4 权重 + INT8 激活 kernel。每 token 对称 INT8 激活量化。使用 INT8 IMMA 张量核心（SM120 上 296 TF，与 FP8 QMMA 峰值相同）。
- **为什么先试**：最贴近现有 Marlin 代码路径（INT4 已经走 Marlin）；kernel 已经成熟，有 ampere/ada/hopper 实例化。
- **SM120 工作量**：CHANGE_0125 已经加了 Marlin SM120 tile 表。再加 W4-INT8 实例化只需在同一 dispatch 表里加几条新 tile 元组。预计**中等工作量**。

### 选项 B：W4A8-Machete（vllm/compressed-tensors）

- **来源**：https://github.com/vllm-project/vllm `vllm/model_executor/layers/quantization/utils/machete_utils.py` 和基于 CUTLASS 的 kernel。
- **做什么**：CUTLASS 混合输入 GEMM，INT4 权重（packed）+ FP8 e4m3 激活。Hopper 优化。
- **SM120 工作量**：Hopper/SM90 一等公民；SM120（Blackwell）回移状态未知。可能开箱可用，也可能不可用。**工作量风险高**。
- **为什么考虑**：原生 FP8 激活对某些 transformer 架构比 INT8 更友好。

### 选项 C：自定义 CUTLASS 混合输入 GEMM

- **做什么**：构建一个 CUTLASS 3.x 混合输入 GEMM（INT4 权重、FP8 激活、BF16 累加器+输出），在 warp 序言中按 K-block 反量化。
- **工作量**：最高。需要 CUTLASS 专业知识与 SM120 调优。
- **何时启用**：仅当 A 和 B 都失败时。

## 推荐计划

1. **第 1 轮迭代**（本提案，若获批）：先试 **选项 A — QQQ 风格 W4-INT8 Marlin**。最贴近现有基础设施。
2. 若选项 A 因 SM120 实例化工作量过大或精度回退被卡住，回退到 **选项 B — Machete**。
3. **选项 C** 留作最后保留。

## 详细实现计划（选项 A — 获批后下一轮迭代填充）

将要变动的文件（提案阶段尚未写代码）：

- `sgl-kernel/csrc/`：新增 W4-INT8 Marlin kernel 文件（或在现有 `gptq_marlin.cu` 中以 INT8 激活的新模板实例化方式合并）。
- `sgl-kernel/cmake/`：新增 SM120 W4A8 tile 表。
- `python/sglang/srt/layers/quantization/gptq.py`：新增 `_soar_maybe_setup_w4a8_int8` 辅助函数：
  - 验证层是 INT4 GPTQ（通过现有位宽守卫跳过 INT8 sparse_qkv）。
  - **保留**现有 Marlin INT4 权重格式不变（不再量化！）。
  - 增加每 token INT8 激活量化器（在 `apply()` 中在线进行）。
  - 从 sgl-kernel 调用新的 `marlin_gemm_w4a8_int8_kernel(...)`。
- `python/sglang/srt/models/minicpm.py`：复用 CHANGE_W4A8_001 已有的 `_soar_w4a8_eligible` 标记（无需改 MiniCPM 模型代码）。
- `benchmark/soar/demo_sala/prepare_env.sh`：新增 env flag `SOAR_W4A8_INT8_GEMM`（独立于已弃用的 `SOAR_W4A8_FP8_GEMM`）。

## 验证命令（计划）

与第 001 轮工作流一致：

```bash
# 激活量化器与反量化工具的 CPU 单元测试
python3 test/srt/quantization/test_w4a8_int8_quantizer.py

# fcloud
python3 scripts/fcloud/fcloud_workflow.py setup
sed -i 's/SOAR_W4A8_INT8_GEMM:-0/SOAR_W4A8_INT8_GEMM:-1/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 成功 / 失败判据

| 指标 | 通过 | 失败 |
|---|---|---|
| 精度（归一化） | ≥ 99%（C=1.0） | < 99%（任何 C 下降） |
| S1 | ≤ 121.71s（Marlin 基线） | > 121.71s |
| S8 | ≤ 44.09s | > 44.09s |
| Smax | ≤ 35.86s | > 35.86s |

若精度过、速度不过，先**调 INT8 激活块大小**再放弃该 kernel。任何 kernel 精度不过则放弃该 kernel，尝试下一个选项。

## 回滚说明

W4A8-INT8 路径将以 `SOAR_W4A8_INT8_GEMM=0` 默认 env 门控关闭。关闭：
```bash
sed -i 's/SOAR_W4A8_INT8_GEMM:-1/SOAR_W4A8_INT8_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```
代码回滚：`git revert <第 002 轮迭代 commit sha>`。

## 等待用户决策的开放问题

1. **批准选项 A**（QQQ 风格 W4-INT8 Marlin）作为首个真正 W4A8 尝试？**是 / 否**
2. 若是，对 **kernel 来源** 是否有约束（必须从 QQQ repo 移植？必须从零写 CUTLASS？是否可以以 Apache-2.0 vendoring 方式引入 kernel？）
3. 在回退到选项 B 之前，kernel bring-up 的时间/工作量预算？

本提案在你批准之前**不做任何代码改动**。

---

**英文配套文档**：`PROPOSAL_W4A8_REAL_001.en.md`。
