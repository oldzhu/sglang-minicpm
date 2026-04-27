# 提案：真正的 W4A8 — INT4 存储 + FP8 计算 kernel

**状态：提案（待批准）。** 未获批准前不做任何代码改动。

## 背景与动机

CHANGE_W4A8_001（commit `7ce21c3f5`）**命名错误**。该实现实际做的是 **W8A8 FP8 blockwise** —— 加载期把 GPTQ INT4 权重反量化到 BF16，再量化为 **FP8（8 位）存储**，运行时跑 cutlass FP8 × FP8 GEMM。这把权重内存占用相对 Marlin INT4 基线（W4A16）翻了一倍。

fcloud（SM120 RTX PRO 6000）结果：
- S1 +118%、S8 +56%、Smax +30% vs v18 基线（W4A16 Marlin）。

这一回退是在内存带宽受限的 decode 工作负载下放弃 INT4 2× 权重带宽优势的**必然结果**。它**并不**反驳 W4A8 假设。

**真正的 W4A8** 同时保持 INT4 权重存储 *并* 使用 8 位张量核心做 MMA，从而拿到双重收益：
- 2× 权重带宽（INT4 vs FP8/BF16 存储）
- 2× 算力 vs BF16，**但仅通过 FP8 QMMA = SM120 上 296 TF 才能保证**

### SM120 数据类型可用性（关键 — 决定 kernel 选型）

根据 `docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md`（本次比赛权威硬件参考）：

| 张量核心数据类型 | SM120 上 TFLOPS | 状态 |
|---|---|---|
| BF16/FP16 | 148 | 已列出 |
| **FP8 (e4m3 / e5m2)** | **296** | **已列出，有保证** |
| FP4 | 593 | 已列出 |
| **INT8** | **未列出** | 未验证 — 在 Blackwell 消费级上相对 Ada/Hopper 可能被砍 |

这一点是决定性的：**SM120 的 INT8 IMMA 吞吐没有在官方规格中列出。** NVIDIA Blackwell 消费级（GB202）数据手册确实包含 INT8 张量核心，但相对 FP8 QMMA 的吞吐可能下降。在该硬件上选 INT8 激活并不能保证算力收益。**FP8 激活才是安全目标。**

相对 Marlin W4A16 的理论 S1 改进（FP8 激活路径）：
- 与 Marlin 同等权重带宽（保留 INT4 存储）
- FP8 激活 = 比 BF16 小 2× → 减少激活内存流量
- FP8 QMMA 296 TF = BF16 的 2×（对 prefill/S8/Smax 比 decode 影响更大）
- 实际预期收益：**S1（decode）5–15%**、**S8/Smax（prefill 混合）10–25%**

## 规则合规检查（SOAR 约束）

- **现场量化、≤ 5h**：权重反量化/再量化在每次提交加载时进行，激活量化在每次前向中进行。两者都是确定性的、快速的，远在预算内。
- **2GB 提交**：kernel 编译进 sgl-kernel wheel；不增加模型工件大小。
- **Apache 2.0 / 可复现 / 可解释**：下文所有候选 kernel 都是宽松许可。
- **不依赖禁用技巧**：不滥用 prefix cache，不修改评估脚本。

## 精度/稳定性风险

| 路径 | 激活精度 | 精度风险 |
|---|---|---|
| W4-FP8（Machete 风格或自定义 CUTLASS） | FP8 e4m3（每 token 缩放） | **低-中等**：FP8 e4m3 仅 3 位尾数，但每 token 缩放使其对 transformer 激活竞争力接近 BF16；在生产栈中已广泛部署。 |
| W4-INT8 Marlin（QQQ 风格） | INT8（每 token 对称） | **中等**：INT8 激活在 vllm/QQQ 中已大量验证。风险约等于 FP8 e5m2 KV（已在基线中）。**但受 SM120 上 INT8 IMMA 吞吐问题制约。** |
| 自定义 CUTLASS 混合输入 | 可配置 | 视选择而定 |

合格判据：本地公共集**归一化精度 ≥ 99%（C=1.0）**，与 v18 基线一致。若候选 kernel 把 C 降到 1.0 以下，则放弃该具体 kernel，试下一个。

## 三个候选路径（已重排序 — FP8 优先以保证 SM120 算力收益）

### 选项 A（推荐）：W4 + FP8 激活 — Machete 风格或自定义 CUTLASS 混合输入

- **来源**：vllm Machete kernel（https://github.com/vllm-project/vllm `csrc/quantization/machete/`）；上游 CUTLASS 3.x 混合输入示例。
- **做什么**：混合输入 GEMM，**INT4 packed 权重** + **FP8 e4m3 激活**，反量化进 FP8 寄存器对喂给 SM120 上 **296 TFLOPS 的 FP8 QMMA**。INT4 存储端到端保留（HBM → L2 → SMEM → 寄存器）。
- **为什么先试**：SM120 FP8 QMMA 吞吐**在官方硬件参考中明确列出 296 TF**。这是该硬件上唯一有算力收益保证的 8 位 MMA 路径。
- **SM120 工作量**：Machete 是 Hopper/SM90 一等公民；SM120（Blackwell）回移是主要风险。CUTLASS 3.x 已有 SM120 FP8 collective builder（sgl-kernel `cutlass_w8a8_fp8` 在用），所以混合输入变体可行但需要 bring-up。预计**中-高工作量**，主要在 SM120 tile/instruction 选型。

### 选项 B（备选）：W4 + INT8 激活 — QQQ 风格 Marlin

- **来源**：https://github.com/IST-DASLab/marlin（W4A8 fork）和 https://github.com/HandH1998/QQQ
- **做什么**：Marlin 风格 INT4 权重 + INT8 激活 kernel，使用 INT8 IMMA 张量核心。
- **注意点**：SM120 INT8 IMMA 吞吐**未在官方硬件参考中列出**。选这条路径前必须先在 fcloud GPU 上跑微基准（`cutlass_profiler` 大 M INT8 GEMM）确认 INT8 IMMA ≥ FP8 QMMA。若它落到 BF16 速率（148 TF），则该路径只拿到带宽收益 — 与 Marlin W4A16 相同 — 不值得做。
- **为什么备选**：最贴近现有 Marlin 代码路径；kernel 已经成熟，有 ampere/ada/hopper 实例化；CHANGE_0125 已加 Marlin SM120 tile 表。**一旦 INT8 IMMA 吞吐验证通过，工作量最低。**

### 选项 C（最后保留）：自定义 CUTLASS 混合输入 GEMM

- **做什么**：从零构建 CUTLASS 3.x 混合输入 GEMM（INT4 权重，FP8 激活，BF16 累加器+输出），在 warp 序言中按 K-block 反量化，针对 SM120 调优。
- **工作量**：最高。需要 CUTLASS 专业知识与完整 SM120 调优扫描。
- **何时启用**：仅当 A（Machete bring-up）和 B（INT8 IMMA 微基准 + QQQ 移植）都失败时。

## 推荐计划

1. **第 0 阶段（任何 kernel 工作前的廉价微基准）**：在 fcloud 上跑 `cutlass_profiler`，测量 SM120 大 M 下 INT8 IMMA 吞吐。结果决定选项 B 是否还可行。
2. **第 1 轮迭代**（若获批）：先试 **选项 A — W4-FP8 Machete 风格**。FP8 QMMA 是 SM120 上唯一保证 296 TF 的路径。
3. 若选项 A bring-up 成本过高 *且* 第 0 阶段确认 INT8 IMMA ≈ FP8 QMMA，回退到 **选项 B — QQQ W4-INT8 Marlin**。
4. **选项 C** 留作最后保留。

## 详细实现计划（选项 A — 获批后下一轮迭代填充）

将要变动的文件（提案阶段尚未写代码）：

- `sgl-kernel/csrc/`：新增 W4-FP8 混合输入 kernel（移植 Machete 或为 SM120 写自定义 CUTLASS 3.x 混合输入 collective）。
- `sgl-kernel/cmake/`：新增 SM120 W4A8-FP8 tile 表；复用现有 CUTLASS SM120 FP8 编译选项。
- `python/sglang/srt/layers/quantization/gptq.py`：新增 `_soar_maybe_setup_w4a8_fp8_real` 辅助函数：
  - 验证层是 INT4 GPTQ（通过现有位宽守卫跳过 INT8 sparse_qkv）。
  - **保留 INT4 权重存储**（如果 kernel 需要可重打包到 Machete 格式；不上转到 FP8 存储 — 那正是 iteration_001 的 bug）。
  - 增加每 token FP8 e4m3 激活量化器（在 `apply()` 中在线进行）。
  - 从 sgl-kernel 调用新的 `machete_gemm_w4a8_fp8(...)`（或等价物）。
- `python/sglang/srt/models/minicpm.py`：复用 CHANGE_W4A8_001 已有的 `_soar_w4a8_eligible` 标记（无需改 MiniCPM 模型代码）。
- `benchmark/soar/demo_sala/prepare_env.sh`：新增 env flag `SOAR_W4A8_FP8_REAL_GEMM`（与已弃用、用于 W8A8 错标路径的 `SOAR_W4A8_FP8_GEMM` 区分）。

## 验证命令（计划）

与第 001 轮工作流一致：

```bash
# FP8 激活量化器与反量化工具的 CPU 单元测试
python3 test/srt/quantization/test_w4a8_fp8_real_quantizer.py

# fcloud
python3 scripts/fcloud/fcloud_workflow.py setup
sed -i 's/SOAR_W4A8_FP8_REAL_GEMM:-0/SOAR_W4A8_FP8_REAL_GEMM:-1/' /root/submission_sim/prepare_env.sh
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

W4A8-FP8-real 路径将以 `SOAR_W4A8_FP8_REAL_GEMM=0` 默认 env 门控关闭。关闭：
```bash
sed -i 's/SOAR_W4A8_FP8_REAL_GEMM:-1/SOAR_W4A8_FP8_REAL_GEMM:-0/' /root/submission_sim/prepare_env.sh
python3 scripts/fcloud/fcloud_workflow.py restart-server
```
代码回滚：`git revert <第 002 轮迭代 commit sha>`。

## 等待用户决策的开放问题

1. **批准选项 A**（W4-FP8 混合输入 GEMM，SM120 上 FP8 QMMA 保证 296 TF）作为首个真正 W4A8 尝试？**是 / 否**
2. 若是，对 **kernel 来源** 是否有约束：(a) 把 vllm Machete 移植到 SM120，(b) 从零写一个自定义 CUTLASS 3.x 混合输入 collective，还是 (c) 可以以 Apache-2.0 vendoring 方式引入第三方 repo 的 kernel？
3. 是否先跑 **第 0 阶段 INT8 IMMA 微基准**（廉价，fcloud 时间 < 1h）以保留选项 B（W4-INT8 Marlin）作为真正的备选？还是跳过第 0 阶段全力押注 FP8？
4. 在升级到选项 C（完整自定义 CUTLASS）之前，选项 A bring-up 的时间/工作量预算？

本提案在你批准之前**不做任何代码改动**。

---

**英文配套文档**：`PROPOSAL_W4A8_REAL_001.en.md`。
