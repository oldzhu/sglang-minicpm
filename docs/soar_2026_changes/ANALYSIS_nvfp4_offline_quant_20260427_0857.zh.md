# 分析 — 离线 NVFP4 权重量化作为未来优化方向

**日期**: 2026-04-27 08:57
**背景**: 接 `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md`。用户提问："如果做离线 NVFP4 量化代替离线 INT4 量化，会不会得到更好的性能？是不是可以放进推荐项之后的改进选项？"
**状态**: 仅分析，无代码改动。结论: NVFP4 列为**延后的 stretch 目标 (#5)**，不作为下一轮主推方向。

---

## 1. 理论上的上限

SM120 (RTX PRO 6000 Blackwell) tensor core 峰值：

| 路径 | 峰值 TFLOPS |
|---|---|
| BF16 HMMA（当前 GPTQ + Marlin） | 148 |
| FP8 QMMA (W4A8 / mxfp8) | 296 |
| **FP4 QMMA (NVFP4 / MXFP4)** | **593** |

如果权重和激活都能走原生 FP4 QMMA，GEMM 峰值理论上 **是当前路径的 4×**。在长上下文、prefill 主导的负载（这正是官方 speed 数据集的特征）上，GEMM 是真实瓶颈，因此理论上限确实诱人。

## 2. 实测下限（这一课已经付过学费）

- **Test 21 已经在本模型上跑过离线 NVFP4 量化。**
- 结果: 精度崩盘到 ~12 %（远低于 97 % 的硬性下限，C = 0）。
- 这是**实证**：在 MiniCPM-SALA 上，朴素 PTQ NVFP4 是不可行的。

## 3. 为什么 NVFP4 崩了，而 GPTQ INT4 可以

| 因素 | GPTQ INT4（当前） | NVFP4（Test 21） |
|---|---|---|
| 编码 | 均匀整数，16 个码点 | 浮点 (1 符号 + 2 指数 + 1 尾数)，16 个码点，对数间隔 |
| 缩放粒度 | per-group BF16 scale，group_size=128 | 每 ~16 元素块共享一个 block-scale |
| 与 LLM 权重的契合度 | 均值为 0、近似 Laplace 分布 → 均匀栅格 + 小组合适 | 对数间隔在零附近精度差，而权重质量集中在零附近 |
| 校准敏感度 | 容忍度高；现成 GPTQ 即可 | 需要旋转变换 (QuaRot / SpinQuant) + outlier 处理，否则小于 7B 的模型在 MMLU 类任务上掉 >5 分 |
| 与递归状态的兼容性 | 误差不在 decode 步间累积 | 24 个 lightning 层的 state 是乘性演化的 → 每层微小的 FP4 误差会在 1000+ decode 步间复合放大 |

## 4. 其他现实障碍

1. **Lightning kernel 不原生消费 FP4**：即使权重是 FP4，SimpleGLA 的 Triton kernel 也要改造才能支持 FP4 输入。Marlin upstream 有 W4A4 原型，但是 FP4 激活 / INT4 权重，方向不对。
2. **submission size 收益微乎其微**：
   - GPTQ INT4 + group_size=128 + BF16 scale → 约 4.125 bit/元素。
   - NVFP4 + 16 元素块共享 block-scale → 约 4.5 bit/元素（FP8 block-scale 甚至更多）。
   - 即 NVFP4 **并不更小**，反而略大。2 GB 上限上没收益。
3. **校准成本**：真要把精度恢复回去，需要 QuaRot / SpinQuant 旋转预处理 + per-layer NVFP4 校准。这是多日量级的迭代，与当年搭 GPTQ + sparse_qkv 流水线相当。
4. **C 系数风险**：任何低于 97 % 的精度都会让分数清零。Test 21 已经直接证明这个风险对 NVFP4 + 本模型是真实且大的。

## 5. 更新后的优化优先级列表

| 排名 | 方向 | 预期收益 | 风险 | 工作量 |
|---|---|---|---|---|
| 1 | W4A8 (FP8 激活 + INT4 权重，走 Marlin W4A8 路径) | prefill GEMM ~1.5-2×，进入 FP8 QMMA 296 TF | 中 | 高 |
| 2 | Lightning state FP8（提案 #3） | 长上下文 decode 提升 10-20 % | 中 | 中 |
| 3 | Lightning fused kernel（提案 #4） | decode 提升 5-15 % | 低 | 中 |
| 4 | 投机解码（提案 #6） | S₁ 提升 1.3-2× | 低 | 中-高 |
| **5** | **NVFP4 权重（延后）** | **理论 GEMM 峰值 ~4×，但 Test 21 在 12 % 精度处崩盘** | **极高** | **高（需 QuaRot/SpinQuant + 多日校准）** |

## 6. 建议

把 NVFP4 当作 **W4A8 成功之后的 stretch 目标**，不要作为下一轮主推：

- **第 1 步（提案 #1）**：先把 W4A8 跑通。捕获理论上限的 ~2/3（148 → 296 TFLOPS），且精度风险显著低于 NVFP4，因为权重保留我们已验证的 INT4。
- **第 2 步（仅在 W4A8 成功且时间/预算还剩时）**：重新评估 NVFP4，但要带齐全套机械——QuaRot 旋转、细致 per-layer 校准、lightning kernel 的 FP4 改造。当作多日校准周期单独立项，全程跟踪精度回归。
- **不要**把 GPTQ INT4 → NVFP4 当作下一轮的*主*改动。Test 21 已经替我们交过这笔学费。

## 7. 决策汇总

| 问题 | 回答 |
|---|---|
| NVFP4 性能能比 INT4 好吗？ | 理论上可以（~4× GEMM 峰值）。 |
| 它能做下一轮的可行选项吗？ | 不行。Test 21 已证明朴素 PTQ NVFP4 在本模型上精度崩盘。 |
| 它在优先级里排在哪？ | 第 5 位，延后到 W4A8（第 1 位）验证之后。 |
| 怎样才能让它变得可行？ | (a) 加旋转变换 (QuaRot / SpinQuant)；(b) FP4-aware lightning kernel；(c) 多日校准 + 精度回归跟踪；(d) 接受对 C 系数的可观风险。 |

---

## 8. 交叉引用

- 上一份研究备忘: `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md`
- 量化栈澄清: `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md`
- 失败的 NVFP4 实验记录: `TEST_RESULTS_TRACKING.md` 中的 Test 21
