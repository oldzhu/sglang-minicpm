# 调研 — 第 7 周冠军（香草小张）技术复盘

**日期**：2026-05-04
**来源**：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
**冠军**：香草小张（华中科技大学本科生组队）—— 第 7 周（2026-04-29），得分 **88.35**（#1）
**单次跃迁**：此前最高 ~81-82 → 88.35（+~7 分）
**含义**：榜单 1-2 名（88.35 / 86.68）已与第 3 名（67.8）拉开决定性差距 —— 两支队伍走的应是同一条路线。

## 冠军取胜的两根支柱

### 1. NVFP4 权重量化 + **FourOverSix** 自适应 block scale

**基础路线**：GPTQ + NVFP4（FP4 E2M1）+ Marlin W4A16 decode。

**NVFP4 可表示值**：`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`，最大绝对值 = 6。

**标准 NVFP4 问题**：每个 block 用 `M=6` 将权重归一到 `[-6, 6]`。
4 与 6 之间间隔过宽，对于权重集中在 `[2/3·6, 6]` 的 block，量化误差集中在
`[4, 6]` 段（除了端点 4 和 6 之外没有其他可表示值）。

**FourOverSix 思路**（论文：Cook/Guo/Xiao/Lin/Han，MIT+NVIDIA，arXiv:2512.02010）：
对每个 block，在 **M=6** 与 **M=4** 之间选择反量化 MSE 更小者。M=4 放弃 (4, 6]
区间，换取 [2, 4] 这一权重密集段的更细分辨率（block 权重映射到 [-4, 4]，
`2, 3, 4` 三个值都可用）。

```
# M=4 对应的 block scale ≈ M=6 时的 1.5 倍
scale_m4 = fp8(scale_m6.float() * 1.5)
mse_m6   = ((W_block - dequant(W_block, scale_m6)) ** 2).mean()
mse_m4   = ((W_block - dequant(W_block, scale_m4)) ** 2).mean()
final_scale = scale_m4 if mse_m4 < mse_m6 else scale_m6
```

**输出格式不变**：仍是标准 NVFP4 = 4-bit 权重 + FP8 block scale。
推理 kernel **完全不需要改动**。吞吐与基线 NVFP4 相同。

**冠军集成方式**：把自适应 scale 选择嵌入 GPTQ：
```
估算 block scale → 比较 M=6 与 M=4 反量化误差
              → 选择误差更小的 scale
              → 进入 GPTQ 权重优化迭代
```
先定 M、再在该框架下跑 GPTQ —— 避免 scale 选择与 GPTQ 迭代相互干扰。

**实测**：约 40-43% block 选择 M=4。MLP 层 M=4 比例明显高于 attention QKV。
各类评测任务上都有稳定且可量化的精度提升。

### 2. Medusa 推测解码适配 MiniCPM-SALA 混合注意力

**基础概念**（Cai 等，ICML 2024）：不引入独立 draft 模型 —— 在主模型最后一层
hidden state 之上加多个轻量预测头，每个 head `k` 预测后续第 `k` 个 token：
```
p_t^(k) = softmax( W₂⁽ᵏ⁾ · (SiLU(W₁⁽ᵏ⁾ · h_t) + h_t) )
```
`W₁` 初始化为 **全零**，使训练初期 head 的预测与主模型完全一致（训练稳定）。

**Tree attention verify**：
1. 每个 head 产出 top-`s` 候选；
2. 通过笛卡尔积构成候选树（如 2 head × top-2 = 4 条路径）；
3. **一次** 主模型 forward，通过修改 attention mask 让每个节点只能看到自己的
   前驱路径（看不到 sibling 分支）—— 一次性 verify 整棵树；
4. 选最长可接受前缀。

**MiniCPM-SALA 难点 —— GLA 递推 state**：模型大量使用 GLA（gated linear attention）：
```
h_t = exp(-γ) · h_{t-1} + k_t · v_t^T
```
Tree verify 在此处会出错：sibling 分支必须各自从同一 parent state 出发，
线性继承前一个候选的 state 会把跨分支历史信息带入当前分支 → 答案错误。

**冠军做法**：为 verify 路径增加 **GLA state 分叉逻辑** —— 每个分支独立从
`h_{parent}` 起步，互不污染。

**训练数据加权**：按验证集分布对训练数据加权重采样 → 接受率相对均匀采样有
可测量提升。

**实测**：
- Medusa K=1 verify 开销 ≈ 0.39 ms — 远低于单次正常 decode step 的耗时；
- 各并发档位下 decode 端到端吞吐都有稳定正向收益。

## 这套组合为什么能霸榜

| 杠杆 | 准确率影响 | 速度影响 | 涉及层 |
|------|-----------|---------|---------|
| FourOverSix | +0.5-1pt（让 C 档有余量）| 直接为零；为 W4A16 提供更安全的精度底 | 全部 linear 层 |
| Medusa K=1 | 无损（verify 拒绝错误候选）| Decode tokens/step 各档约 +15-30% | 新增 head + GLA 分叉 |
| **组合** | C ≥ 0.96 稳定保持 | Smax 段 decode 主导 | — |

Decode 收益在 **三档** 速度（S1/S8/Smax 各 30%+ 权重）上都叠加。
冠军一次提交从 ~82 跳到 88.35。

## 仍不清楚的细节（需补研究）

1. **Medusa head 训练语料** —— 文中说"加权对齐评测分布"，但评测集文本私有；
   他们用的代理分布未公开。我们必须自行构造代理（公开来源中长上下文 QA/CWE 类样本）。
2. **量化感知训练** —— 不清楚 head 是基于 FP16 主模型还是 NVFP4 主模型训练的。
   顺序对接受率有影响。
3. **Tree 形状**（head 数 × top-s × depth）—— 文中未说。标准 Medusa-1 论文用
   5 个 head × top-s × depth-2。文中"K=1"可能表示"1 个 head"或"深度 1"。
   verify 0.39 ms 的开销与浅树相符。
4. **Marlin kernel** —— 他们保留标准 Marlin W4A16 路径（不改 kernel）。
   我们仓库已通过 sgl-kernel 提供 Marlin；需确认 NVFP4 W4A16 支持情况。

## 仓库交叉引用

- 既有 Marlin 工作：[docs/soar_2026_changes/CHANGE_0075_*](CHANGE_0075_marlin_kv.md)（如适用）。
- SM120 硬件规格：[docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md) —— FP4 = 593 TFLOPS（BF16 的 4 倍）。
- 优化目录：[docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)。
- 2026-05-04 排行榜：榜首 1-2 = 88.35 / 86.68；team-beta = 30.04（#22）。距 #5 差 +20.62（68%）。

## 行动：见提案

下一步分阶段实施这两项优化的提案：[PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md)。
