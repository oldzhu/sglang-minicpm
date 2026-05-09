# 调研 — MiniCPM-SALA 上的推测解码方案对比（iter 001）

日期：2026-05-09
分支：`mixed_minicpm_cudagraph`
配套：[PROPOSAL_medusa_minicpm_sala_001.zh.md](PROPOSAL_medusa_minicpm_sala_001.zh.md)

目标：在 SOAR 2026 当前基线（GPTQ + FP8_e5m2 KV + dense + Tier1 + flashinfer，commit `ac91b1afe`）下，对推测解码家族做一次横向对比，作为推荐 Medusa 为下一步主攻方向的依据。

## 1. 所有候选都受 3 条共同约束

MiniCPM-SALA 架构对**任何**推测解码方案都强加 3 条限制，与 draft token 由谁产生无关：

1. **24 个 GLA（Lightning Attention）递推层**，状态 `h_t = exp(−γ)·h_{t−1} + k_t·v_tᵀ`。tree verify 的 K 条候选必须每条 sibling 从 parent state 分叉；线性历史的实现会静默污染 sibling 分支。
2. **8 个 dense self-attention 层**（标准 KV cache pages）。tree-mask 在 sglang EAGLE worker 已成熟，可复用。
3. **2 GB 提交 tarball 上限**。draft 端如果带权重，必须很小（~100 MB）。复用主模型 `lm_head` 能省下几百 MB。

凡是忽视 #1 的方案都会像我们 **Test 22 EAGLE3** 一样败掉（随机 draft，accept_rate=0.26，S1 +65%，C=0）。结论不是"EAGLE 不行"，而是"任何 learned-draft 方案在 SALA 上都必须先在框架层解决 GLA-fork"。

## 2. 候选方法对比表

| 方法 | Draft 生成器 | 训练成本 | 提交大小 | 需要 GLA-fork? | sglang 现支持 | 最佳并发档 |
|---|---|---|---|---|---|---|
| **Medusa** | 主模型最后 hidden state 上的 K 个 MLP head，复用 `lm_head` | 单卡 GPU 数小时 | ~50–100 MB | **是** | 无 | S1 ★★★ |
| EAGLE / EAGLE3 | 小型自回归 draft 模型（1 transformer 层 + 分类器） | 多卡 GPU 数天 | ~200–400 MB | **是** | 完整支持（`speculative-algorithm EAGLE3`） | S1 ★★★ |
| MTP（DeepSeek-V3 / Meta 风格） | 主模型末尾追加额外 transformer block，与 main 一起 pretrain | 在 main pretrain 阶段联合训练 | 直接增主权重 | **是** | 部分（`NEXTN`） | S1 ★★ |
| Lookahead / SpS / n-gram | 当前请求历史 token 的 n-gram 查表 | 零 | 零 | 否（无递推 draft state） | 完整（`NGRAM`） | S1 ★★ |
| Self-speculative（跳层） | 用主模型本身、跳掉若干层当 draft | 零 | 零 | **是（更糟）**——跳 lightning 层会破坏递推 | 无 | S1 ★★ |
| Standalone draft model | 单独一个小 LM | 数天 | 500 MB+ | **是** | 完整（`STANDALONE`） | S1 ★★ |

（★ = 在本基线上预期加速幅度；依据：冠军报告 + Test 22 实证 + sglang 文档。）

## 3. 逐项分析

### 3.1 Medusa（推荐——见 PROPOSAL_001）

**机制。** 主模型最后 hidden state 上的 K 个轻量 MLP head 分别预测 t+1, t+2, …, t+K。每个 head 取 top-s 候选后笛卡尔积成树 → 主模型一次 verify forward → 接受最长前缀。

**优点**
- **提交成本最小。** 复用 `lm_head` 后 K=2 ≈ 64 MB；远小于 2 GB。
- **正确实现下无损。** 与 no-spec baseline 同精度。
- **K=1 verify 单步开销 ≈0.39 ms**（冠军实测）。低于一次正常 decode。
- **与权重量化方式解耦。** GPTQ、FP8、NVFP4、BF16 都能用。先在 GPTQ 上落地，将来 NVFP4 修好可直接复用。
- **不需要联调主模型。** Heads 单独训，主权重 byte-identical。提交精度不漂移。

**缺点**
- 需要做 GLA 状态分叉（真正的工程量）。
- sglang 完全没 Medusa 代码，得从零搭。
- Heads 是 SALA 专属，开源也不能直接复用（冠军是 NVFP4 build，我们是 GPTQ build）。

**风险画像。** 几乎全部集中在 GLA-fork 正确性（提案 R2）。一旦 `accept_threshold=1.0` byte-identity 闸通过，速度收益基本是免费的。

### 3.2 EAGLE / EAGLE3

**机制。** 小型自回归 draft 模型（1 transformer block + 分类器）基于主模型 hidden state 训练。EAGLE3 加了多 token 训练 + 更丰富的验证树。

**优点**
- **sglang 已集成**（`--speculative-algorithm EAGLE3`）。
- 公开论文中在 Llama 类模型上 accept rate 高于 Medusa（论文报 0.6+）。
- Tree-verify 基础设施可复用。

**缺点**
- **训练成本是 Medusa 的 5–10×。** EAGLE3 需要数 GPU-day 在校准数据上训练；我们的离线算力预算有限。
- **提交大小 200–400 MB**（draft 层 + 分类器，`lm_head` 不复用）。仍 < 2 GB 但占预算。
- **同一个 GLA-fork 阻塞**。我们已经撞过一次（Test 22，commit 548c8c153，mem-frac 0.72）——随机 draft 下 accept_rate=0.26，S1 +65%，C=0。训练好的 draft 能拉高 accept_rate 但不能解决底层的 state-fork bug；一样会静默污染 sibling 分支。
- **依赖 tokenizer / vocab。** SALA vocab 非标准，draft 得从零搭。

**结论。** EAGLE3 严格优于 Medusa 的前提是：(a) GLA-fork 已经解决，(b) 我们能花 5+ GPU-day 训练。两个条件目前都不满足。**排在 Medusa 之后。** 如果 Medusa K=2 上限到 accept_rate ≈ 0.5 就停了，再回头考虑 EAGLE3 作为升级路径。

### 3.3 MTP（DeepSeek-V3 / Meta 风格的 Multi-Token Prediction）

**机制。** 主模型末尾追加额外 transformer block，在主模型 pretrain 阶段加 "next-k" 辅助 loss 联合训练。推理时这些 block 产 draft，主模型 verify。

**优点**
- 文献报告中 accept rate 最高（DeepSeek-V3 在长上下文 0.85+）。
- 集成最紧 → verify mismatch 最小。

**致命缺点**
- **必须联调主模型** —— 我们不能重训 MiniCPM-SALA 主权重（基线精度是参赛指标，主权重一改就破）。
- **规则冲突**：规则禁止提交预量化权重。重 pretrain MTP 然后在 5 小时内现场量化不可行。
- 同样受 GLA-fork 制约。

**结论。** **硬性排除。** MTP 需要的主模型训练在本赛事不可能做。

### 3.4 N-gram / Lookahead / SpS（sglang `NGRAM`）

**机制。** 维护当前请求历史 token 的 n-gram 表。当前缀命中时，把历史延续作为 draft。

**优点**
- **零训练，零提交大小。**
- **不涉及 GLA-fork**（draft 没有递推状态——每个 token 都从头由主模型 verify）。
- sglang 已支持。

**缺点**
- Accept rate 极依赖分布。重复代码/文本上很强；SOAR 的 5 类任务（qa/mcq/cwe/fwe/niah）中只有 `niah`、`cwe` 重复较多，`qa`、`mcq`、`fwe` 主要是新 token 流，n-gram 命中率低。
- 冠军明确选 Medusa 而非 n-gram，暗示 n-gram 对本工作负载不够。
- 加速上限远低于 learned draft（典型 S1 +10–15%）。

**结论。** **便宜的保险，不是主力。** Medusa 落地后可作为零成本叠加。不值得作为单独主攻。

### 3.5 Self-speculative decoding（跳层）

**机制。** 用主模型本身、把最后 N 个 transformer block 跳掉当 draft。

**优点**
- 零训练，零提交大小。

**致命缺点**
- **跳掉 24 个 lightning 层中任何一个都会破坏递推态演化** —— "draft" 不再是主模型的精度低版，而是输出分布完全不同的另一个模型。Accept rate 崩塌。
- 只跳 8 个 dense self-attention 层 → 没什么"短路径"可言（lightning 层占大部分 FLOPs）。

**结论。** **架构上对 SALA 不利。** 跳过。

### 3.6 Standalone draft model（sglang `STANDALONE`）

**机制。** 训练一个独立架构的小 LM 作 draft。

**优点**
- sglang 已支持。

**缺点**
- 提交大小：训练好的小 LM ≥ 200 MB。
- 训练成本与 EAGLE3 同量级。
- **每参数 accept rate 最差**（draft 与主模型不共享 hidden state）。
- 同样的 GLA-fork。

**结论。** **被 Medusa 与 EAGLE3 严格 dominate。** 无理由考虑。

## 4. 决策矩阵

每方法在 6 个轴上 1–5 打分（高=好）：

| 方法 | S1 加速 | 精度安全 | 训练成本（低=好） | 提交大小（小=好） | sglang 支持 | GLA-fork 痛感（低=好） | **总分** |
|---|---|---|---|---|---|---|---|
| Medusa | 5 | 5 | 4 | 5 | 1 | 3 | **23** |
| EAGLE3 | 5 | 4 | 2 | 3 | 5 | 3 | 22 |
| n-gram | 2 | 5 | 5 | 5 | 5 | 5 | 27¹ |
| MTP | 5 | 5 | 0 | 2 | 2 | 3 | 排除 |
| Self-spec | 2 | 1 | 5 | 5 | 1 | 0 | 14 |
| Standalone | 3 | 4 | 2 | 2 | 5 | 3 | 19 |

¹ N-gram 高分反映"免费且安全"，但 S1 上限低，**互补于** learned draft 而不是替代。建议在 Medusa 落地后叠加。

## 5. 推荐

1. **主攻：Medusa**（见 PROPOSAL_medusa_minicpm_sala_001）。S1 收益最大（占 40% 权重），提交成本最小，不重训主模型。GLA-fork 是任何 learned-draft 方法都绕不开的工程量，先做了将来通用。
2. **保险：n-gram（`--speculative-algorithm NGRAM`）。** Medusa 正确性锁定后顺带打开当安全网。成本几乎为零；在 cwe/niah 上可能再多收几个百分点。
3. **备选：EAGLE3。** 仅当 Medusa K=2 上限不到 0.5 accept_rate 时启动。前提是 GLA-fork 已建好（工作可迁移）。
4. **硬性排除：MTP、Self-spec、Standalone。** 要么规则不容、要么严格 dominated。

## 6. 下一步具体动作（不动代码）

- 批 PROPOSAL_medusa_minicpm_sala_001 的 R1（plumbing spike）。成本：2–3 天编码 + 1 次 fcloud。
- 可选：并行做一次免费的 `NGRAM` 烟囱测试（纯 server-arg，零代码，零风险）。可能小赢，且与 Medusa 工作不冲突。

## 7. 待解问题

- 冠军是否开源了 Medusa head 训练代码？若开源，我们也许能复用其数据治理脚本（征得作者同意）。若未开源，按文章描述复现。
- sglang `eagle_utils.py` 的 tree-mask 工具能否干净抽出供 Medusa 的稠密 head topology 使用？（R1 spike 中验证。）
- torch.compile 是否能与 verify forward graph 共存？（冠军未提；R1 假设关闭，R3 再视情况。）

## 参考

- Medusa 论文：Cai et al., ICML 2024 —— https://arxiv.org/abs/2401.10774
- EAGLE / EAGLE3：Li et al. —— https://arxiv.org/abs/2503.01840
- DeepSeek-V3 MTP：DeepSeek 技术报告
- N-gram speculative decoding：sglang `advanced_features/speculative_decoding.ipynb`
- 冠军原文：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- 我们之前 EAGLE3 尝试：TEST_RESULTS_TRACKING.md Test 22-acc；CHANGE_0072_three_path_optimization_research.zh.md
