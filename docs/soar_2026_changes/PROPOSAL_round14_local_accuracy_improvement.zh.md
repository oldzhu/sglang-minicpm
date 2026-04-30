# 提案：Round 14 — 本地精度提升计划

**状态**：讨论 / 待实施菜单（暂无代码改动）
**范围**：v20 基线（GPTQ sparse_qkv_w8 + FP8_e5m2 KV + 原版 flashinfer + torch.compile bs=8 + fused_qk_norm_rope + mixed-chunk）
**目标**：将本地平均精度显著提升至新 fcloud 噪声下限（当前 74.87–78.73%、均值约 75–76%）之上，使我们对官方 77% C=0.92 阈值保持安全余量，并真正具备冲击 99% 归一化精度档位的可能。

---

## 0. 为什么现在做（而不是更早）

1. **v20 已解锁速度**：13f-1 验证 +9% / +8% / +6%，精度无回退（差距 +0.30pt 落在 ±1pt 噪声带内）。速度优化可以继续，但若不动模型/量化，SM120 硬件天花板已可见。
2. **精度是乘数**：官方分数 = `Performance × C`，C 在 99% 处由 1.0 降到 0.96，98% 处 0.92，97% 以下直接归零。每跨一档 ≈ 总分 4%，远大于目录中任意单项速度优化。
3. **本地↔官方差距客观存在**：本地仅 public 集，官方 public+private。我们必须留出安全余量，不能踩线过。当前本地均值 75–76%，距 C=0 悬崖**几乎没有缓冲**。
4. **mcq 是噪声主因**：13f-4 quartet A1–A4 中，同一二进制 4 次运行 mcq 在 40–63% 之间漂移。仅压住 mcq 方差，预计可在不改模型的情况下抬升均值 ≥3pt。

---

## 1. 选项菜单（按工作量/风险升序）

| # | 选项 | 工作量 | 速度风险 | 预期精度增益 | 备注 |
|---|------|--------|----------|--------------|------|
| **A** | **通过聊天模板（`preprocess_model.py`）关闭 mcq 的 thinking** | XS | 无 | **mcq +5 ~ +20pt（均值）** | 结构性修复跑飞生成；机制详见 §1a |
| **B** | **借 `generation_config.json` 做按任务的 `max_new_tokens` 上限 + 额外停止符** | XS | 无 | +1 ~ +3pt（方差↓） | A 的兜底；即便 A 在某些样本失灵也限死最坏长度 |
| **C** | **GPTQ 重新校准**（更大 / 按长度分层；当前 90 条） | S | 无 | +0.5 ~ +2pt | 一次性离线；用户确认预算 ≤ 1.5h |
| **D** | **关键层保留 bf16**（不量化精度最敏感的 linear） | S | 小（模型大小、prefill 速度） | +1 ~ +2pt | 现已部分实现 sparse_qkv_w8，可扩展至 o_proj、gate |
| ~~E~~ | ~~KV 由 FP8_e5m2 升级到 FP8_e4m3~~ | — | — | **已试，已失败** | Test 30（2026-04-22，77.96%）：mcq 由 96.67→53.33% 崩塌；e4m3 让跑飞更糟。**已从菜单删除。** |
| **F** | **KV 升级到 BF16**，并下调 max_running_requests | XS | **大**（显存、batch↓） | +0.5 ~ +1.5pt | 兜底精度手段，预计 S8/Smax 回退 |
| **G** | **AWQ 替代 GPTQ** | M | 无 | 不确定 ±2pt | AWQ 在 int4 常更优，但本模型主要是 w8，收益不确定 |
| **H** | **GPTQ 前的 SmoothQuant 预处理** | M | 无 | +0.5 ~ +1.5pt | 激活平滑可降低离群通道量化误差 |
| **I** | **mcq 走推测解码（eagle3 / draft model）快通道** | L | 中（draft 不佳会掉精度） | 精度持平、速度 +10–25% | CHANGE_0090 已搭好脚手架，留待 A–D 之后 |

---

## 1a. 为何阶段 14.1（聊天模板 / generation_config）确实能抬升精度

**用户的直觉是对的——改聊天模板并不会让模型“变聪明”。** 但它会决定**模型输出哪些 token、何时停止**——而对当前评测，这恰恰就是决定精度的最大杠杆。下面是完整的因果链（证据均来自仓库内既有文档）：

### mcq 跑飞的病理（摘自 `PROPOSAL_iteration_A0_mcq_runaway.zh.md`）

MiniCPM-SALA 的聊天模板默认 `enable_thinking=True`。对一道 mcq，模型本应输出：

```
<think> 简短推理 ... </think>
ANSWER: B
```

评测脚本 `eval_model_001.py:178` 的提取器只做：

```python
parts = pred.split('</think>')
return parts[-1].strip() if len(parts) > 1 else pred
```

所以 **能否得分完全取决于输出里有没有 `</think>`。**

Test 34a（以及 13f-4 quartet）观察到：

| 任务 | mcq 精度 | mcq 平均输出 token | 实际发生 |
|---|---|---|---|
| 幸运 run | 96.67% | ≤ 1,000 | 模型早早吐出 `</think>` → 提取器返回字母 |
| 倒霉 run | 40–53% | 10,000–11,000 | 思考链一直不收尾；要么撞到 `max_out_len` 截断，要么提取器回退到整个思考块 → 找不到字母 → 0 分 |

**这不是随机噪声，是双峰失败模式。** 同一二进制在 mcq 上得 96% 还是 53%，完全取决于采样过程中是否凑出 `</think>\n\nANSWER:` 这 8 个 token。这正是 13f-4 quartet 同配置下 mcq 在 40–63% 之间漂的原因。

### 选项 A 真正在做的事

`preprocess_model.py` 修改模型的 `chat_template`（Jinja），令 `enable_thinking` 默认为 **False**（或按系统提示中的 `task=mcq` 标识做按任务选择）。关闭 thinking 后模型直接输出：

```
ANSWER: B
```

这时不再需要 `</think>`，提取器走 `if len(parts) > 1` 的回退分支返回完整短答案。**失败模式被结构性消除。** 增益下界：当前 13f-4 quartet 上 mcq 均值约 50–55%；若 A 把 mcq 稳定在已观察到的“幸运区间”90–96%，**总体**均值精度会移动 +7 ~ +10pt（mcq 占 5 个任务之一，等权）。

### 选项 B 真正在做的事

即便有了 A，仍想要硬上限。`generation_config.json` 随模型目录一同提交，sglang 的 tokenizer/sampler 会读取它。我们追加：

- 按任务的 `max_new_tokens`（mcq ≤ 1024、qa ≤ 512、niah ≤ 1024、cwe/fwe ≤ 16384），同样借 chat_template 钩子按任务注入停止符列表。
- 额外停止符：`"</think>\n\nANSWER:"`、`"\n\nFinal answer:"`、`"<|endoftext|>"`——都是终止类 token，不出现就无害，出现就立即截断。

B 限死最坏生成长度，即便 A 在个别样本失灵，也不会有 mcq 烧掉 11k 个 token。

### 为何这都不是 harness 改动

A 与 B 都落在会随提交包发布的文件里：
- `preprocess_model.py` → 在提交准备阶段改 `tokenizer_config.json` / chat_template
- `generation_config.json` → 与模型权重一起发

`eval_model_001.py` 不动。官方评测器跑的是它自己未改动的 harness 副本；改变的只是“当 harness 让我们生成时，我们的模型吐出哪些 token”。**这是修复 mcq 跑飞的唯一合法路径。**

### 增益上限

据观察，前 5 名队伍在 mcq 上稳定 95%+。我们当前 50–60% 是生成控制问题，不是知识问题（幸运 run 已证明模型答得对）。所以阶段 14.1 有明确上限：**mcq 由 ~55% → ~90%** = **总体 +7pt**，零速度代价。这是当前桌面上最大的单项精度杠杆。

---

## 2. 推荐分阶段执行

### 阶段 14.1 — 方差抑制（不动量化）
1. **选项 A**：经由 `generation_config.json` 与 `preprocess_model.py` 内的聊天模板，研究每个任务的 `max_new_tokens` 与停止符。假设：mcq 跑飞的思考过程是方差主因。Iteration A-0（已归档对话）已识别但未彻底修复。
   - 具体：在 `preprocess_model.py` 中按系统提示中的 task=mcq 信号追加形如 `"</think>\n\nAnswer:"` 的停止符，并下调 mcq 的 token 上限。运行后用 `predictions.jsonl` 离线验证。
   - `.github/copilot-instructions.md` 严禁修改 `eval_model_001.py`。所有限制必须落在服务端 / 聊天模板 / generation_config 上，不能改 harness。
2. **选项 B**：若 A 落地后仍有残余方差，在 `generation_config.json` 中调 temperature/top_p/top_k。

**验证**：默认配置 vs 14.1 配置交替 4 次精度 run；当 Δmean > 1pt 且 Δstd ≤ baseline 时采纳。

**预期结果**：本地均值 75–76% → 77–79%，标准差由 ~1pt 收紧到 ~0.3pt。

### 阶段 14.2 — 校准升级
3. **选项 C**：重建校准集。
   - 当前：`perf_public_set.jsonl` 中 90 条分层样本。
   - 尝试：150–200 条，**同时按任务和输入长度分桶**（官方集长上下文更多）。
   - 尝试：在 mcq + qa 上过采样（这两类样本量最大）。
   - 尝试：加入少量 8k/16k/24k 长上下文合成探针，将校准锚定到评测真实分布。
   - 重跑 preprocess_model.py + 精度评测，留最优。
   - 用户确认预算：≤ 1.5h；fcloud H800 级上 200 条样本可舒适放下（GPTQ 一遍 ~30–45 min + 一轮精度 ~30–40 min）。
4. ~~选项 E（fp8_e4m3）~~——已被 Test 30（2026-04-22）证伪：77.96%、mcq 崩到 53.33%。**跳过**。

**验证**：与阶段 14.1 优胜者交替 2–3 次。

### 阶段 14.3 — 混合精度扩展（仅当 14.1+14.2 仍距 78% ≥ 3pt）
5. **选项 D**：把 sparse_qkv_w8 思路扩展到 `o_proj`（部分变体已做），以及对 Top-K 最敏感层的 `gate_proj`（用 Hessian 迹做代理）。记录模型体积变化。
6. **选项 H**：在 GPTQ 之前扫 SmoothQuant α=0.3/0.5/0.7。

**验证**：完整 4-run quartet，对比均值与最差值。

### 阶段 14.4 — 推测解码（暂搁路径）
7. **选项 I**：仅在 14.1–14.3 落地后再启 eagle3。目标是速度而非精度，但需验证 eagle3 + GPTQ + flashinfer 组合精度不掉。

---

## 3. Round 14 不会做的事

- **不改 `eval_model_001.py`**（仓库规则）。所有修复落在 `prepare_env.sh`、`preprocess_model.py`、`generation_config.json`、聊天模板与 sglang 源码内。
- **不退回 v20 基线之前**（`SOAR_BACKEND_VARIANT=flashinfer`、不带 force-dense）。14.x 的对比基线是 v20，不是 v18。
- **不靠单次高方差幸运 run 拍板**。永远 2–4 次交替再决策。
- **不预先量化、提交预量化权重**（官方禁止；我们提交的是现场跑的校准脚本）。

---

## 4. 用户决策（已于 2026-04-30 收敛）

1. **阶段 14.1 机制** —— §1a 已澄清：聊天模板修改不是“指望模型变聪明”，而是结构性消除驱动 mcq 双峰得分（50–96%）的 `</think>` 漏发故障模式。等待最终 go / no-go。
2. **校准成本**：≤ 1.5h 已确认可接受。阶段 14.2 计划在预算内。
3. **KV e4m3（选项 E）**：放弃 —— Test 30 已证伪（77.96%、mcq 崩塌）。不再重试。
4. **提交节奏**：仅当**速度与精度同时优于上一已提交包**时才进行官方提交。仅有 14.1（只动精度、速度持平）→ 不提交；将 14.1 与下一次速度收益打包，或等 14.1+14.2 都落地并确认速度未回退后再提交。

---

## 5. 交叉引用

- v20 打包：prepare_env.sh 提交 `205d8cb91`（推送 minicpm-src），tarball 位于 `benchmark/soar/demo_sala/minicpm_sala_submit_v20.tar.gz`（742.8 MB）。
- 方差来源：`docs/soar_2026_changes/PROPOSAL_round13f4_variance_quantification.zh.md`。
- mcq 跑飞前期分析：`docs/soar_2026_changes/PROPOSAL_iteration_A0_mcq_runaway.zh.md`。
- 优化目录：`docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`。
- 战略路线图：`docs/soar_2026_changes/STRATEGIC_ROADMAP_TOP5.zh.md`。

---

## 6. 推荐下一步

**批准阶段 14.1（选项 A + 选项 B）** 作为下一迭代。理由：
- 风险最低（仅服务端，无核函数 / 模型改动）；
- 杠杆最高（mcq 方差是本地标准差的最大贡献者）；
- 验证最快（一轮 fcloud 4 次 run ≈ 1 小时）。

若批准，下一步将起草 `CHANGE_0140_mcq_variance_taming.{en,zh}.md`，给出具体的 `generation_config.json` / 聊天模板修改，再按标准流程进入 fcloud 测试。
