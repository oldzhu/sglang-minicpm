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
| **A** | **mcq 停止符 / 思考 token 上限** | XS | 无 | 均值 +1 ~ +3pt（方差↓） | 服务端 / 聊天模板修改，最安全的第一步 |
| **B** | **优化 generation_config 默认值**（非 mcq 任务的 temp/top_p/重复惩罚） | XS | 无 | +0.5 ~ +1.5pt | 通过停止符做任务级覆盖，无需改 harness |
| **C** | **GPTQ 重新校准**（当前 90 条分层样本） | S | 无 | +0.5 ~ +2pt | 一次性离线工作；提交端量化时间预算 ≤5h |
| **D** | **关键层保留 bf16**（不量化精度最敏感的 linear） | S | 小（模型大小、prefill 速度） | +1 ~ +2pt | 现已部分实现 sparse_qkv_w8，可扩展至 o_proj、gate |
| **E** | **KV 由 FP8_e5m2 升级到 FP8_e4m3** | XS | 小（依赖核函数支持） | +0 ~ +1pt | e4m3 尾数更长更适合 KV，需检查 flashinfer 支持 |
| **F** | **KV 升级到 BF16**，并下调 max_running_requests | XS | **大**（显存、batch↓） | +0.5 ~ +1.5pt | 兜底精度手段，预计 S8/Smax 回退 |
| **G** | **AWQ 替代 GPTQ** | M | 无 | 不确定 ±2pt | AWQ 在 int4 常更优，但本模型主要是 w8，收益不确定 |
| **H** | **GPTQ 前的 SmoothQuant 预处理** | M | 无 | +0.5 ~ +1.5pt | 激活平滑可降低离群通道量化误差 |
| **I** | **mcq 走推测解码（eagle3 / draft model）快通道** | L | 中（draft 不佳会掉精度） | 精度持平、速度 +10–25% | CHANGE_0090 已搭好脚手架，留待 A–E 之后 |

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
4. **选项 E**：若 sgl-kernel + flashinfer 在 SM120 上已支持，直接试 `--kv-cache-dtype fp8_e4m3`。纯运行时改动。

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

## 4. 待用户决策的开放问题

1. **是否先批准阶段 14.1？** 工作量 XS，速度零风险，正中方差主因。一次 fcloud 轮次内可出 v20.1 候选包。
2. **校准成本预算**：官方要求量化 + 评测 ≤ 5 小时。当前 90 条样本约 20–30 min（fcloud H800 级），200 条仍宽裕。请确认 OK。
3. **KV dtype 试探**：是否允许在阶段 14.1 期间作为独立变量试 `fp8_e4m3`（选项 E）？
4. **提交节奏**：每个阶段（14.1、14.2…）一次官方提交，还是待 14.1+14.2 都落地再交？每次提交消耗 `team-beta` 一个名额；当前排名 #19（56.63 分），距第 5 名 ≥79.55 分还差 ~40%。

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
