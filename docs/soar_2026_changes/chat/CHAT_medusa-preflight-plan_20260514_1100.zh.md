# CHAT — Medusa 预飞行 Diff 计划（2026-05-14）

## 第 1 轮 — 用户请求预飞行 + 每个 diff 文档化

**用户**：
- "is the pre-flight can run successfully at local without nvidia gpu? In addition, previously u mentioned prediction verification works fine with the full attention layer but need to customize to work with lightning attention? why? is this customization already done in our code?"

**Agent 回答**：
1. 预飞行不能在 CPU 上跑 — sglang 是端到端 GPU（flashinfer / GLA / causal_conv1d / sgl-kernel verify 全部是 CUDA）。实际目标：fcloud 上的短交互脚本（~5–10 分钟），而非完整评测（~2 小时）。
2. Lightning attn 的 verify 定制在 `hybrid_linear_attn_backend.py` 中**已实现** — 显式的 `is_target_verify` 分支处理 topk=1 线性链（即 Medusa K=1 / Ngram 场景），仅用 `custom_mask`。代码注释确认 Medusa/Ngram 不使用 `retrive_*` 字段。所以 lightning attn **可能不是**灾难性失败的来源。剩余不确定性：MedusaWorker 是否触发 `SpeculativeState` mamba 池分配（很可能是 — Stage 3a 跑得没崩）。
3. 推荐路径：在 fcloud 上构造单进程预飞行脚本（~10 分钟/迭代），逐步 diff MedusaWorker 与 NgramWorker，修到干净。

**用户**：
- "let us do pre-flight in fcloud to clean the diff, pls do document each diff analysis and the change to clean the diff, including the why decide to clean or not clean the diff."

**Agent 行动**：
- 创建 `docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.{en,zh}.md`：
  - §1 背景 — 三个工作假设（位置不变量、retrive_* 方向、隐藏态捕获位置）+ lightning attn SpeculativeState 不确定性。
  - §3 预飞行脚本设计 — 单个 fcloud 脚本运行阶段 A（NgramWorker 参考）+ 阶段 B（MedusaWorker 测试）+ 阶段 C（逐字段 diff）。dump seq_lens / spec_info 字段 / out_cache_loc / next_token_logits argmax / accept_length / next_token_ids。
  - §4 **Diff 迭代协议** — 强制每 diff 表格：`[字段][A][B][不同？][严重性 (critical/cosmetic/unknown)][决定 (clean/keep/investigate)][原因]`。所有迭代追加到同一文档作为 §5.1、§5.2、...。
  - §6 风险：engine 初始化非确定性、draft 内容控制、post-verify bug、已训练头延后到后续迭代。
  - §7 两个批准门：(1) 脚本内容 + fcloud 启动，(2) 每轮迭代的代码改动 + 下一轮。
- 等待用户对以下事项的批准：
  - (a) CHANGE_0165 中的计划本身。
  - (b) 允许编写实际 `preflight_medusa_vs_ngram.py` 脚本。
  - (c) 允许为迭代 1 启动 fcloud。

## 结果 / 待解决问题

- 暂未编辑代码。暂未触发 fcloud。
- 下一步 agent 动作（等用户"go"）：编写 `benchmark/soar/demo_sala/preflight_medusa_vs_ngram.py` 并在 fcloud 前展示给审阅。

## 交叉引用

- CHANGE_0165 EN：[docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md](docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md)
- CHANGE_0165 ZH：[docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md](docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)
- 先前失败：[docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md](docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md)
- 早期（现已取代）提案：[docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.en.md](docs/soar_2026_changes/PROPOSAL_medusa_k1_positional_offbyone_fix.en.md)
