# CHANGE_0165 — Medusa 预飞行 Diff（MedusaWorker 与 NgramWorker 逐步对比）

状态：**提案** — 在任何 fcloud 操作前需用户明确批准。
负责人：agent
关联：CHANGE_0163（Stage 3b 已训练头）、CHANGE_0164（ndt=2 灾难性尝试）、PROPOSAL_medusa_k1_positional_offbyone_fix。

## 1. 背景与动机

Stage 3a（当前已交付状态，commit `a489d78d4`）保持 `MedusaWorker.draft_token_num = num_heads = 1`。verify kernel 只遍历单节点树（bonus 根），提交其 argmax 后结束；本质上是带额外开销的 dense decode。准确率比 dense 基线低约 0.6pt；加速比结构性 ~0（除 bonus 外没有可接受的 draft 位置）。

Stage 3b ndt=2（CHANGE_0164）尝试通过设置 `draft_token_num = num_heads + 1 = 2` 加上手搓的 positions / retrive_* 元数据来修复。结果：**15.13% 准确率，C=0，mcq 失控（avg_out=64460），评测 7911s**。

三个工作假设仍然成立（仅靠代码层审阅都未验证）：

1. **位置不变量违反** — 在槽位 `seq_lens` 处喂入 `output_ids[-1]`，使 KV 与模型对位置 `seq_lens+1` 的预期自回归状态错位。
2. **retrive_* 元数据方向** — `verify_tree_greedy` 可能期望 `reconstruct_indices_from_tree_mask` kernel 产生的方向；手搓的等价物在 ndt=2 时可能在细节处不同。
3. **隐藏态捕获位置** — `CaptureHiddenMode.LAST` 返回最后一个 verify 位置的隐藏态；对 ndt=2，已训练头可能需要 bonus 位置的隐藏态。

第四个（可能性较低）顾虑在 [hybrid_linear_attn_backend.py](python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py)：

4. **Lightning attention 的 `SpeculativeState` mamba 池** — verify 路径断言 mamba 池为 `SpeculativeState`；需确认 MedusaWorker 触发了该分配。（Stage 3a 没崩，所以应该没问题，但值得验证一次。）

没有运行时证据时无法对这些排序。每个假设跑一次 2 小时准确率评测代价过大（且脆弱 — 参考 CHANGE_0164 那种"有意思但没产出的"失败）。预飞行 diff 是降风险工具。

## 2. 规则合规声明

- 预飞行是 `benchmark/soar/demo_sala/` 下的**诊断脚本** — 通过现有 sglang `Engine` API 以单进程范围运行。
- 不修改 `eval_model_001.py`（遵守评测脚本完整性规则）。
- 在发现 diff 并完成修复文档化 + 批准前，不修改提交侧文件。
- 所有 fcloud 操作遵守**省钱规则**（迭代间 pause），每轮都需用户批准。

## 3. 预飞行脚本：设计（提案）

**文件名**：`benchmark/soar/demo_sala/preflight_medusa_vs_ngram.py`

**做什么**（每次 fcloud 调用执行一轮迭代）：

1. 构造确定性的 1-batch 输入：单个固定 prompt（约 256 token），用 `perf_public_set.jsonl` 第一个 mcq 样本。
2. **阶段 A — NgramWorker 参考运行**：
   - 用 `--speculative-algorithm NGRAM --speculative-num-draft-tokens 2` 启动 sglang Engine。
   - 运行 prefill + 1 步 verify。
   - 以 JSON/pickle 捕获并 dump：
     - `batch.seq_lens`（`prepare_for_verify` 前后）
     - `spec_info.draft_token`、`positions`、`tree_mask`、`retrive_index`、`retrive_next_token`、`retrive_next_sibling`、`draft_token_num`
     - `model_worker_batch.input_ids`
     - `out_cache_loc`（写入的槽位）
     - 前向后：`logits_output.next_token_logits` 在每个 verify 位置的 argmax
     - `spec_info.accept_length`、`next_token_ids`
3. **阶段 B — MedusaWorker 测试运行**：
   - 用 `--speculative-algorithm MEDUSA --speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2` 重启 Engine。
   - 同 prompt、同 prefill、1 步 verify。
   - dump 相同字段。
4. **阶段 C — Diff**：
   - (A) 与 (B) 逐字段 diff。
   - 打印结构化报告：`[字段] [A 值] [B 值] [是否不同] [已知原因]`。

**脚本必须包含的合理性检查**：
- 两次运行使用相同模型权重（GPTQ + FP8 KV）。
- 两次运行使用相同 RNG seed 和 decode 模式（`temperature=0`、贪心）。
- Ngram 运行时强制 ngram_cache 返回已知 token 序列（或使用预热 cache），保证 `draft_token[0..1]` 确定；否则比较的 draft-内容轴不受控。
  - **如果预热困难的回退**：仅在本实验中向 NgramWorker 注入 `_force_draft_tokens=[tok0, tok1]` 覆盖（任何真实评测前撤回）。
- Medusa 运行使用相同的 `tok0`、`tok1` 强制 draft 时，diff 隔离**仅元数据**差异，与 draft 内容差异分开。

**每轮迭代预估时长**：fcloud 上约 5–10 分钟（engine init 占大头；两次单步运行只是秒级）。

## 4. Diff 迭代协议（强制）

每次 diff 迭代遵循此协议。**所有迭代追加到同一文档对**，作为编号子章节（`§5.1`、`§5.2`、…）。

对 MedusaWorker (B) 与 NgramWorker (A) 之间发现的每个 diff：

| 字段 | A (Ngram) | B (Medusa) | 不同？ | 严重性 | 决定 | 原因 |
|---|---|---|---|---|---|---|
| （每行一个被审字段） | | | 是/否 | critical / cosmetic / unknown | clean / keep / investigate | （一句话理由） |

**严重性标签**：
- **critical**：改变模型前向语义或 KV 写入。fcloud 评测前必须 clean。
- **cosmetic**：不影响 KV/logits/accept 流程（例如 kernel 都接受的 `int64` vs `int32`）。可保留并记录。
- **unknown**：不明显哪侧对。要么 (a) 加探针重跑预飞行，要么 (b) 尝试 clean 成与 Ngram 一致后重跑预飞行看是否改善。

**决定规则**：
- **clean** = 修改 MedusaWorker 使其与 NgramWorker 产生相同值。
- **keep** = 保留 Medusa 版本，但说明为何是有意为之且不破坏行为。
- **investigate** = 需要更多探针；不改代码，加诊断后重跑。

每次迭代结束时：
- 更新 MedusaWorker 代码（`python/sglang/srt/speculative/medusa_worker.py` 与提交镜像）。
- Commit 信息：`medusa: preflight iter N — clean <field>` 或 `medusa: preflight iter N — investigate <field>`。
- Push 到 `minicpm-src`。
- 在本文档追加 §5.N，包含上表 + 代码 diff 摘要。
- fcloud 上重跑预飞行确认 diff 消失（且未出现新 diff）。
- Pause fcloud。

**退出条件**：预飞行报告 zero **critical** diff。**cosmetic** diff 可保留并记录。此时跑完整准确率 + 速度评测；若通过（acc ≥ 78%，S1 ≤ Stage 3a S1），交付 Stage 3b ndt=2；否则 §1 的工作假设是错的，在新 CHANGE_xxxx 下开启新的调查。

## 5. 迭代日志

（空 — 每次预飞行运行后填充。）

## 6. 风险

- **R1**：Engine 启动非确定性（如 flashinfer kernel JIT 编译顺序）可能改变 A 与 B 运行之间的 KV 布局。缓解：尽量在两个阶段间持久化 Engine；否则固定随机 seed，并在同一 Python 进程内紧接 A 运行 B。
- **R2**：在 Ngram 中强制 draft token 以求确定性可能不直观。回退：dump *两侧*值，接受 draft 内容差异，只 diff **元数据**字段（positions、tree_mask、retrive_*）。draft 内容在 §5 报告中归一化处理。
- **R3**：若 bug 在 **post-verify**（在 `spec_info.verify()` accept 流程消费错误 logits 索引）中，单步预飞行可能显示输入干净但输出错误。缓解：dump `next_token_logits` 以及 `accept_length` 以及 `next_token_ids`，使 verify 步本身也被 diff。
- **R4**：Stage 3b 已训练头路径增加第 6 个比较轴（在捕获 hidden 上跑头前向）。等到 Stage 3a 等价 ndt=2 干净后再处理这点。首次预飞行对 A 的 draft-内容覆盖 AND B 的自然路径都使用 Stage 3a 回退（`output_ids[-1]` 作为 draft）。

## 7. 批准门

任何 fcloud 操作之前，agent 将：
- (a) 展示本文档供用户审阅。
- (b) 展示拟定的预飞行脚本内容。
- (c) 等待对脚本内容 + fcloud start-instance 的明确"go"。

每轮迭代之后，agent 将：
- (a) 展示 §5.N 表格。
- (b) 展示拟改代码（如有）。
- (c) 等待对代码改动 AND 下轮 fcloud 的明确"go"。

## 8. 验证命令

预飞行（每次迭代，在 fcloud 上）：
```bash
# 用户启动 fcloud（JWT 检查后）
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync

# 跑预飞行（约 5–10 分钟）
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && source prepare_env.sh && \
   python3 preflight_medusa_vs_ngram.py 2>&1 | tee /tmp/preflight_iterN.log'

# 拉回报告
python3 scripts/fcloud/fcloud_exec.py download \
  --remote /tmp/preflight_iterN.log \
  --local benchmark/soar/demo_sala/preflight_iterN.log

python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

最终评测（仅在 zero critical diff 后）：
```bash
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py full     # sync + restart + accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

## 9. 回滚

若协议结束仍未得到可用 ndt=2：
- 撤回 `medusa_worker.py` 中任何 Stage 3b 相关编辑（当前基线 = `a489d78d4` Stage 3a）。
- Stage 3a 交付配置保持不变。
- 在 §5 文档化失败方向分析，并开新的 CHANGE_xxxx 探索替代方向（如 Medusa 走 Eagle 路径而非 Ngram 路径）。

## 10. 下一步建议（迭代后）

- 若干净的 ndt=2 交付且 acc ≥ 78%、S1 < Stage 3a S1：通过 tree drafts 扩展到 ndt > 2（Stage 4 范畴，K=2/K=4）。
- 若干净的 ndt=2 交付但加速收益边际：profile verify 前向以确认 kernel 开销 vs 仅 bonus 基线 — 若 verify_forward / decode_forward > 2× 则 K=1 无法获胜。
