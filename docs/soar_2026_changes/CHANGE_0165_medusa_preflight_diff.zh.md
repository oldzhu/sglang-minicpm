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

### §5.1 第 1 轮 — Stage 3a 基线（commit f375082a2 + 19740212d）

**配置**
- 驱动脚本：`benchmark/soar/demo_sala/preflight_drive.sh {ngram|medusa}`（提交 d9394bec6 → 19740212d）。
  - 剥离 `--enable-torch-compile` 与 `--torch-compile-max-bs N`，并强制 `--disable-cuda-graph`，把启动时间从 ~15 分钟（16 个 batch 桶全捕获）缩短到 ~2 分钟 —— pre-flight 只需要一次 verify。
  - 每个模式启动独立 sglang server：`bs=1`，`ndt = {ngram:2, medusa-stage-3a:虽然传入 H+1=2 但 MedusaWorker 内部硬编码为 1}`。
- 探针 prompt：`"Question: What is the capital city of France?\nAnswer:"`，`temperature=0.0`，`max_new_tokens=3`。
- 双侧各在 3 个 phase dump：`pre_verify`、`post_forward`、`post_verify`。
  - `/tmp/dump_ngram.pkl` = 41908 B（27 条记录：9 次 verify × 3 phase）
  - `/tmp/dump_medusa.pkl` = 41207 B（27 条记录）
- diff 工具：`benchmark/soar/demo_sala/preflight_diff.py --ngram … --medusa …`（对比双侧**第一次** verify）。

**结果（13 个字段不同）**

| Phase | 字段 | ngram | medusa (stage 3a) | 严重级别 | 决策 | 原因 |
|---|---|---|---|---|---|---|
| pre_verify | `draft_token_num` | 2 | 1 | **关键 / 根因** | iter 2 清理 | MedusaWorker 强制 `ndt=1`，无视 `--speculative-num-draft-tokens` 参数。**所有 shape 差异都从这里级联**。 |
| pre_verify | `input_ids` | (2,) `[11225, 0]` | (1,) `[11225]` | 关键（级联） | ndt=2 后消失 | 多出的 slot = 填充 draft token `0` |
| pre_verify | `out_cache_loc` | (2,) `[8, 9]` | (1,) `[9]` | 关键（级联） | ndt=2 后消失 | 两个 KV 槽 vs 一个 |
| pre_verify | `spec_draft_token` | (2,) `[11225, 0]` | (1,) `[11225]` | 关键（级联） | ndt=2 后消失 | 同 `input_ids`（ngram 传入 `output_ids[-1]` + 一个 ngram 建议 token 用 `0` 填充） |
| pre_verify | `spec_positions` | (2,) `[7, 8]` | (1,) `[8]` | 关键（级联） | ndt=2 后消失 | 从当前 `seq_len` 跨越 2 个 slot 的位置 |
| pre_verify | `spec_custom_mask` | (18,) | (8,) | 关键（级联） | ndt=2 后消失 | 全 mask 布局 `(seq_len + ndt) * ndt` ⇒ `(7+2)*2=18` vs `(8+0)*1=8`。编码 `ndt` 个 query × `seq_len + ndt` 个 key 的对角因果 mask。 |
| pre_verify | `spec_retrive_index` | (1, 2) `[0, 1]` | (1, 1) `[0]` | 关键（级联） | ndt=2 后消失 | 树检索索引；K=1 链 shape = `(bs, ndt)` |
| pre_verify | `spec_retrive_next_token` | (1, 2) `[1, -1]` | (1, 1) `[-1]` | 关键（级联） | ndt=2 后消失 | 链指针（0→1→叶） |
| pre_verify | `spec_retrive_next_sibling` | (1, 2) `[-1, -1]` | (1, 1) `[-1]` | 关键（级联） | ndt=2 后消失 | 链无 sibling |
| pre_verify | `seq_lens` / `seq_lens_cpu` | 7 | 8 | 表面 / 衍生 | 接受 | `ndt=1` 比 `ndt=2` 少消费一个 token，所以 `ndt=1` 多跑一轮 decode 才追上 —— 步数计数偏移，不是算法 bug。 |
| post_forward | `logits_argmax` | (2,) `[72, 72]` | (1,) `[72]` | 关键（级联） | ndt=2 后消失 | 直接由输入 shape 差异引起；**重叠 slot 上的值相同**（`72`） → 给定输入下 forward 是对的。 |
| post_forward | `logits_shape` | `(2, 73448)` | `(1, 73448)` | 关键（级联） | ndt=2 后消失 | 同上 |
| post_verify | `accept_length`, `accepted_indices`, `next_token_ids`, `num_accepted_tokens` | 全相等 | 全相等 | n/a（干净） | 保留 | verify 步本身在被接受输出上一致（`next_token=72`，0 个 bonus 被接受）。确认 **verify 走树逻辑不是 bug**。 |

**结论**

- 单一根因：Stage 3a 的 `MedusaWorker.draft_token_num = 1`。所有 shape 差异都从此级联。verify 走树 / tree-index / 接受打分逻辑没有 bug —— 喂相同 shape 的输入，模型产生匹配的 logits，verify 步给出一致的输出。
- 这是 Iter 1 的**预期**基线（按 CHANGE_0165 §0 计划 Stage 3a 不动代码）：我们需要在动代码前先用实证确认所有差异都收敛到一个根因。

**Iter 2 计划**

1. 修改 `MedusaWorker`：(a) 让 `draft_token_num=2` 生效（匹配 `--speculative-num-draft-tokens 2`）；(b) 用 NgramWorker 同样的 `reconstruct_indices_from_tree_mask` kernel 构建 `spec_positions`、`spec_custom_mask`、`spec_retrive_*`；(c) K=1 链的 draft token 用 `[output_ids[-1], 0]`（填充） —— 与 Ngram 的 draft 内容 shape 一致，保持 Stage 3a 语义（head 0 实际未被使用；训练好的 head 路径是 Stage 3c）。
2. 重跑 `preflight_drive.sh medusa`，预期 §5.2 报告 `pre_verify` / `post_forward` **0 个关键差异**，`post_verify` 完全相等。
3. 仅在那之后跑完整精度 + 速度评测。

**产物**

- Dump：fcloud `/tmp/dump_ngram.pkl`、`/tmp/dump_medusa.pkl`（已同步到本地 `/tmp/`）。
- diff 日志：`/tmp/iter1_diff.txt`。
- Server 日志：fcloud `/tmp/server_ngram.log`、`/tmp/server_medusa.log`。
- 本轮提交：`d9394bec6`（preflight 基础设施）、`11555decd`（驱动脚本）、`283c8a106`（去掉 set -e）、`f375082a2`（导出 MODEL_PATH）、`19740212d`（关闭 graph+compile）。

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
