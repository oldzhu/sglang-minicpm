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

### §5.2 第 2 轮 —— ndt=2 + 内核生成 retrive_* + 全掩码（提交 `89a30d5d0`）

**Setup**

- 代码改动（单个提交，`python/sglang/srt/speculative/medusa_worker.py`）：
  1. `self.draft_token_num = self.num_heads + 1`（K=1 时 =2），不再是 `=num_heads`。
  2. `_forward_verify_k1` 改写，镜像 NgramWorker 布局：
     - 每个 req 构造 draft 链 `[base, head_pred or 0]`（Stage 3a：head 未训练 → `head_pred=0`）。
     - 构造 `(ndt, ndt)` 下三角 tree mask，按 USE_FULL_MASK 拼成 `(ndt, seq_len-1+ndt)`，跨 batch 串接。
     - 分配 `positions`、`retrive_index`、`retrive_next_token`、`retrive_next_sibling`，由 `sgl_kernel.speculative.reconstruct_indices_from_tree_mask` 填充 —— 与 NgramWorker 同一内核。
  3. 训练 head 的 hidden state 切片：`hs.view(bs, ndt, -1)[:, 0, :]`，因为现在 `hs.shape[0] == bs * ndt = 2`。
- Pre-flight 执行：同一驱动、同一 prompt。Ngram dump 41908 B / 27 records；**Medusa dump 23529 B / 15 records**（少一步 verify，因为探针完成后 scheduler 在 idle KV 检查时崩溃 —— 探针后才发生，不影响已记录数据：pre/post/post 三阶段完整）。

**结果（差异由 13 项降到 5 项）**

| Phase | 字段 | ngram | medusa（iter 2） | 严重度 | 诊断 |
|---|---|---|---|---|---|
| pre_verify | `draft_token_num` | 2 | **2** | ✅ EQUAL | 修复 #1 生效。 |
| pre_verify | `input_ids` | (2,) `[11225, 0]` | (2,) `[11225, 0]` | ✅ EQUAL | ndt=2 布局已采用；pad=0 一致。 |
| pre_verify | `spec_draft_token` | (2,) `[11225, 0]` | (2,) `[11225, 0]` | ✅ EQUAL | Stage 3a head 返回 0 → 与 ngram fallback 完全一致。 |
| pre_verify | `spec_custom_mask` | (18,) | (18,) | ✅ EQUAL | USE_FULL_MASK + 下三角拼装得到相同布局。 |
| pre_verify | `spec_retrive_index` / `next_token` / `next_sibling` | (1,2) | (1,2) | ✅ EQUAL | `reconstruct_indices_from_tree_mask` 内核生成相同树。 |
| pre_verify | `seq_lens` / `seq_lens_cpu` | 7 | **8** | 语义性 off-by-one | **剩余根因**。MedusaWorker 进入 verify 时 `seq_lens` 已比 ngram 多 1。 |
| pre_verify | `spec_positions` | `[7, 8]` | `[8, 9]` | seq_lens 级联 | 内核写 `[seq_len-1, seq_len, ...]`，因 seq_lens=8 整体 +1。 |
| pre_verify | `out_cache_loc` | `[8, 9]` | `[9, 10]` | seq_lens 级联 | KV 分配器给出后续 2 个空槽，因 medusa 多占 1 槽。 |
| post_forward | `logits_argmax[0]` | 72 | 72 | ✅ slot 0 相等 | 形状一致后模型在 base 位置产生相同 token。 |
| post_forward | `logits_argmax[1]` | 72 | **59320** | seq_lens 级联 | 模型在 position 9 vs 8 不同 KV 历史下被调用 → speculative slot 的 logits 不同。 |
| post_forward | `logits_shape` | (2, 73448) | (2, 73448) | ✅ EQUAL | 布局已修。 |
| post_verify | accept_length / accepted_indices / next_token_ids / num_accepted_tokens | 全部相等 | 全部相等 | ✅ EQUAL | 虽然 slot 1 logits 不同，accept_length=0（draft 为 `0` 不会被接受）→ 两边 next_token=72。verify-walk 干净。 |

**结论**

- Iter 2 完全实现提案中的布局修复：**之前破裂的 8 个字段全部对齐**（`draft_token_num`、`input_ids`、`spec_draft_token`、`spec_custom_mask`、`spec_retrive_index`、`spec_retrive_next_token`、`spec_retrive_next_sibling`、`logits_shape`）。`post_verify` 保持 4×EQUAL。
- **唯一剩余根因**是 MedusaWorker 中 `seq_lens` 比 ngram 多 1。这就是 CHANGE_0160 / CHANGE_0161 / `PROPOSAL_medusa_k1_positional_offbyone_fix` 一直在外围打转的"位置 off-by-one"。pre-flight 框架将其锁定为**单个字段**（`seq_lens`）在**单个阶段**（`pre_verify`）的偏差 —— 消除了布局级联带来的歧义。

**+1 从哪里来？**

待 iter 3 验证的假设：
1. **H1（最可能）** —— Stage 3a "bonus 位"的 KV 写入：上一次 decode 步骤把 bonus token 的 KV 写到 `seq_len` 处，在 `prepare_for_verify` 之前就把 `seq_lens` 推进了 1。NgramWorker 不写 bonus KV，所以保持原值。
2. **H2** —— MedusaWorker 的 `prepare_for_verify` 错误地把 `seq_lens` +1（本应由 verify forward 在执行时按 ndt 推进，而非提前）。
3. **H3** —— 探针是*第二个* batch 步骤（首个 decode 已输出 11225），Medusa 在那个初始 decode 的 "extend" 模式下错误地提交了两个 KV 槽。

Pre-flight 可以增加第 4 阶段 `pre_prepare_for_verify`（进入 `spec_info.prepare_for_verify` 之前）来区分：如果差异在此阶段就出现 → 是 decode 端记账问题（H1/H3）；如果只在 `prepare_for_verify` 之后才出现 → 是 H2。

**Iter 3 计划（提案 —— 需用户"go"）**

1. 在两个 worker 中加入 `phase="pre_prepare_for_verify"` dump，记录 `seq_lens`、`seq_lens_cpu`、`out_cache_loc`、`req_pool_indices`。
2. 再跑一次 preflight，做 diff。
3. 若 H1/H3：定位 MedusaWorker 主 decode 中提交 bonus 槽的位置 → 要么不提交，要么 verify 前把 `batch.seq_lens` 减 1。按 CHANGE_0160/0161 的经验，**正确**做法是源头不双计 bonus 槽。
4. 若 H2：简化 `prepare_for_verify` 去掉多余的 +1。
5. 迭代到 pre_verify 0 关键差异，然后再跑 accuracy。

**Iter 3 通过条件**：`seq_lens`、`seq_lens_cpu`、`spec_positions`、`out_cache_loc`、`logits_argmax` 全部 ngram == medusa。`post_verify` 继续 4×EQUAL。

**产物**

- Dumps：fcloud 上的 `/tmp/dump_ngram.pkl`、`/tmp/dump_medusa.pkl`（本轮）。
- Diff 日志：`/tmp/iter2_diff.txt`（本地也有副本）。
- 本轮提交：`89a30d5d0`（medusa: preflight iter 2 — adopt ndt=2 + kernel-built retrive_* + full mask）。

### 5.3 第 3 轮结果 —— 定位到 bonus 槽位的记账问题

**目标**：通过在两个 worker 中 `spec_info.prepare_for_verify(batch, page_size)` 调用前增加第四个 dump 阶段 `pre_prepare_for_verify`，捕获该调用前的 batch 状态，区分假设 H1/H3（verify-prep 之前上游就提交了 KV 槽位）与 H2（`prepare_for_verify` 内部进行了 +1）。

**代码改动**（提交 `3d2435d26` + `346c3f666`）：
- `ngram_worker.py`：在 `batch.spec_info.prepare_for_verify(...)` 之前 dump（`seq_lens`、`seq_lens_cpu`、`out_cache_loc`、`req_pool_indices`）。
- `medusa_worker.py`：在 `spec_info.prepare_for_verify(...)` 之前做相同的 dump。
- `preflight_diff.py`：将新阶段加入 `PHASE_ORDER` 和 argparse `choices`。

**结果（pre_verify 仍有 5 字段差异；pre_prepare_for_verify 有 3 字段差异）**：

| 阶段 | 字段 | ngram | medusa | 状态 |
|---|---|---|---|---|
| pre_prepare_for_verify | `seq_lens` | `[7]` | `[8]` | **不同** |
| pre_prepare_for_verify | `seq_lens_cpu` | `[7]` | `[8]` | **不同** |
| pre_prepare_for_verify | `out_cache_loc` | `[1,2,3,4,5,6,7]`（shape 7） | `[8]`（shape 1） | **不同（形状+取值）** |
| pre_prepare_for_verify | `batch_size`、`req_pool_indices` | — | — | 相同 |

**解释 —— H1/H3 确认，H2 排除**：

`seq_lens` 的 +1 **早于** `prepare_for_verify` 调用就已经存在。因此 `prepare_for_verify` 本身无罪；多分配的那个 KV 槽位是在 **EXTEND 完成 → 下一轮 DECODE/verify 入口** 这段上游流程里提交的。

**根因定位** 在 [`schedule_batch.py`](../../python/sglang/srt/managers/schedule_batch.py) 的 `prepare_for_decode()` 第 1948 行：

```python
def prepare_for_decode(self):
    ...
    if not self.spec_algorithm.is_none():
        return  # spec worker 自行管理 decode-prep
    # 否则：alloc 1 个槽位，seq_lens.add_(1)，kv_committed_len += 1
```

以及两个 worker 在 EXTEND 路径上的差异：
- **NgramWorker** EXTEND：`_prepare_for_speculative_decoding` 提前返回；`spec_algorithm` 保持 `NGRAM` → 下一轮 `prepare_for_decode` 走早返回 → `seq_lens` 保持为 7。
- **MedusaWorker** EXTEND：显式设置 `batch.spec_algorithm = SpeculativeAlgorithm.NONE` → 下一轮 `prepare_for_decode` 进入正常分支 → 分配 1 个槽位、`seq_lens` 推到 8、`kv_committed_len += 1`。当 MedusaWorker 再次进入 DECODE 重新构建 spec_info 时，bonus 槽位已经被提交。

这正是 CHANGE_0160 / CHANGE_0161 追了很久的“+1 之谜”，现在终于精确定位。

**产物**

- Dumps：fcloud 上 `/tmp/dump_ngram.pkl`（52937B / 36 条记录）、`/tmp/dump_medusa.pkl`（29590B / 20 条记录）。
- Diff 日志：`/tmp/iter3_diff.txt`（全阶段）、`/tmp/iter3_pre_prep.txt`（仅 pre_prepare_for_verify）。
- 本轮提交：`3d2435d26`（iter3 dump 增量）、`a87da57dd` + `346c3f666`（preflight_diff 阶段注册修复）。

### 5.4 第 4 轮计划 —— 删除 MedusaWorker EXTEND 中的 spec_algorithm 重置

**拟修复**（`medusa_worker.py::forward_batch_generation` 中一行）：

```python
# EXTEND 路径
if batch.forward_mode.is_extend():
-    batch.spec_algorithm = SpeculativeAlgorithm.NONE   # 删除此行
    model_worker_batch = batch.get_model_worker_batch()
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
    return GenerationBatchResult(...)
```

**安全性说明**：
1. NgramWorker（标准参照）在 extend 时不重置 `spec_algorithm`。
2. EXTEND 的 `model_worker_batch` 已经以 `forward_mode=EXTEND` 构建，目标模型无论 `spec_algorithm` 取何值都会正常跑 extend。
3. 保持 `spec_algorithm=NGRAM` 使得 scheduler 的 `prepare_for_decode` 走早返回路径（与 ngram 完全一致），把 KV/seq_lens 的管理完全交给 `prepare_for_verify`——这正是设计意图。

**验证计划**：
1. 应用修改并 push。
2. 在 fcloud 上重新运行 `preflight_drive.sh ngram` + `preflight_drive.sh medusa`。
3. 再次执行 `preflight_diff.py` —— 预期所有 4 个阶段（pre_prepare_for_verify、pre_verify、post_forward、post_verify）的差异字段数均为 **0**。
4. **通过判据**：4 个阶段的差异总数 = 0。

**风险**：
- R-iter4-1：如果 scheduler 中有针对 EXTEND 输出的、专门检查 `spec_algorithm.is_none()` 的分支（例如 `scheduler_output_processor_mixin.py` 第 372 行的 `next_token_ids.tolist()`），行为可能改变。缓解措施：应用前先阅读 `scheduler_output_processor_mixin.py` 中的 EXTEND 结果分支；如果 `is_none()` 分支里有 MedusaWorker 必需的逻辑，需要找替代方案（例如包装 target_worker 调用、手动执行每请求 append）。

下一步（待 preflight 差异归零之后）：
- 通过切回标准 `fcloud_workflow.py restart-server` 流程，重新启用 cuda-graph + torch-compile（不再使用 `preflight_drive.sh`）。
- 运行完整 accuracy 测试 + S1/S8/Smax 速度测试（按项目规则需用户显式确认）。

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
