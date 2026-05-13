# CHANGE_0164 — Medusa Stage 3b K=1 重构为标准 `draft_token_num=2` 布局

日期：2026-05-13
分支：`mixed_minicpm_cudagraph`
相关文档：`PROPOSAL_medusa_stage3b_k1_draft_token_num_2.{en,zh}.md`、
          `CHANGE_0163_medusa_stage3b_trained_heads.{en,zh}.md`、
          `CHAT_medusa-stage3b-distribution-mismatch_20260513_0337.{en,zh}.md`

## 背景与动机

CHANGE_0163 引入了使用 `draft_token_num=1` verify 布局的 Stage 3b
Medusa worker。三轮完整的 fcloud 迭代（S1 = 202–217 s，accept_len 始终 = 1.00）
表明无论 head 训练质量如何，draft 都**永远无法被接受**。
阅读 `sgl_kernel::VerifyTreeGreedy` CUDA 内核
(`sgl-kernel/csrc/speculative/eagle_utils.cu`) 后发现存在两个耦合的 bug：

1. **速度 bug** —— 内核将 `retrive_index[0]` 视为"root，永远接受"，
   并将下标 1..ndt-1 处的子节点和 `target_predict[root]` 进行比对。
   当 `ndt=1` 时，没有任何子节点，因此 draft 永远无法被验证。
2. **正确性 bug** —— 内核的最后一行
   `predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx]`
   提交的 token 是基于 `(prefix + draft)` 上下文得到的模型预测。
   而我们 Stage 3a 回退路径 (`draft = output_ids[-1]`) 把这个位置塞成了
   "上一个已提交的 token T_N"，于是提交的 token 实际上是基于
   `prefix + T_N + T_N`（重复）采样的，而不是 `prefix + T_N` —— 每次
   Medusa decode 步骤都引入一个小但结构性的分布偏置，这与 v23-medusa-passthrough
   提交在官方评测中得到 78.71 %（vs v22 基线 79.29 %）的结果一致。

EAGLE / NGRAM 的标准布局使用 `ndt = num_drafts + 1`：

* position 0 = **bonus** = 上一步已提交的 token（`output_ids[-1]`）
* position 1..ndt-1 = 推测 draft

`prepare_env.sh` 已经导出 `NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))`
（K=1 时 = 2），所以服务端参数面已经正确，只有 worker 代码不一致。

## 合规性说明

* 不改动官方评测脚本或评分路径。
* 不改动模型权重、量化、KV-cache dtype。
* 不改动 `--force-dense-minicpm` + FP8 KV + Tier1 长上下文这套基线。
* 服务端参数表面不变 —— 前后 sglang 看到的都是
  `--speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2`。
* 只改 MedusaWorker 内部的 verify 构造。

## 实现计划（改动前）

1. `__init__`：把 `self.draft_token_num` 从 `num_heads`（=1）改为
   `num_heads + 1`（=2）。
2. `_forward_verify_k1`：
   * 构造形状 `(bs*2,)` 的扁平 `draft_token`，按
     `[output_ids[-1], head_pred_or_fallback]` 交错存放。
   * `retrive_index = arange(bs*2).view(bs, 2)`。
   * `retrive_next_token = [[1, -1], ...]`（bonus → draft 线性链）。
   * `retrive_next_sibling = [[-1, -1], ...]`。
   * `positions = seq_lens[i] + arange(2)` 后展平。
   * `tree_mask` 按 NGRAM `USE_FULL_MASK` 惯例：每个请求
     `(ndt, seq_len_i - 1 + ndt)`，前缀全 1，尾部 `(ndt, ndt)`
     下三角，展平后 concat。
   * `CaptureHiddenMode.FULL`（需要 bonus 位置的 hidden，而不是
     LAST）。
   * forward 之后把 `hidden_states` reshape 成 `(bs, ndt, h)`，取
     position 0 —— 即 bonus 位置的 hidden state，它代表模型消费
     `prefix + T_N` 之后的状态，与 v1 训练器拟合的
     `CaptureHiddenMode.LAST` 视图字节等价。
   * 该 bonus hidden 同时供 head 前向（下一步 draft）与 offline dump 使用。
3. 移除已失效的 `CHANGE_0160` bonus 清零循环（它只在
   `accept_length >= 1` 时触发，而 ndt=1 下从未触发；ndt=2 时
   `NgramVerifyInput.verify` 的标准 `_free_cache` 路径会正确管理
   `req_to_token`）。

## 实际代码改动（改动后）

* `python/sglang/srt/speculative/medusa_worker.py`
* `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py`
  （通过 `cp` 完全同步副本，确保 demo_sala 提交 tarball 中携带的
  worker 与 upstream 树完全一致）

两份文件现在都满足：

* `self.draft_token_num = self.num_heads + 1`（K=1 时 = 2）。
* `_forward_verify_k1` 构造上述 2 节点 verify 输入。
* 当训练 head 启用 或 hidden 采集模式启用时使用 `CaptureHiddenMode.FULL`。
* reshape 后的 hidden states 的 position 0 同时供 head 与 dump 使用。

## 预期行为

* **每一步 Medusa decode 的正确性恢复**：bonus 位置始终提交
  `target_predict[0]` —— 即基于 `prefix + T_N` 条件计算的 logits
  的 argmax。该 token 与非推测 decode 输出字节等价。
  最差情况（draft 被拒）只提交 1 个正确 token，与标准 decode 一致。
* **速度**：如果训练 head 的 draft 等于模型真实下一个 token
  （离线 shifted-label 匹配率 99.91 %），内核 walk 子节点、接受、
  每步提交 2 个 token（`accept_len ≈ 2.0`）；如果 head 不命中，
  退化为每步 1 个 token（= 标准 decode 速度，无回归）。
* **Stage 3a 回退**（未加载训练 head，`SOAR_MEDUSA_HEAD_PATH=""`）：
  draft = 重复的最后一个 token → 始终被拒 → 每步只提交 1 个正确 token。
  消除了静默 corruption，但也不会有加速（准确率回到基线）。

## 验证命令

```bash
# fcloud 一轮（需用户确认）：
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# (a) 正确性：Stage 3a 回退应回到 v22 基线准确率
SOAR_SPEC_MEDUSA=1 SOAR_MEDUSA_HEAD_PATH="" \
  python3 scripts/fcloud/fcloud_workflow.py accuracy
# 期望：acc ≈ 79.3%（vs 当前 v23 77.87%）

# (b) 加载训练 head v2 后的速度与 accept_len：
SOAR_SPEC_MEDUSA=1 \
SOAR_MEDUSA_HEAD_PATH=/root/models/medusa_head_v2.pt \
  python3 scripts/fcloud/fcloud_workflow.py speed --variant all
# 期望：若 head 能迁移，S1 从 216 s → 120–140 s，
#       accept_len ≈ 1.8–2.0；否则不应比当前 S1 = 202 s 差。

python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

## 结果汇总表（fcloud 跑完后填写）

| 配置 | S1 (s) | S8 (s) | Smax (s) | accept_len | acc (%) | 备注 |
|---|---|---|---|---|---|---|
| v22 基线（无 Medusa）            | 121.71 | 44.09 | 35.86 | n/a  | 79.29 | 参考 |
| CHANGE_0163 + ndt=1（坏的）      | 202.70 | 61.60 | 43.29 | 1.00 | 77.87 | corruption bug |
| CHANGE_0164 + ndt=2 Stage 3a    |  TBD   |  TBD  |  TBD  | 1.00 |  TBD  | 仅修正正确性 |
| CHANGE_0164 + ndt=2 Stage 3b v2 |  TBD   |  TBD  |  TBD  |  TBD |  TBD  | 加载训练 head |

## 回滚说明

回滚到 CHANGE_0163（ndt=1）行为：

```bash
git revert <this-commit>
# 或手动改：
#   self.draft_token_num = self.num_heads        # 原值 num_heads + 1
#   _forward_verify_k1 回到单 position 布局
```

提交回滚：在 `prepare_env.sh` 中（或启动时覆盖）设置
`SOAR_SPEC_MEDUSA=0` —— 完全关闭 worker，回到纯 v22 行为。

## 下一步建议

1. 若 accept_len ≥ 1.8 且 S1 ≤ 140 s：打包为 v24 提交。
2. 若 accept_len ≈ 1.0–1.3：排查训练 / 推理分布失配（运行时采样
   bonus 位置 hidden state，与离线训练器的 hidden 分布对比）。
3. 若准确率掉到 79 % 以下：审视 ndt=2 下 `_free_cache` 与
   `req_to_token` pool 的交互（`CHANGE_0160` 那个补丁是给 ndt=1
   设计的；ndt=2 走 NGRAM 标准 eviction 路径，理论上正确，但建议
   先在 10 条样本子集上跑一次冒烟测试）。

---

## 结果（2026-05-13）—— 灾难性失败，必须回退

**测试 ID**：`Stage3b-ndt2-CATASTROPHIC`（commit `4b442f421`，fcloud `ai-e7e98a7c52`）。

| 指标 | 数值 | 与基线（Test 12 = 79.29%）对比 |
|------|------|------------------------------|
| ori_accuracy | **15.13 %** | −64.16 pt |
| normalized   | **18.92 %** | 远低于 97 % → **C = 0（淘汰）** |
| mcq          | **0.00 %**  | 失控：平均输出 64460 tokens（吃满 max_out_len）|
| niah         | 3.33 %      | −96.67 pt |
| cwe          | 15.67 %     | −56.33 pt |
| fwe          | 20.00 %     | −78.89 pt |
| qa           | 36.67 %     | −26.66 pt |
| 评测时长     | 7911 s（2 h 12 m）| 约为基线 2.6× |

kernel 一路报告 `accept_len = 1.46, accept_rate = 0.73`，结构上推测流水正常，但提交的 token 是错的。每条 MCQ 都把 65536 max_out_len 吃满，强烈暗示 bonus 位置输出的 token 已被破坏（永远输出不到 stop token）。

### 假设（尚未二分定位）

1. **bonus 的 position 不对**：我们用 `positions = seq_lens + arange(ndt)`，预期 bonus 在 `seq_lens` 槽位、draft 在 `seq_lens+1` 槽位。NGRAM 自带的 `_prepare_for_speculative_decoding` 调用 `reconstruct_indices_from_tree_mask(..., batch.seq_lens, positions, ...)` 从 tree mask 反推 positions。如果 NGRAM 约定 verify 树的 root 应位于 `seq_lens-1`（覆盖最后已提交 token 的槽位），那我们的 bonus KV 写到了错的行，后续读到的全是脏数据。
2. **trained head 的 hidden 分布不匹配**：head 是在 ndt=1 路径上采集的 `hidden[:,0,:]` 上重训的。在 ndt=2 下，position-0 hidden 的 attention 上下文不同；若 tree_mask 设置有偏差，捕获到的训练分布与在线推理分布发散，会让 head 草稿胡说。但单凭这点只会让 accept rate 变差，不应破坏 bonus 输出，所以这不是单独原因。
3. **`prepare_for_verify` 对 `seq_lens` 的副作用**：共享的 `NgramVerifyInput.prepare_for_verify` 内部可能把 `seq_lens` 加上 `ndt`。如果我们的 `positions` 是在调用之前就算好的，那真正的 forward 写 KV 时会比预期多偏移 `ndt`。

### 决策：回退

- 把 `medusa_worker.py`（两份拷贝）回到 Stage 3a `ndt=1` 路径（commit `3a15a6de3` / `Stage3a-force-dense` 基线：78.40 % acc，S1=204.86 s）。
- 保留 CHANGE_0164 文档作为失败记录。
- 立项后续调查：在 `_forward_verify_k1` 中 dump 一次 micro-batch 的 `(positions, seq_lens_in, seq_lens_out, draft_tokens, input_ids_used_by_attention, committed_token)` 五元组，与单 token NgramWorker decode 同 prompt 逐字节比对，先定位 off-by-one 再重新尝试 ndt=2。

### 回退命令

```
git revert 4b442f421       # 或
git checkout 3a15a6de3 -- python/sglang/srt/speculative/medusa_worker.py \
                          benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py
```

已作为 commit `a489d78d4` 推到 `minicpm-src/mixed_minicpm_cudagraph`（2026-05-13）。

---

## 回退后根因分析（2026-05-13，离线）

回退之后阅读源码，定位了真正的 bug 类型，并排除了之前三个假设中的两个。

### 已阅读

1. [python/sglang/srt/speculative/ngram_info.py](../../python/sglang/srt/speculative/ngram_info.py) 第 50–115 行（`NgramVerifyInput` 构造 + `prepare_for_verify`），第 374–446 行（`verify`）。
2. [python/sglang/srt/speculative/ngram_worker.py](../../python/sglang/srt/speculative/ngram_worker.py) 第 142–200 行（`_prepare_for_speculative_decoding`，是 `reconstruct_indices_from_tree_mask` 的正典调用方）。
3. [sgl-kernel/tests/speculative/test_ngram_utils.py](../../sgl-kernel/tests/speculative/test_ngram_utils.py)（单一正典 kernel 单元测试，钉死磁盘约定：bs=1，ndt=4，seq_lens=[12] → `positions = [12, 13, 13, 14]`）。

### 已确认的事实

- **`NgramVerifyInput` 是通用的 tree-verify 基础设施，不是 NGRAM 算法专属。** 类名很糟糕 —— 应该叫 `TreeVerifyInput`。`SpecInputType.NGRAM_VERIFY` 只是个 tag；`verify_tree_greedy` 里没有任何 NGRAM 特有逻辑。Medusa 复用它是正确且有意为之的（Stage 3a 已经这么做，且是目前最稳的 Medusa 状态：78.40 % acc）。
- **`prepare_for_verify` 不会改写 `batch.seq_lens`**（只改 `batch.input_ids`、`batch.out_cache_loc`，以及 `req_to_token[idx, seq_lens:seq_lens+ndt]`）。`seq_lens` 每步只在 `.verify()` 里增加一次：`batch.seq_lens.add_(self.accept_length + 1)`。→ **假设 3 排除**。
- **kernel 约定**（来自单测）：`positions[0] = seq_lens`（不是 `seq_lens-1`）。verify 树的 root 放在下一个空闲槽位，而不是上一个已提交 token 的槽位。→ **假设 1 的原始形式排除**（我们手写的 `positions = seq_lens + arange(ndt)` 在数值上与 kernel 约定一致）。
- **假设 2（hidden 分布不匹配）** 真实存在，但单独不足以解释 `mcq=0%` 且 avg_out=64460。差的 draft 只会让 `accept_rate` 下降，不应该破坏 **bonus** token（root 位置的 target_predict，本质是模型自身的贪心输出）。→ **假设 2 排除为主因**。

### 真正的根因：一个早就存在的 positional 不变量违例，在 ndt=1 下被掩盖，到 ndt=2 时致命爆发

sglang 的 spec-decode 不变量：

> KV 槽位 `k` 存的是 `origin_input_ids ++ output_ids` 中 **概念位置 `k`** 处的 token（id + positional embedding `k`）的 KV。

NGRAM 约定下保证了这个不变量：

- 第 N 步的 `input_ids[0]` 是 **来自 n-gram 缓存的第一个推测 token**（针对位置 `seq_lens` 的新预测），**不是**前一步提交的 bonus 重喂。
- 上一步的 bonus 的 KV 已经在第 N−1 步当它作为 `input_ids[0]` 喂入时计算了 —— 位置是正确的 `seq_lens_{N-1} = seq_lens_N - (accept_length_{N-1} + 1)`。链条自然延伸，槽位 `k` 始终对应概念位置 `k` 的 token。

而 Medusa 现在的实现（Stage 3a 和被回退的 ndt=2 尝试）**违反了这个不变量**：

- Stage 3a（`ndt=1`）把 `input_ids[0] = output_ids[-1]`（最后已提交的 token，它的概念位置是 `seq_lens - 1`）喂在位置 `seq_lens`。
- 模型在槽位 `seq_lens` 算出的 KV，positional embedding 是 `seq_lens`，但 token id 是位于概念位置 `seq_lens - 1` 的那个 token。差一位。
- 提交的 `target_predict[0]` 是模型在 *这个差一位输入* 条件下的预测，被记到 `output_ids` 中作为概念位置 `seq_lens` 的 token。下一步又重复差一位。错位每步都在，但模型够鲁棒，仅造成约 0.58 pt 的下降（Stage 3a：78.71 % vs v22 baseline 79.29 %）。

ndt=2 路径下，同样的差一位作用在 **两个** 位置上：

- `input_ids = [bonus = output_ids[-1], draft = head_pred]`，`positions = [seq_lens, seq_lens+1]`。
- bonus 还是差一位（概念位置 `seq_lens-1` 的 token 被喂到位置 `seq_lens`）。
- `target_predict[0]` 是在 *差一位 bonus 输入* 条件下的预测 —— 分布与真正的「committed history 之后的下一个 token」明显不同。
- `target_predict[1]` 是在（差一位 bonus）+（head draft）条件下的预测，错得更狠。
- `verify_tree_greedy` 在 `target_predict[0] == head_draft` 时接受 draft（这就是 `accept_rate=0.73` 日志测的东西 —— 但是相对 *被污染的* `target_predict[0]`，不是真实 argmax）。
- 下一步要用的 bonus（接受时 `predicts[1] = target_predict[1]`，拒绝时 `predicts[0] = target_predict[0]`）也是从被污染的分布里采的。
- 每步的偏差累积；在 MCQ 上模型永远到不了 stop token，把 `max_out_len = 65536` 跑满。

所以 `accept_rate = 0.73` 这条日志是 **误导** 的：它只证明 tree-walk kernel 结构上是对的，**不** 证明提交的 token 是对的。

### 为什么之前没抓到

- Stage 2 passthrough（v23）在 forward 之前把 `spec_algorithm = NONE` 翻掉，跑出来与 v22 字节等价，根本没构造 spec_info，所以没有 off-by-one（80.11 % acc，S1=118.28 s —— 反证 baseline 干净）。
- Stage 3a ndt=1 有 off-by-one，但只损失约 0.6 pt：(a) 每步只一个 token 受影响，(b) 模型对位置扰动在域内 prefix 不敏感，(c) `verify_tree_greedy` 在 ndt=1 下不论 accept 逻辑都直接提交 `target_predict[0]`，所以结果在结构上等价于（轻微扰动的）dense decode。
- ndt=2 通过 `target_predict[1]` 的乘性放大 + bonus 的递推链，把误差放大到致命级别。

### 后续：修正方案

下一步在 `PROPOSAL_medusa_k1_positional_offbyone_fix.{en,zh}.md` 提出。要点：

- 在 `prepare_for_verify` **之前** 把 `batch.seq_lens` 减 1，让槽位分配从 bonus 的正确概念位置（`seq_lens - 1`）开始；`verify()` 末尾的 `seq_lens.add_(accept_length + 1)` 会把 `seq_lens` 自然带回正确的步后值。
- 改用 canonical `reconstruct_indices_from_tree_mask` kernel（和 NGRAM 完全一致）来反推 `positions / retrive_*`，不要再手写 —— 减小 off-by-one 出错面。
- 任何完整 eval 之前先在 10 条请求小集上端到端验证。

