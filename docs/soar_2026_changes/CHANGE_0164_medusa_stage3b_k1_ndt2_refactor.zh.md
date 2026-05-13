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
