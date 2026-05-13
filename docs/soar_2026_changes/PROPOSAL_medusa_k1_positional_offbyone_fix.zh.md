# PROPOSAL — Medusa K=1 位置 off-by-one 修复（ndt=2 修正设计）

日期：2026-05-13
状态：**PROPOSAL — 待评审，未改代码**
目标分支：`mixed_minicpm_cudagraph`
前置：[CHANGE_0164 回退后根因分析](CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md#回退后根因分析2026-05-13离线)
参考：
- [python/sglang/srt/speculative/ngram_info.py](../../python/sglang/srt/speculative/ngram_info.py)
- [python/sglang/srt/speculative/ngram_worker.py](../../python/sglang/srt/speculative/ngram_worker.py)
- [sgl-kernel/tests/speculative/test_ngram_utils.py](../../sgl-kernel/tests/speculative/test_ngram_utils.py)
- [python/sglang/srt/speculative/medusa_worker.py](../../python/sglang/srt/speculative/medusa_worker.py)（当前 Stage 3a）

## 目标

恢复 `MedusaWorker._forward_verify_k1` 的 spec-decode 位置不变量，使：

1. **Stage 3a（ndt=1，无 head）** 与 dense decode 字节等价，找回 v23 那 ~0.6 pt 回退（78.71 % → 79.29 %）。
2. **Stage 3b（ndt=2，trained head）** 不再有 CHANGE_0164 的灾难性漂移（15.13 % acc）。当 head 的预测与模型真正下一个 token 一致时每步提交 2 个 token（S1 加速）；不一致时回退到每步 1 个正确 token（无回退）。

## 背景

Stage 3a 和被回退的 Stage 3b 共同违反的不变量：

> KV 槽位 `k` 存的是 `origin_input_ids ++ output_ids` 中 **概念位置 `k`** 处 token（id + positional embedding `k`）的 KV。

现有代码把 `output_ids[-1]`（概念位置 `seq_lens - 1` 处的 token）喂在位置 `seq_lens`。模型在槽位 `seq_lens` 写 KV，positional embedding 是 `seq_lens`，但 token-id 是错位 token 的。Medusa 每步 decode 都差一位。

NGRAM 没这个 bug，因为它的 `input_ids[0]` 是 **n-gram 缓存给出的新预测**（针对槽位 `seq_lens`），不是上一步 bonus 的重喂。

Medusa 无法不重喂 bonus（LM head 的预测是 **唯一** 能给出 "槽位 seq_lens 的有效预测" 的来源；trained head 给的是槽位 `seq_lens + 1` 的草稿，不是 `seq_lens` 的）。修法是把槽位分配到 **正确** 位置上。

## 规则合规

- 不动官方评测脚本和评分路径。
- 不动模型权重、量化、KV-cache dtype。
- 不动 baseline `--force-dense-minicpm` + FP8 KV + Tier1 长上下文 server args。
- Server-arg 表面无变化。
- 只改 `MedusaWorker._forward_verify_k1`；`NgramVerifyInput` 和 sgl-kernel verify 路径原样复用（且与 upstream 一致）。

## 设计方案

### 第一步 — 先修 ndt=1 的位置不变量（Stage 3a；**独立、低风险**）

在 `_forward_verify_k1` 中，在构造 `NgramVerifyInput` **之前**：

```python
# 把 seq_lens 减 1，让槽位分配从 bonus 的真实概念位置开始。
# NgramVerifyInput.verify() 结尾的 `batch.seq_lens.add_(accept_length + 1)`
# 会自然把 seq_lens 带回正确的步后值（ndt=1 时 accept_length=0，+1 正好
# 对应那 1 个新提交 token）。
batch.seq_lens = batch.seq_lens - 1
batch.seq_lens_cpu = batch.seq_lens_cpu - 1
```

然后 `positions = batch.seq_lens.clone()`（现在等于 `seq_lens_original - 1`，即 `output_ids[-1]` 的正确概念位置）。

**`prepare_for_verify` 内部的连带影响**：它会写 `req_to_token[idx, seq_lens : seq_lens + ndt]`。减 1 后这个区间是 `[seq_lens_original - 1, seq_lens_original)`，正好覆盖 bonus token 应该所在的槽位。这是正确的（且实际上 **修复** 了此前几步累积下来的不变量违例）。

**关键风险点**：`prepare_for_verify` 调 `alloc_token_slots(... len(batch.input_ids))`（page_size=1）从页池分一批新槽位，与 `seq_lens` 无关。verify 结尾 `_free_cache` 按 `accept_length` 释放未接受的槽位。需要确认把 `seq_lens_original - 1` 槽位覆盖写不会造成重复释放或页池脏数据。**这是主要风险，实现前必须先读 `get_src_tgt_cache_loc` / `_free_cache`**。

如果有冲突，采用 §"备用设计" 的方案。

### 第二步 — Stage 3b 扩展到 ndt=2（trained head）

第一步落地、Stage 3a acc 验到 ~79.3 % 之后：

```python
self.draft_token_num = self.num_heads + 1  # K=1 时 = 2

# 每个请求 input_ids = [bonus = output_ids[-1], draft = head_pred_or_fallback]
# 每个请求 positions = [seq_lens_original - 1, seq_lens_original]
#   （即第一步那个减 1 之后的 seq_lens）
```

`positions`、`retrive_index`、`retrive_next_token`、`retrive_next_sibling` **改为调用 canonical kernel** 反推，不再手写：

```python
# 紧凑 (bs, ndt, ndt) tree mask：K=1 线性链的下三角。
# bs=1 ndt=2 时是 [[1,0],[1,1]]（拍平到长度 bs*ndt*ndt）。
compact_mask = torch.tensor(
    [[1, 0, 1, 1]] * bs, dtype=torch.bool, device=self.device
).reshape(bs * self.draft_token_num * self.draft_token_num)

reconstruct_indices_from_tree_mask(
    compact_mask,
    batch.seq_lens,           # 第一步已经减 1
    positions,                # 输出，shape (bs*ndt,)
    retrive_index,            # 输出，shape (bs, ndt)
    retrive_next_token,       # 输出，shape (bs, ndt)
    retrive_next_sibling,     # 输出，shape (bs, ndt)
    bs,
    self.draft_token_num,
)
```

与 `ngram_worker._prepare_for_speculative_decoding` 逐行对齐，避免再引入隐蔽的 off-by-one。

`USE_FULL_MASK=True`（flashinfer 需要）时按 NGRAM 同样的方式把 `compact_mask` 扩展为每请求 `(ndt, seq_len_i - 1 + ndt)` 形状：`req_mask = torch.cat([ones(ndt, seq_len-1), compact[i].view(ndt, ndt)], dim=1)`。

`CaptureHiddenMode.FULL`、head forward 用 `hidden[:, 0, :]`（bonus 位置）、`req._medusa_draft_token` 缓存草稿 — 与被回退的 CHANGE_0164 一致。

### 备用设计（若 `_free_cache` 不容忍 seq_lens 减 1）

若读 `_free_cache` / `get_src_tgt_cache_loc` 发现覆盖已分配槽位会双释放或污染页池，则改为 **不重喂 bonus**：

- ndt = `num_heads`（K=1 时 = 1，K=2 时 = 2，依此类推）。
- `input_ids[0..ndt-1]` = head 对槽位 `seq_lens .. seq_lens + ndt - 1` 的预测。
- K=1 时只有 1 个 head 预测。没有 "root 总接受" 的小把戏 — 每个提交 token 必须真满足 verify 接受规则。
- 这需要重新定义 head 训练目标（原：从当前 hidden 预测 bonus；新：从上一步 hidden 预测 bonus + draft）。代价较高，除非第一步走不通才走这条。

## 详细实施计划（评审 checklist）

1. **读 `spec_utils.get_src_tgt_cache_loc` 与 `NgramVerifyInput._free_cache`**，确认 seq_lens 减 1 安全。
2. **写一段 30 行的独立 Python 脚本**：单 prompt eager 跑 MedusaWorker 一步，dump `(positions, seq_lens_in, seq_lens_out, req_to_token[idx, seq_lens-2:seq_lens+ndt+1], input_ids, predicts, accept_length)` 到 JSON。
3. **同样的脚本跑一次单步 NgramWorker**（强制 `ngram_cache.batch_get` 返回 `[output_ids[-1], output_ids[-1]]`），逐字节比对 `req_to_token` 更新和 `predicts[0]`。
4. **实现第一步**（ndt=1 + seq_lens 减 1）。本地预飞：server 启动正常、10 条短 prompt 无 crash、10 条子集 acc ≥ 78 %。
5. **fcloud 跑第一步完整 accuracy**。
6. **实现第二步**（ndt=2 + `reconstruct_indices_from_tree_mask`）。同样预飞门控。
7. **fcloud 跑第二步完整 accuracy + S1/S8/Smax**。

## 验证标准

| 测试 | 通过条件 |
|---|---|
| 第一步（Stage 3a，ndt=1，修好位置，无 head） | ori_accuracy ≥ 79.0 %（找回 v23 那 0.6 pt）；S1 ≥ baseline（无 spec、无加速预期） |
| 第二步（Stage 3b，ndt=2，修好位置，head v2） | ori_accuracy ≥ 79.0 %；accept_len ≥ 1.5；S1 ≤ 140 s |
| 全部 | 公开 90-set 上 mcq avg_out_len < 4096（无 runaway） |

## 回退

每步独立提交，单条 `git revert` 即可。Stage 3a（commit `3a15a6de3`）是 fallback 底线。

## 风险

- **页池一致性**：seq_lens 减 1 是最大风险。靠第 2 步的源码阅读 + 独立脚本验证缓解。
- **`reconstruct_indices_from_tree_mask` 对 `draft_token_num` 的隐含约束**：现有单测只覆盖 ndt=4。在我们的 build 上先跑这个单测。
- **head v2 分布不匹配**：与本修复 **独立**。若第一步过、第二步 accept_rate 低，说明位置 bug 已修，head 需要继续训。这就是想要的二分结论。

## 下一步建议

- 第一步实施计划里的本地单步验证脚本通过之前，不启 fcloud。
- 这个提案中的 **第一步独立有价值**：哪怕 Step 2 无限期搁置，先把 v23 的 0.6 pt 回退找回来也值得。
