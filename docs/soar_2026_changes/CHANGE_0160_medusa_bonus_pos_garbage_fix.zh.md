# CHANGE_0160 — MEDUSA：Verify 后将奖励 token 的 req_to_token 位置置零

**文件：** `python/sglang/srt/speculative/medusa_worker.py`  
**依赖：** CHANGE_0158（稀疏 KV 零过滤），CHANGE_0159（主 KV 零过滤）  
**提交：** 待定

---

## 背景与动机

CHANGE_0158 和 CHANGE_0159 应用后，服务器仍以
`available_size = max + 14` 崩溃——KV 池分配器计数器出现小幅正超额。
该计数器不应超过 `max`；任何正超额均表示某个槽在从未分配的情况下被
释放（双重释放或幽灵释放）。

### 根因追溯

MEDUSA K=1 路径（`_forward_verify_k1`）中，当 `accept_length = 1` 时：

1. `prepare_for_verify` 为每个请求分配 **1 个 page-slot** 到
   `batch.out_cache_loc`（大小 = `bs × draft_token_num = bs × 1`）。

2. verify 前向完成后，`_free_cache`（位于 `spec_info.verify()` 内部）调用：
   ```python
   assign_req_to_token_pool(
       ...,
       start=seq_lens_old,
       end=seq_lens_old + accept_length + 1,   # = seq_lens_old + 2
       out_cache_loc=tgt_cache_loc,             # 每请求仅有 1 个元素！
   )
   ```
   Triton 内核读取 `out_cache_loc[0]`（草稿槽，有效）以及
   `out_cache_loc[1]`（**越界**）。该 GPU 内存地址处的值——通常非零——
   被写入 `req_to_token[req_pool_idx, seq_lens_old + accept_length]`
   （即**奖励 token 位置**）。

3. 请求完成时，`cache_finished_req` 读取
   `req_to_token[:kv_committed_len]` 并调用
   `token_to_kv_pool_allocator.free(...)`。
   CHANGE_0159 过滤零值（`kv_indices.ne(0)`），但奖励位置持有
   **非零垃圾值**——通过过滤并被释放。由于该槽从未被分配，
   `available_size` 每次事件增加 1。

4. 当垃圾值恰好是相同的非零索引（如 14）时，经过几个请求后
   累积超额达 +14，触发断言。

---

## 规则合规声明

这是一个 bug 修复，而非新功能。每次 MEDUSA verify 步骤增加一个
O(bs) 的 Python 级别循环。无内核变更，无精度影响，无提交包大小影响。
完全符合 SOAR 规范。

---

## 实现方案

**位置：** `python/sglang/srt/speculative/medusa_worker.py`，
`_forward_verify_k1()`，紧接 `spec_info.verify()` 返回之后。

**逻辑：**
```python
accept_lens_cpu = spec_info.accept_length.cpu().tolist()
for _i, _req in enumerate(batch.reqs):
    if accept_lens_cpu[_i] >= 1:
        _bonus_pos = int(batch.seq_lens[_i].item()) - 1
        batch.req_to_token_pool.req_to_token[_req.req_pool_idx, _bonus_pos] = 0
```

`spec_info.verify()` 返回后：
- `batch.seq_lens[i]` = `seq_lens_old[i] + accept_length[i] + 1`
- 奖励位置 = `seq_lens_old[i] + accept_length[i]`
            = `batch.seq_lens[i] - 1`

将该位置置零后，CHANGE_0159 的过滤器（`kv_indices.ne(0)`）在清理时
可正确排除未分配的奖励槽。

奖励 token 的 KV 从未被计算（前向传播仅处理草稿 token），因此置零
在语义上是正确的。

---

## 实际代码变更

### `python/sglang/srt/speculative/medusa_worker.py`

```diff
         logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(
             batch, logits_output, self.page_size
         )
-        accept_lens = spec_info.accept_length  # (bs,) tensor, always 0 for K=1
+        accept_lens = spec_info.accept_length  # (bs,) tensor
+
+        # CHANGE_0160: Verify 后将奖励 token 位置置零（详见代码注释）
+        accept_lens_cpu = spec_info.accept_length.cpu().tolist()
+        for _i, _req in enumerate(batch.reqs):
+            if accept_lens_cpu[_i] >= 1:
+                _bonus_pos = int(batch.seq_lens[_i].item()) - 1
+                batch.req_to_token_pool.req_to_token[_req.req_pool_idx, _bonus_pos] = 0

         # 7. 恢复 forward_mode / spec_algorithm 用于调度器记账。
```

---

## 验证命令

```bash
# 在 fcloud 上——使用 CHANGE_0160 重启服务器：
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 运行速度测试（服务器必须在全部 48 个 S1 请求中不崩溃）：
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# 运行精度评估：
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

**成功标准：**
- 任何速度测试期间服务器不崩溃
- 服务器日志中 `available_size` 从不超过 `max`
- 精度 ≥ 75%（理想情况与 CHANGE_0157 基准 76.04% 相同）

---

## 结果汇总表

| 指标 | 修复前（仅 CHANGE_0158+0159） | CHANGE_0160 后 |
|------|-------------------------------|----------------|
| 服务器稳定性 | ~5 个请求后崩溃（available_size=max+14） | 待定 |
| S1 耗时（秒） | N/A（崩溃） | 待定 |
| S8 耗时（秒） | N/A（崩溃） | 待定 |
| Smax 耗时（秒） | N/A（崩溃） | 待定 |
| 精度 | N/A（崩溃） | 待定 |

---

## 回滚说明

```bash
git revert HEAD   # 或手动删除 medusa_worker.py 中的 CHANGE_0160 代码块
git push minicpm-src mixed_minicpm_cudagraph
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
```

---

## 后续建议

1. **奖励槽的正确分配（长期修复）：** 修改 `prepare_for_verify` 分配
   `draft_token_num + 1` 个 page-slot，使奖励 token 获得有效 KV 槽。
   这从根本上消除越界读取，并可能提升精度（奖励 token KV 在下一步
   重用页面时被计算）。

2. **Stage 3b（K>1）：** K=1 稳定后，扩展至 K=2+ Medusa 头。相同的
   奖励置零逻辑将适用于每个被接受的草稿路径。

3. **速度测试：** 将 S1/S8/Smax 与 Stage 2 cgraph 基准
   （S1=118.28s，S8=43.87s，Smax=35.75s）比较，量化 MEDUSA 开销。
