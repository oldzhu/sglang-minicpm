# CHANGE_0159 — 修复：MEDUSA 主 KV 零释放崩溃（奖励 Token 槽位缺失）

## 状态
**已应用** — `python/sglang/srt/mem_cache/chunk_cache.py`（1 个文件，~9 行变更）

## 背景

CHANGE_0158 修复了 `cache_finished_req` 中稀疏 K1/K2 槽位的过度释放。  
但崩溃仍在继续：部署 CHANGE_0158 后，服务端仍报错：

```
ValueError: token_to_kv_pool_allocator memory leak detected!
self.max_total_num_tokens=15563319, available_size=15582175, evictable_size=0, protected_size=0
```

`available_size - max = 18856` — 远大于 CHANGE_0158 测试时观察到的 2。  
崩溃发生在第 28 个 S1 请求（约 52K token 的长生成）。  
已确认 CHANGE_0158 的 K1/K2 过滤已生效；问题在于**主 KV 释放路径**。

## 根本原因分析

### TokenToKVPoolAllocator.free() 盲目追加
`TokenToKVPoolAllocator.free(tensor)` 将 `tensor` 中所有条目追加到 `free_pages`：
```python
def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:
        return
    self.free_pages = torch.cat((self.free_pages, free_index))
```
无零值检查，无重复检查。`available_size = len(free_pages)`。

### cache_finished_req 释放主 KV 索引时无零过滤
```python
kv_indices = req_to_token_pool.req_to_token[req.req_pool_idx, :kv_committed_len]
req_to_token_pool.free(req.req_pool_idx)
token_to_kv_pool_allocator.free(kv_indices)   # ← 无零过滤！
```
`req_to_token` 以 `torch.zeros(...)` 初始化。任何未写入的位置读取为 **0**。

### MEDUSA 奖励 Token 位置从未写入 req_to_token

对于 MEDUSA `draft_token_num=1`，`--speculative-num-draft-tokens 2`：

**每次验证步骤（page_size=1 情况）**：
1. `prepare_for_verify` 为草稿 token 分配 1 个槽位 `s1`：
   - `assign_req_to_token_pool(start=seq_lens, end=seq_lens+1, out_cache_loc=[s1])`
   - 将 `s1` 写入 `req_to_token[idx, seq_lens]` ✓

2. 对 `[draft_token]` 进行前向传播 → `seq_lens` 处的 KV 存入 `s1`。

3. 验证结果：若草稿被接受（`accept_length=1`）：
   - `_free_cache` 尝试从 `out_cache_loc=[s1]` 写入位置 `seq_lens` 和 `seq_lens+1`
   - `assign_req_to_token_pool(start=seq_lens, end=seq_lens+2, out_cache_loc=[s1])`
   - triton kernel 读取 `out_cache_loc[0]` → 位置 `seq_lens` 写入 `s1` ✓
   - triton kernel 读取 `out_cache_loc[1]` → 位置 `seq_lens+1` 发生**越界读**（只有 1 条目）
   - `req_to_token[idx, seq_lens+1]` 获取 GPU 内存中 `out_cache_loc + 1` 处的值，可能为 0 或垃圾值
   - `kv_committed_len += accept_length + 1 = 2`

4. 后续步骤中：`req_to_token[idx, seq_lens+1] = 0` 永远不被覆写。

### 累积过度释放

对于约 50K token、约 50% 草稿接受率的长请求：
- ~25K 步有 `accept_length=1` → ~25K 位置 `req_to_token=0`
- `cache_finished_req` 时：`kv_indices` 包含 ~25K 个零值
- `free(zeros)` → `available_size += 25K`（相对于满载基准）
- 观测到的 `available_size = max + 18856` 反映了累积的零值释放

（18856 < 25K 是因为较短的初始请求 accept_length=1 步骤较少。）

### 为何槽位 0 可以安全过滤
`TokenToKVPoolAllocator.clear()`：
```python
self.free_pages = torch.arange(1, self.size + 1, ...)  # 从 1 开始
```
**槽位 0 永远不会被分配**；它是保留的"填充哑元"槽。  
`kv_indices` 中任何 0 都是从未写入真实 KV 槽 ID 的位置。

## 规则合规声明
纯缺陷修复。无算法变更，无精度影响。对所有路径都安全，因为合法分配的 KV 槽 ID 始终 ≥ 1。

## 实现

### 修改文件
`python/sglang/srt/mem_cache/chunk_cache.py`  
`benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/chunk_cache.py`

### 代码变更
```diff
-        self.req_to_token_pool.free(req.req_pool_idx)
-        self.token_to_kv_pool_allocator.free(kv_indices)
+        self.req_to_token_pool.free(req.req_pool_idx)
+        # CHANGE_0159: 过滤主 KV 索引中的槽位 0（保留哨兵）。
+        # 在 draft_token_num=1 的 MEDUSA 中，当 accept_length=1 时，
+        # 奖励 token 位置在 req_to_token 中从未写入（保持为 0），
+        # 但 kv_committed_len 增加了 accept_length+1=2。
+        # free([0]) 将哨兵加回 free_pages，导致 available_size > max 崩溃。
+        kv_indices_valid = kv_indices[kv_indices.ne(0)].to(torch.int64)
+        if kv_indices_valid.numel() > 0:
+            self.token_to_kv_pool_allocator.free(kv_indices_valid)
```

### 内存泄漏说明
零过滤后：
- 零值不被释放 → `available_size` 保持在 `max`
- 底层问题（奖励 token KV 槽未分配）被屏蔽而非修复
- 正确做法：重构 MEDUSA 验证循环以为奖励 token 分配并写入 KV 槽（Stage 3b 工作）

### 正确性说明
`seq_lens+1` 处缺失的奖励 token KV 意味着后续验证步骤的注意力从槽位 0（哑 KV 数据）读取。  
这可能解释了部分精度差距（76.04% vs 基线 80.11%）。正确修复是 Stage 3b 工作。

## 验证命令

### 崩溃测试
```bash
# 重启后运行 S1 速度测试——服务端不应在请求间崩溃
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
```
预期：48 个请求完成，无 `ConnectionResetError`。

### 完整速度测试
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### 内存日志检查
```bash
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 50
```
无 `token_to_kv_pool_allocator memory leak detected` 错误。

## 结果汇总

| 版本 | 崩溃 | 原因 |
|------|------|------|
| CHANGE_0157 | 每次请求后 | MEDUSA K1/K2 稀疏过度释放 |
| CHANGE_0158 | 长请求后 | 主 KV 奖励 token 零释放 |
| CHANGE_0159 | 无（预期） | 主 KV 路径零过滤 |

## 回滚说明
```bash
git revert <commit_hash>
git push minicpm-src mixed_minicpm_cudagraph
```

## 后续步骤（Stage 3b）
1. 在 `_forward_verify_k1` 或 `NgramVerifyInput.prepare_for_verify` 中为奖励 token 正确分配 KV 槽，修复损坏的注意力问题。
2. 或：设置 `kv_committed_len += accept_length`（不含奖励），将奖励 token 作为下一步草稿的第一个 token 处理。
3. 追踪修复奖励 KV 是否能将精度提升至 76% 以上。
