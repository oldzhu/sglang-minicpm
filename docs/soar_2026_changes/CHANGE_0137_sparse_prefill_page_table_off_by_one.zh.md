# CHANGE_0137 — 稀疏 prefill 中 `sparse_page_table` 切片的 off-by-one(发现 + 修复方案)

**状态**: 在验证 CHANGE_0136 sanity 时发现;修复方案尚未应用。
**发现时间**: 2026-04-29(`SOAR_SPARSE_DENSE_LEN=524288` 第三次启动)。
**复现提交**: `7a7a568eb`(`minicpm-src/mixed_minicpm_cudagraph`)。

## 现象

第一条 eval prefill 请求(103 tokens,单序列,无前缀)在
`python/sglang/srt/layers/attention/minicpm_backend.py:1087` 崩溃:

```python
metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = \
    page_table[dense_bs, : kv_len] * 2
```

```
RuntimeError: The expanded size of the tensor (103) must match the existing
size (104) at non-singleton dimension 0.  Target sizes: [103].  Tensor sizes: [104]
```

完整堆栈终止于 `MiniCPMSparseBackend.forward_extend` 中
`if forward_batch.sparse_batch_size < bs:` 分支(即至少一条请求走稀疏 backend 内部
**dense fallback** 路径时)。

## 根因

分配器与写入端使用了两个**不同**的长度量:

| 位置 | 使用值 | 文件 / 行 |
|------|--------|-----------|
| 分配器 (确定 `sparse_page_table.shape[1]`) | `forward_batch.extend_seq_lens_cpu[i]`(else 分支) | `minicpm_sparse_utils.py:1421` |
| 写入端 (切片 `:kv_len`) | `forward_batch.seq_lens_cpu[dense_bs]` | `minicpm_backend.py:1086` |

在 **overlap-mode 调度** (`event_loop_overlap`,`disable_overlap_schedule=False`
的默认值) 下,`seq_lens_cpu[i]` 反映的是 **已被预先 +1** 的下一步序列长度
(为下一个 decode token 提前更新),而 prefill batch 的 metadata 尚未重建。
因此对一个本次 chunk 贡献 103 tokens 的请求:

```
extend_seq_lens_cpu[i] = 103   (chunk 长度,分配器使用)
seq_lens_cpu[i]        = 104   (前缀 0 + 103,再 + 1 来自 overlap 预更新)
```

`max_sparse_cache_len = max(prev, 103) = 103` → `sparse_page_table.shape[1] = 103`。
但写入端的切片是 `[..., :104]`:LHS 在行宽 103 处被截断,RHS 从
`page_table[dense_bs, :104]` 实际取到 104 个元素。两侧大小不一致 → 广播赋值失败。

## 为什么 CHANGE_0136 之前没暴露

历史上的稀疏路径运行和已发布的提交 baseline 至少满足以下任一条件,使得
line 1087 的 dense-fallback 写入**从未触发**到一个被截断的窄行:

1. **Test 12 / 提交 baseline** — 启用 `--force-dense-minicpm`。
   `server_args` 后处理在 `server_args.py:1525` 把
   `minicpm_flashinfer → flashinfer`,full-attention backend 变为 stock
   `FlashInferAttnBackend`。`MiniCPMSparseBackend` **从未被实例化**,分配器/写入
   两端代码都不会跑。bug 不可见。

2. **`--dense-as-sparse`(多个 baseline 使用)** — 在
   `MiniCPMSparseBackend.__init__` 中强制 `self.dense_len = 0`。
   `minicpm_sparse_utils.py:1404` 处的构建循环对**每条请求**都走
   `seq_lens_cpu[i] >= dense_len` 分支,
   `max_sparse_cache_len = sparse_topk * block_size`(很大),足以吸收 overlap
   带来的 +1。又因 `sparse_batch_size == bs`,`minicpm_backend.py:1057` 的
   dense_bs 分支整段跳过。bug 不可见。

3. **`SOAR_SPARSE_MODE=1`(默认 `dense_len = sparse_dense_len = 8192`)** —
   公共 eval 集 ~150 条样本,prompt 长度从 ~100(mcq) 到 ~128k(cwe/niah)
   不等。许多请求 `seq_lens < 8192`,会走 else 分支,会进入
   `dense_bs_list`,理论上能触发该 bug。实际上 Round 13d 在启动期就
   `current_seed during cudagraph capture` 崩了,根本没进 forward_extend。
   只要 Round 13d 能跑到 eval,就会撞上同一个 bug。

4. **CHANGE_0136 + `SOAR_SPARSE_DENSE_LEN=524288`** — `dense_len` 大于一切
   合理 prompt 长度,**所有**请求都走 else 分支,**所有**请求都进
   `dense_bs_list`(因为 `sparse_batch_size = 0 < bs`)。line 1087 dense_bs
   写入正好成为热路径,`max_sparse_cache_len = max(prev, extend_seq_lens_cpu[i])`
   恰好比写入端的 `seq_lens_cpu[i]` 少 1。第一条样本就撞。

因此 CHANGE_0136 **不是 bug 来源**,只是把一个被旧 server-args 长期掩盖的潜伏
缺陷暴露出来。这与 CHANGE_0133 的发现(decode 阶段 `compress_k1/k2` over-fill)
一脉相承 — HEAD 上的稀疏路径有多个 off-by-one,只有当 dense 路由比例或
prefix-cache 模式改变时才会浮现。

## 建议修复(尚未应用)

让分配器使用与写入端相同的长度:

```diff
--- a/python/sglang/srt/layers/attention/minicpm_sparse_utils.py
@@ build_sparse_prefill_metadata
             else:
-                max_sparse_cache_len = max(
-                    max_sparse_cache_len, forward_batch.extend_seq_lens_cpu[i]
-                )
+                # Writer 使用 forward_batch.seq_lens_cpu[i](overlap 预更新后),
+                # 可能比 extend_seq_lens_cpu[i] 多 1。按更大的值分配,
+                # 防止 minicpm_backend.py:1087 dense_bs page-table 拷贝越界。
+                max_sparse_cache_len = max(
+                    max_sparse_cache_len,
+                    int(forward_batch.seq_lens_cpu[i]),
+                )
```

说明:
- 使用 `seq_lens_cpu`(完整 kv 长度),而非 `extend_seq_lens_cpu`(chunk 长度)。
  写入端复制的是 dense_bs 行的**整个** page table,不只是当前 chunk。
- `int(...)` 保证 reduction 在 Python int 内进行,避免 0-d tensor → Python int
  的偶发开销。
- `seq_lens_cpu[i] >= dense_len` 分支已经按 `sparse_topk * block_size` 分配
  (与请求长度无关),不受影响。

### 备选防御性修复(写入端)

若想再加一道保险,也在写入端 clamp:

```diff
--- a/python/sglang/srt/layers/attention/minicpm_backend.py
@@ forward_extend
-                metadata.sparse_page_table[sparse_page_table_idx_start, : kv_len] = page_table[dense_bs, : kv_len] * 2
-                metadata.sparse_page_table[sparse_page_table_idx_start + 1, : kv_len] = page_table[dense_bs, : kv_len] * 2 + 1
+                copy_len = min(int(kv_len), metadata.sparse_page_table.shape[1])
+                metadata.sparse_page_table[sparse_page_table_idx_start, :copy_len] = page_table[dense_bs, :copy_len] * 2
+                metadata.sparse_page_table[sparse_page_table_idx_start + 1, :copy_len] = page_table[dense_bs, :copy_len] * 2 + 1
```

但这样会向下游消费者(例如
`sparse_cache_seqlens_int32 = (metadata.sparse_page_table != 0).sum(dim=1)`)
**隐藏**长度差,导致计数少 1。**正确做法是分配器侧修复**。

## 验证计划(CHANGE_0137 落地后)

1. 应用分配器侧补丁。
2. 重跑 CHANGE_0136 sanity (`SOAR_SPARSE_DENSE_LEN=524288`):
   - 服务器启动 OK(已验证)。
   - 第一条 prefill batch 必须完成、无 size 错误。
   - S₁ 墙钟应在 Test 12 ±5% 之内(`dense_len=524288` 下没有请求走稀疏路径,
     除分配器开销外应等同纯 dense)。
3. 然后再跑 `=65536` 保守、`=16384` 激进。

## 不在本次修复范围

- CHANGE_0133 描述的 decode 阶段 `compress_k1/k2` over-fill(独立症状)。
- `seq_lens_cpu` vs `extend_seq_lens_cpu` 的差异是 overlap-mode 最近回归还是
  长期存在。(修复不依赖考古;分配器按写入端使用的长度分配即可。)

## 交叉引用

- 发现过程: [chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md](chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md)
- 测试行: TEST_RESULTS_TRACKING.md `R13f-CHANGE_0136-sanity`。
- 相关潜伏 bug: [CHANGE_0133_sparse_compress_buffer_oversize.en.md](CHANGE_0133_sparse_compress_buffer_oversize.en.md)。
- 触发文档: [CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md](CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md)。
