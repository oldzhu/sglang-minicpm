# CHANGE_0135 — Sparse 路径清理（移除 `--dense-as-sparse`）与 Option-B 性能剖析计划

状态：APPLIED（prepare_env.sh 修改）+ PROPOSAL（剖析计划，等待 fcloud）
分支：`mixed_minicpm_cudagraph`
关联：CHANGE_0133（compress 缓冲区超填）、CHANGE_0134（eval model_path / sparse 激活规则）

## 背景与动机

Round 13e Test 1（BF16 + 原生 sparse + FP8 KV，并发=32）在 85/150 处中止，
出现多次 3000s 读超时。CHANGE_0134 已排除 GPTQ-vs-BF16 的 tokenizer
差异为根因。剩下的假设是长上下文（32K-128K）下 sparse-attn 自身的耗时。

在投入深度剖析前，我们先识别并清理两个配置层面的问题：

### 问题 1 — 我们的配置下 `--dense-as-sparse` 是有害的

MiniCPM 自定义 backend 内部有一个按请求的 seq-len 阈值
`dense_len = hf_config.sparse_dense_len`（默认 `512`）。见
[`minicpm_sparse_utils.py:1023`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L1023)：

```python
if forward_batch.seq_lens_cpu[i] >= self.config.dense_len or dense_as_sparse:
    # sparse 分支（top-k 评分 + sparse FlashAttention）
else:
    # 同 backend 内的 dense 分支（FA 全注意力）
```

`--dense-as-sparse` 强制 `dense_len = 0`，把**所有**请求都送进 sparse
分支，包括那些原本会走更便宜的 dense FA 分支的短请求。

- 在 `--attention-backend flashinfer`（官方 toolkit 默认）下：
  `--dense-as-sparse` 是死代码——根本不会加载自定义 MiniCPM backend，
  sparse 代码永远不会执行。删除以保持配置干净。
- 在 `--attention-backend minicpm_flashinfer`（我们 Round 13e 配置）下：
  `--dense-as-sparse` **有实际负面影响**。它把短请求强行送进昂贵的
  top-k+sparse-FA，而 dense FA 分支在精度等价的同时更快。

backend 内部的 sparse/dense 混合路由本来就是 MiniCPM4 混合架构的设计
意图——短 prefill / 短 decode 走 dense，长上下文走 sparse。我们应该让
这个路由生效，而不是覆盖它。

### 问题 2 — `dense_len` 只能从 config 读，无 CLI 标志

阈值来源：

- HF config 顶层 `sparse_dense_len`，或
- 嵌套的 `sparse_config.dense_len`，或
- 兜底 `512`

（见 [`python/sglang/srt/configs/minicpm.py:83-95`](../../python/sglang/srt/configs/minicpm.py#L83-L95)
与 [`python/sglang/srt/configs/model_config.py:281-283`](../../python/sglang/srt/configs/model_config.py#L281-L283)）。

目前没有 `--sparse-dense-len` 这类 server arg。要调阈值，今天必须
在预处理阶段改模型 `config.json`。新增一个 CLI 覆盖大约 10 行代码
（`server_args.py` + `minicpm_backend.py:238`），但推迟到剖析数据
显示该阈值对我们工作负载有意义之后再做。

## 规则合规说明

- 不影响精度。移除 `--dense-as-sparse` 只改变*哪个*注意力 kernel 跑；
  两个分支产生数学等价的注意力输出（dense FA = 全注意力；sparse
  分支是模型公开发布的设计，用于 ≥ `dense_len` 的长序列）。
- 不改模型。仅配置层面。
- 不改动 `eval_model*.py`。
- 与提交包兼容——只影响 `prepare_env.sh` 的 noquant 分支，不是当前
  提交基线（GPTQ 走 dense）。

## 实施（已应用）

`benchmark/soar/demo_sala/prepare_env.sh`，`noquant` 分支——移除
`--dense-as-sparse` 并加注释：

```diff
+	# NOTE: --dense-as-sparse intentionally removed (Round 13e analysis):
+	#   - Under flashinfer backend it's a no-op (custom MiniCPM backend not loaded).
+	#   - Under minicpm_flashinfer it forces requests with seq_len < hf_config.sparse_dense_len
+	#     (default 512) through the expensive sparse top-k+sparse-FA path, which is
+	#     slower than the dense FA branch they would otherwise take. Letting the
+	#     model-config dense_len threshold route short requests to dense and long
+	#     requests to sparse matches the mixed architecture's design intent.
-	export SGLANG_SERVER_ARGS="... --dense-as-sparse --kv-cache-dtype fp8_e5m2 ..."
+	export SGLANG_SERVER_ARGS="... --kv-cache-dtype fp8_e5m2 ..."
```

## Option-B 剖析计划（提案——上 fcloud 前需用户批准）

目标：在投入优化时间前，先在代表性 bs × seq_len 下找出 sparse decode
路径上真正的热点 kernel。

### 剖析配置

- 量化：`--quant-mode noquant`（BF16）——与 Round 13e 一致以便直接对照。
- Server args：清理后的配置（不带 `--dense-as-sparse`）。
- 并发：**单请求**（避免 `--max-concurrent` 竞争 / KV 压力的噪声）。
- 工作负载：1 个 ~64K 输入 + ~256 解码 token 的 prompt，然后 1 个
  ~128K 输入 + ~256 解码 token 的 prompt。两个都应走 sparse 分支
  （≫ 512 的 dense_len 阈值）。
- Profiler：torch profiler（`with profile(...)`）放在**一个独立的
  小脚本**里，通过 sglang HTTP API 调用。**不修改** eval harness。
  warmup 后采集 32 个 decode step。把 JSON trace + Chrome trace 存到
  fcloud `/root/profile_round13e/`。

### 我们想看什么

1. 按总耗时排前 5 的 GPU kernel（区分 top-k 评分 vs sparse FA vs
   metadata builders vs lightning-attn vs GEMM）。
2. CPU 侧 metadata 构建开销（[`minicpm_backend.py:467`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L467)
   每步会调用 `build_sparse_decode_metadata`）。
3. cudagraph 是否覆盖了 sparse 分支（CHANGE_0070 表明 KV indptr 已
   覆盖；CHANGE_0133 修了 compress 缓冲区超填——在移除
   `--dense-as-sparse` 之后两者是否仍然成立需要验证）。
4. decode 阶段的内存带宽利用率（目标：sparse FA 至少达到 1398 GB/s
   的 50%，才能算"已经调到位"）。

### 基于剖析结果的决策树

- **如果某 1 个 kernel 占比 >40% 且是已知 op（sparse FA / top-k）**：
  → 在 CHANGE_0136 中提出针对性的 kernel 优化（很可能是 Triton 重写
  或手写 CUDA）。
- **如果耗时分散在 CPU metadata + 大量小 launch**：
  → 在 CHANGE_0136 中提出扩大 graph 捕获范围 / 缓存 metadata 的方案。
- **如果 sparse FA 已经在 >70% 带宽下 memory-bound**：
  → 正式关闭 sparse 路径；该硬件上这个 sparse 架构没有更多余地。
  从 `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` 恢复 dense 路径优化。

### 范围红线

- 单次剖析会话（含 server 启动约 30 分钟 fcloud 时间）。
- 一份剖析结果文档（CHANGE_0135_001 或 RESEARCH_）。
- 剖析会话期间不做任何源码改动。
- 所有 kernel 优化的代码改动都进单独的 CHANGE_0136+ 提案，并需用户
  显式批准。

## 验证

prepare_env.sh 改动本身不需要单独 fcloud 验证（只是去掉一个 arg）。
验证将随 Option-B 剖析运行一并完成。

## 回滚

revert 本次 commit。`--dense-as-sparse` 会回到 noquant 分支。

## 下一步

1. 用户批准 Option-B 剖析计划。
2. 用户启动 fcloud。
3. Agent 运行 64K / 128K 单请求剖析，采集 torch profiler trace。
4. Agent 关闭 fcloud，离线分析 trace，写 CHANGE_0135_001（剖析结果
   + 决策树结论）。
5. Agent 提出 CHANGE_0136（kernel 优化）或正式关闭 sparse 路径。
