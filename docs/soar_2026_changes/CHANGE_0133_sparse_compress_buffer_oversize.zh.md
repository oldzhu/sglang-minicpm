# CHANGE_0133 — 稀疏 decode `compress_k1/k2` 缓冲区超额填充（提案）

## 状态：提案（待审批）

## 背景

Round 13d / 13e 在 HEAD 上重测 GPTQ + FP8 KV + 原生稀疏 路径，两次都
出现长上下文 decode 灾难性减速：相对 Apr 9 2026 的 commit
`9d3ecd168`（Test 8b），单请求 decode 慢 20–50×。Round 13d
（concurrency=32）在 1 小时超时时仅完成 ~76/150。Round 13e（BF16 +
稀疏 + FP8 KV，无 GPTQ）在 concurrency=32 复现同一现象：服务端 CPU
锁死在 scheduler 线程 100%，decode 吞吐稳定在 ~320 tok/s，多次请求
触发 eval 客户端 3000s 读超时。

用户要求抓取活进程调用栈，确认 scheduler 是死循环还是慢路径。

## 诊断步骤

1. **py-spy / gdb / `cat /proc/PID/stack` 都不可用** — fcloud 容器
   缺少 `CAP_SYS_PTRACE`（`CapEff: 0xa80425fb`，bit 19 未置位）；
   `/proc/sys/kernel/yama/ptrace_scope = 1`，文件系统只读。
2. **sglang scheduler 启动时调用 `faulthandler.enable()`**，见
   [scheduler.py:2907](../../python/sglang/srt/managers/scheduler.py#L2907)。
   这意味着发送 `SIGABRT` 时 Python 会先把每条线程栈打印到 stderr，
   再终止进程 — 即「带回溯的 kill」。
3. 对锁死的 scheduler PID 4023 发送 `kill -ABRT`，server log 抓到
   完整栈再退出。

## 抓到的栈

```
Fatal Python error: Aborted

Current thread 0x00007ff192ec9740 (most recent call first):
  File ".../sglang/python/sglang/srt/layers/attention/minicpm_backend.py",
       line 1761 in init_forward_metadata_replay_cuda_graph
  File ".../sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
       line 1284 in init_forward_metadata_replay_cuda_graph
  File ".../sglang/python/sglang/srt/model_executor/cuda_graph_runner.py",
       line 821 in replay_prepare
  File ".../sglang/python/sglang/srt/model_executor/cuda_graph_runner.py",
       line 847 in replay
  File ".../sglang/python/sglang/srt/model_executor/model_runner.py",
       line 2251 in _forward_raw
  ...
  File ".../sglang/python/sglang/srt/managers/scheduler.py",
       line 1144 in event_loop_overlap
```

scheduler **不是死循环**。它在 cudagraph-replay 准备的关键路径上做的
是真实工作 — 但工作量被严重放大了。

## 根因

[`python/sglang/srt/layers/attention/minicpm_backend.py:1822-1823`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L1822-L1823)，
位于 `init_forward_metadata_replay_cuda_graph` 中、紧接 
`torch.cuda.synchronize()` 之后：

```python
self.decode_cuda_graph_metadata["compress_k1"][
    :forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :
].fill_(float('-inf'))
self.decode_cuda_graph_metadata["compress_k2"][
    :forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :
].fill_(float('-inf'))
```

这两行在 commit
[`45c159187`](https://github.com/oldzhu/sglang-minicpm/commit/45c159187)
（"[Fix] fix -inf to compress_k1, compress_k2 buffer for cudagraph"，
2026-02-10，+2/-0 行）引入。

### 为什么在长上下文上灾难性

- `self.max_context_len = 524 288`（注册到池的静态上限，**不是**当前
  batch 的实际最长序列）；
- `forward_batch.batch_size = 8`（max-running-requests=8）；
- `self.k1_kernel_stride = 16`（来自模型的 sparse_kernel_stride）；
- 每个 decode step k1 上要清零的行数：
  `8 × 524 288 / 16 = 262 144` 行；
- 每行 `[num_compress_heads × head_dim]` bf16 ≈ 4–8 KB；
- **每个 decode step 仅 k1 就要清零 1–2 GB；k2 再来一遍。**
- 清零之上紧跟着 `torch.cuda.synchronize()`（约 L1813），
  完全无法和别的工作重叠 —— 每个 decode step 都付全额。

### 为什么 Test 8b（4-09）没崩

Test 8b（commit `9d3ecd168`）已经有这两行，但它跑的是不一样的评测
（更短上下文、更低并发）。清零成本随 `max_context_len`（固定）和
`batch_size` 线性增长，但相对每步「有用工作」的比例在以下情况会爆炸：

- 实际 `max_seq_len << max_context_len`（清零的多数行后续根本
  不被读到，是 `(max_context_len/max_seq_len)×` 的浪费），且
- 长上下文 decode 时每步有用工作本身受 attention I/O 时间限制，相对
  量级变大。

我们这次的评测（32K–128K prompt，concurrency=32，bs=8 + queue=24），
decode 时实际的 `max_seq_len` 是完整 prompt 长度（32K–128K），所以
fill 的超额倍数是 **128K 时 4×、32K 时 16×**，每个 decode step
都要付。注意：`--chunked-prefill-size 32768` 仅限制 *prefill* 分块大小，
**不**限制 decode 时的 `seq_lens_cpu.max()` —— 到 decode 时所有 prefill
chunk 已写入 KV cache，`seq_lens` 反映的是完整 prompt 长度。

### 与 Round 13d torch.compile 崩溃的关系（独立问题）

Round 13d 在这个 fill 运行之前还撞到另一个错误：
`RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA
graph capture`（`torch._dynamo` 在 cudagraph 捕获期间，稀疏
attn kernel 或其分解调用了 RNG 状态时抛出）。这个崩溃与
**过填无关**：fill 只是 `tensor.fill_(-inf)`，不使用 RNG。
去掉 `--enable-torch-compile`（commit `613ea54e4`）只是绕开了
RNG/cudagraph-捕获 不兼容，对 fill 本身的代价没有任何影响。

后果：**仅修 CHANGE_0133、并不会重新启用稀疏路径上的
torch.compile**。如果以后要在稀疏路径重新启用 torch.compile，
需要单独调查：定位被捕获区间内接触 RNG 的调用（可能是某个
np dropout helper 在 p=0 时仍拼 RNG，或某 kernel 调用
`torch.cuda.get_rng_state()`），要么把它提出捕获区间、要么
换为不使用 Generator 的确定性实现。

## 提议修复（单文件，低风险）

用 L1742 已经算好的 `max_len = seq_lens_cpu.max().item()` 替换
`self.max_context_len`，用 cudagraph 捕获时的 `bs`（L1722 已经是局部）
替换 `forward_batch.batch_size`。

### 改前（HEAD，L1822-L1823）

```python
self.decode_cuda_graph_metadata["compress_k1"][
    :forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :
].fill_(float('-inf'))
self.decode_cuda_graph_metadata["compress_k2"][
    :forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :
].fill_(float('-inf'))
```

### 改后（提议）

```python
fill_rows_k1 = bs * (
    (max_len + self.k1_kernel_stride - 1) // self.k1_kernel_stride
)
fill_rows_k2 = bs * (
    (max_len + self.k2_kernel_stride - 1) // self.k2_kernel_stride
)
self.decode_cuda_graph_metadata["compress_k1"][:fill_rows_k1].fill_(
    float('-inf')
)
self.decode_cuda_graph_metadata["compress_k2"][:fill_rows_k2].fill_(
    float('-inf')
)
```

### 正确性论证

- 该 `-inf` 填充的目的是「防御性 mask」：在 compress kernel 写入活跃
  compress entry 之前，把多余区域置 `-inf`，防止它们污染后面的
  softmax；
- compress kernel 只**写入**每个 batch entry `i` 的
  `[0, bs × ceil(seq_len_i / kernel_stride))` 区间，且
  `seq_len_i ≤ max_len`。`max_len/stride` 之外的行后续 attention 不会
  读 —— L1825-L1826 紧接着会用
  `metadata.k1.cu_seqlens` / `metadata.k2.cu_seqlens` 把 sparse top-k
  打分边界限定在实际长度内；
- 因此清零 `[0, bs × ceil(max_len/stride))` **足够且安全**：mask
  区域刚好覆盖 kernel 在本 batch 中可能触及的所有位置；
- 用 cudagraph 捕获的 `bs` 替代 `forward_batch.batch_size` 也是正确性
  改进：cudagraph replay 用的张量按 `bs` 大小捕获，用
  `forward_batch.batch_size` 容易出现 capture/replay 大小错配（虽然
  在我们 max-running-requests=8 且无 padding 的情况下两者都等于 8）。

### 预期加速

每 decode step 清零字节数下降 `max_context_len/max_len` 倍：

| 实际 max_len | 减少 |
|--------------|-----------|
|  32 768      | **16×**   |
|  65 536      |  8×       |
| 131 072      |  4×       |
| 262 144      |  2×       |
| 524 288      |  1×（达到上限时） |

我们的公开集（Round 13e 观察到 `max_seq_len_k` ≈ 32K–128K）：
**每 decode step GPU 内存流量降低 4–16×**，且这一步在 cudagraph
replay 关键路径上。

## 验证计划

1. 在 `mixed_minicpm_cudagraph` 应用补丁；
2. push 到 `minicpm-src`；
3. 同步到 fcloud（纯 Python，**不需要重编译 sgl-kernel wheel**）；
4. 用 `SOAR_QUANT_MODE=noquant`（BF16 + 原生稀疏 + FP8 KV）重启
   server；
5. `--concurrency 32` 重跑准确率；
6. **通过标准**：在 1h harness 预算内完成，且 accuracy ≥ 78%
   （Round 13d 在超时前不完整地达到 76%；过填去掉后长上下文 decode
   不再被 per-step 内存流量卡住，更多长样本会在 per-request 3000s
   timeout 内完成，准确率应升高）；
7. 准确率通过则跑速度（s1/s8/smax）和 dense baseline 对比。

## 回滚

```bash
git checkout python/sglang/srt/layers/attention/minicpm_backend.py
```

补丁仅 4 行、单文件、单函数。

## 风险

- **正确性无风险**（mask 区域刚好覆盖 kernel 可能触及的全部位置）；
- **dense 路径无影响** — 这段代码仅在 `--force-dense-minicpm` **关闭**
  时（即稀疏模式）才会进入，当前提交 baseline（dense）不受影响；
- 微小风险：若后续某个 kernel 越过 `bs × ceil(max_len/stride)`
  去读行而不参考 cu_seqlens，可能读到上次更大 batch 残留的非 `-inf`
  数据；先跑完整准确率评测验证后再上提交包。

## 审批后下一步

1. 落补丁 + push；
2. 重启 fcloud，跑准确率 + 速度；
3. 在 [`docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`](TEST_RESULTS_TRACKING.md)
   记录 Test 35；
4. 若稀疏路径恢复可用，评估是否在新 fix 之上重做 GPTQ
   （Round 13e Test 2 — GPTQ + 稀疏 + FP8 KV）。
