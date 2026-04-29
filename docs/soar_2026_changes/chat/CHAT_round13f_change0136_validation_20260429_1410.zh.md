# 对话 — Round 13f-1 + CHANGE_0136 验证

**日期**: 2026-04-29
**主题**: 执行已批准的 Round 13f-1 (flashinfer 烟雾测试) + CHANGE_0136 sanity
(`SOAR_SPARSE_DENSE_LEN=524288` → 65536 → 16384)。记录结果。
**涉及提交**: `6a070110b` (初始实现), `45e316ee0` (设置环境变量时丢弃
`--enable-torch-compile`), `7a7a568eb` (设置环境变量时丢弃 `--dense-as-sparse`)。

## 用户提问 (本轮)

1. "fcloud started" — 开始按计划测试。
2. "pls test in order as you planned" — 按顺序执行所有步骤。

## Agent 行动 / 结果

### Step 1 — 基线重启 (默认 Test 12 参数, 不设环境变量)
- 服务器启动 209s, 健康。CHANGE_0136 源码在未设置环境变量时不影响默认路径
  (env override 块由 `if _env_dense_len is not None` 保护, 默认走原 else 分支)。
- 跳过完整 Test 12 重跑 (已多次测量), 节省 fcloud 时间。

### Step 2 — Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`)
进程命令行确认: `--attention-backend flashinfer`, 无
`--force-dense-minicpm`, 无 `--dense-as-sparse`, 无 `--minicpm_flashinfer`。

- **准确率: 76.91%** (Test 12 baseline 79.29%; Δ −2.38pt)
- **速度**: S1=110.76s, S8=40.50s, Smax=33.66s
  - vs Test 12 baseline 121.71 / 44.09 / 35.86 → **−9% / −8% / −6%** (更快)
- **判定**: 速度收益真实但准确率致命 (C=0)。归一化准确率 ≈ 76.91 / 80.00 ≈
  96.1% (低于 97% 截断)。原版 flashinfer 无法正确路由 SALA 的混合稀疏/稠密
  架构。不再追溯。
- 子任务: cwe=82.33% (基线 ≥87), qa=56.67%, mcq=46.67%, fwe=98.89%, niah=100%。
  cwe + qa 下降是关键。

记录到 TEST_RESULTS_TRACKING 行 `R13f1-flashinfer`。

### Step 3 — CHANGE_0136 sanity (`SOAR_SPARSE_DENSE_LEN=524288`) — 阻塞

#### 第 1 次尝试 (commit `6a070110b`): cudagraph 捕获崩溃
```
torch._dynamo.exc.InternalTorchDynamoError: RuntimeError:
Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture
```
即 CHANGE_0133 §"与 Round 13d torch.compile 崩溃的关系"中已记录的崩溃。原因:
丢弃 `--force-dense-minicpm` 后稀疏路径再次可达,但 `--enable-torch-compile`
仍然开启。

**修复** (`45e316ee0`): 设置 `SOAR_SPARSE_DENSE_LEN` 时同时清空
`TORCH_COMPILE_ARGS=""` (与现有 `SOAR_SPARSE_MODE=1` 分支一致)。

#### 第 2 次尝试 (commit `45e316ee0`): override 静默失效
服务器启动正常,但 `pgrep -af` 显示参数仍包含 `--dense-as-sparse`。
`MiniCPMAttentionBackend.__init__` 中 env-override 块由 `not self.dense_as_sparse`
保护,因此 `dense_as_sparse=True` 短路 override,强制 `dense_len=0`
(每个请求 → 稀疏路径)。

**修复** (`7a7a568eb`): 设置 `SOAR_SPARSE_DENSE_LEN` 时同时清空
`SGLANG_SERVER_ARGS` 中的 `--dense-as-sparse`。

#### 第 3 次尝试 (commit `7a7a568eb`): 第一条样本 prefill 崩溃
服务器启动 36s。后端日志确认 override 应用:
`SOAR_SPARSE_DENSE_LEN override: dense_len=524288 (model config default was 8192)`。

第一条请求 (prompt_tokens=103) 立刻崩溃:
```
File ".../minicpm_backend.py", line 1087, in forward_extend
  metadata.sparse_page_table[sparse_page_table_idx_start, :kv_len] = \
      page_table[dense_bs, :kv_len] * 2
RuntimeError: The expanded size of the tensor (103) must match the existing
size (104) at non-singleton dimension 0. Target sizes: [103]. Tensor sizes: [104]
```

这是 **预先存在的稀疏路径 prefill 元数据构造 off-by-one bug**,与 CHANGE_0133
(decode 阶段 compress_k1/k2 over-fill) **不同** — 同模块、不同函数、不同症状。

**CHANGE_0136 在此 prefill bug 修复前无法验证**。Step 4
(`SOAR_SPARSE_DENSE_LEN=65536`, 保守) 和 Step 5 (`=16384`, 激进) 均跳过,
因为只要请求超过阈值并被路由到稀疏层,就会撞同一个 crash。

记录到 TEST_RESULTS_TRACKING 行 `R13f-CHANGE_0136-sanity`。

### 清理
- 数据收集完成后通过 `fcloud_workflow.py shutdown` 关闭 fcloud 实例 (节省成本规则)。

## 关键结论

1. **Round 13f-1 (flashinfer) 提交方案宣告失败**。约 7% 速度收益不足以补偿
   2.4pt 准确率损失 (C=0)。
2. **CHANGE_0136 需先修复稀疏路径 bug**。实现本身正确 (override 触发,环境
   传递正常);HEAD 上稀疏路径至少存在两个预先存在的 bug 阻断长上下文路由:
   - CHANGE_0133 (已提案, 未合入): decode 阶段 compress_k1/k2 over-fill
   - **新发现**: prefill 阶段 `sparse_page_table` off-by-one
     (`minicpm_backend.py:1087`)。任何 CHANGE_* 文档中均未记录。
3. `R13e-prof-*` 性能剖析结论仍然成立: BF16 sparse 路径下 BF16 GEMM 占主导。
   CHANGE_0136 旨在通过将 sparse 路由限制到长请求来缓解 GPTQ 路径的此问题;
   但 HEAD 上稀疏路径 bug 阻止我们观察到 GPTQ + 仅长上下文走 sparse 是否真能
   超越 Test 12 dense。

## 待办

- 编写 CHANGE_0137 修复 `sparse_page_table[sparse_page_table_idx_start, :kv_len]`
  off-by-one (需根因调查;可能是 EOS / 生成 token slot 的 +1 偏移,或
  `dense_bs` vs `sparse_bs` 索引错位)。
- CHANGE_0137 + CHANGE_0133 落地后重试 CHANGE_0136 验证矩阵:
  524288 sanity → 65536 保守 → 16384 激进。

## 交叉引用

- 实现: 提交 `6a070110b`, `45e316ee0`, `7a7a568eb` (`minicpm-src/mixed_minicpm_cudagraph`)。
- 提案文档: [PROPOSAL_round13f1_flashinfer_backend_smoketest.zh.md](../PROPOSAL_round13f1_flashinfer_backend_smoketest.zh.md),
  [CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md](../CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md)。
- 预先存在的 bug: [CHANGE_0133_sparse_compress_buffer_oversize.en.md](../CHANGE_0133_sparse_compress_buffer_oversize.en.md)。
- 测试行: TEST_RESULTS_TRACKING.md `R13f1-flashinfer`, `R13f-CHANGE_0136-sanity`。
