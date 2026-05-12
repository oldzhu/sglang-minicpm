# CHANGE_0157: 为 MEDUSA 重新启用 CUDA Graph — TARGET_VERIFY 以 Eager 模式运行

**日期**: 2026-05-13  
**提交**: `dc7300710`  
**分支**: `mixed_minicpm_cudagraph`  
**状态**: 已实现，待 fcloud 验证  
**规则合规性**: ✅ 不违反评分规则  

---

## 1. 背景与动机

Stage 3a Medusa（K=1 验证，提交 `94f6ff6c6`）使用 `SOAR_SPEC_MEDUSA_EAGER=1`
配置进行了验证，该配置会向服务器启动参数添加 `--disable-cuda-graph` 并移除
torch-compile。这个临时方案是必要的，因为在 `SOAR_SPEC_MEDUSA_EAGER=0`
（默认生产配置）下，服务器在第一次 TARGET_VERIFY 前向时会崩溃，报错：

```
AttributeError: 'NgramVerifyInput' object has no attribute 'kv_indptr'
```

本次变更目标：**移除 `SOAR_SPEC_MEDUSA_EAGER=1` 临时方案**，使 Stage 3a 可以在
标准生产配置下运行（GPTQ + FP8 KV + 密集模式 + cuda-graph + torch-compile）。

---

## 2. 根本原因分析

### 2.1 MEDUSA cuda-graph 捕获模式

`CudaGraphRunner.__init__()` 设置：

```python
self.capture_forward_mode = ForwardMode.DECODE   # 默认
if (spec_algorithm.is_eagle()
        or spec_algorithm.is_standalone()
        or spec_algorithm.is_ngram()):
    self.capture_forward_mode = ForwardMode.TARGET_VERIFY
    self.num_tokens_per_bs = speculative_num_draft_tokens
```

由于 `MEDUSA` 不在该判断中，MEDUSA 图运行器只捕获 `DECODE` 图。

### 2.2 使用 DECODE 图执行 TARGET_VERIFY 前向

在 `medusa_worker._forward_verify_k1()` 中：

1. `batch.spec_algorithm = NGRAM`（复用 NGRAM 验证基础设施）
2. `batch.forward_mode = TARGET_VERIFY`
3. `batch.spec_info = NgramVerifyInput(draft_token_num=1, ...)`
4. → `target_worker.forward_batch_generation(model_worker_batch, is_verify=True)`
5. → `model_runner.forward(forward_batch)` → `_forward_raw()`

在 `_forward_raw()` 中：

```python
mode_check = forward_batch.forward_mode.is_cuda_graph  # TARGET_VERIFY 返回 True
can_run_graph = bool(
    mode_check()               # True
    and self.graph_runner      # 存在
    and self.graph_runner.can_run(forward_batch)  # True（无 mode 检查！）
)
# → graph_runner.replay() 被调用
```

`can_run()` **没有对 `forward_mode` 的检查**，因此返回 `True`。

`replay_prepare()` 调用：

```python
attn_backend.init_forward_metadata_replay_cuda_graph(
    bs, ...,
    self.capture_forward_mode,   # DECODE
    forward_batch.spec_info,     # NgramVerifyInput
    ...
)
```

在 DECODE replay 路径中：

```python
# indices_updater_decode.call_begin_forward()
else:
    kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices  # 崩溃！
```

`NgramVerifyInput` 没有 `kv_indptr` 或 `kv_indices` 属性——这些属性只存在于
`EagleVerifyInput` 中。

### 2.3 为何 NGRAM/EAGLE 不受影响

对于 NGRAM/EAGLE，`capture_forward_mode = TARGET_VERIFY`。cuda 图以 TARGET_VERIFY
元数据（预填充路径，使用 `generate_attn_arg_prefill`）捕获。`replay_prepare()`
使用预填充路径，调用 spec_info 的 `generate_attn_arg_prefill()`——不访问
`kv_indptr`。

---

## 3. 修复方案

### 3.1 `CudaGraphRunner.can_run()` — MEDUSA + TARGET_VERIFY 返回 False

**文件**: `python/sglang/srt/model_executor/cuda_graph_runner.py`  
（以及 `benchmark/soar/demo_sala/sglang/python/…/cuda_graph_runner.py`）

在 `is_ngram_supported` 之后添加：

```python
# 对于 MEDUSA：cuda 图以 capture_forward_mode=DECODE 捕获。
# 当验证步骤运行时（forward_mode=TARGET_VERIFY），DECODE 图无法处理
# NgramVerifyInput（它缺少 kv_indptr/kv_indices 属性，而 decode 索引更新器
# 需要这些属性）。在此返回 False，使 TARGET_VERIFY 回退到 eager forward_extend()，
# 后者通过 attn_backend.init_forward_metadata() → generate_attn_arg_prefill()
# 经由 prefill_wrappers_verify 处理——这是正确的代码路径。
# DECODE 步骤不受影响（仍然正常使用 cuda 图）。
is_medusa_verify_ok = not (
    self.model_runner.spec_algorithm.is_medusa()
    and forward_batch.forward_mode.is_target_verify()
)

return (
    is_bs_supported
    and is_encoder_lens_supported
    and is_tbo_supported
    and capture_hidden_mode_matches
    and is_ngram_supported
    and is_medusa_verify_ok      # ← 新增
)
```

**效果**：对于 MEDUSA + TARGET_VERIFY，`can_run() = False` →
`_forward_raw()` 回退到 eager 分发。DECODE 步骤不变。

### 3.2 `ModelRunner._forward_raw()` — 将 TARGET_VERIFY 路由到 `forward_extend()`

**文件**: `python/sglang/srt/model_executor/model_runner.py`  
（以及 `benchmark/soar/demo_sala/sglang/python/…/model_runner.py`）

将 extend 分发分支从：

```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
    ret = self.forward_extend(...)
```

改为：

```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) or (
    forward_batch.forward_mode.is_target_verify()
    and self.spec_algorithm.is_medusa()
):
    # TARGET_VERIFY + MEDUSA：验证步骤不捕获 cuda 图
    # (capture_forward_mode=DECODE for MEDUSA)；forward_extend() 通过
    # attn_backend.init_forward_metadata() → prefill_wrappers_verify →
    # generate_attn_arg_prefill() 正确处理 TARGET_VERIFY。
    ret = self.forward_extend(...)
```

**为何 `forward_extend()` 对 TARGET_VERIFY 是正确的**：

`forward_extend()` 调用 `self.attn_backend.init_forward_metadata(forward_batch)`。
在 `FlashInferAttnBackend.init_forward_metadata()` 的 `is_target_verify()` 分支中：

```python
elif forward_batch.forward_mode.is_target_verify():
    self.indices_updater_prefill.update(
        ...,
        prefill_wrappers=self.prefill_wrappers_verify,   # ← 正确的 wrapper
        spec_info=forward_batch.spec_info,
    )
```

这会调用 `NgramVerifyInput` 的 `generate_attn_arg_prefill()`，正确工作，
无需访问 `kv_indptr`。

**NGRAM/EAGLE 不受影响**：它们始终使用 cuda 图（can_run() = True），
因此不会进入此分发分支。

---

## 4. 调用流程摘要（修复后）

| 步骤 | forward_mode | can_run() | 路径 |
|------|-------------|-----------|------|
| 普通 DECODE | DECODE | True | cuda graph replay（DECODE）|
| 预填充/EXTEND | EXTEND | False（非 is_cuda_graph）| eager `forward_extend()` |
| Stage 3a VERIFY | TARGET_VERIFY | **False**（新：MEDUSA 守卫）| eager `forward_extend()` → `init_forward_metadata(TARGET_VERIFY)` → `prefill_wrappers_verify` → `generate_attn_arg_prefill()` |

---

## 5. 修改的文件

| 文件 | 修改行数 | 说明 |
|------|---------|------|
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | +14 | `can_run()` MEDUSA 守卫 |
| `python/sglang/srt/model_executor/model_runner.py` | +9 | `_forward_raw()` TARGET_VERIFY 分发 |
| `benchmark/soar/demo_sala/sglang/python/…/cuda_graph_runner.py` | +14 | 提交副本同步 |
| `benchmark/soar/demo_sala/sglang/python/…/model_runner.py` | +9 | 提交副本同步 |

---

## 6. 验证命令

### 6.1 服务器启动（默认配置，无 SOAR_SPEC_MEDUSA_EAGER 覆盖）

```bash
# 在 fcloud 上 — prepare_env.sh 必须有 SOAR_SPEC_MEDUSA_EAGER=0（默认）
cd /root/submission_sim
source prepare_env.sh
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${SGLANG_SERVER_ARGS[@]}"
```

预期：服务器成功启动，cuda-graph 捕获完成（使用 torch-compile 约 14 分钟）。
推理过程中无 AttributeError。

### 6.2 快速精度检查（MCQ 子集）

```bash
cd /root/data
python3 eval_model_001.py \
  --url http://localhost:30000 \
  --data_path /root/data/perf_public_set.jsonl \
  --task_type mcq \
  --max_questions 20
```

预期：MCQ 精度 ≥ 60%（≥ 12/20），与 Stage 3a GLA 修复基准一致
（EAGER=1 时为 13/20 = 65%）。

### 6.3 完整精度 + 速度基准测试

```bash
# 精度
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 速度基准
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

预期：
- 归一化精度 ≥ 99%（与 Stage 2 cgraph 基准一致）
- S1/S8/Smax 时间大致与 Stage 2 cgraph 基准相当
  （S1=118.28s, S8=43.87s, Smax=35.75s）
- 验证以 eager 模式运行，单 token 开销略高于基准，但 DECODE 仍使用 cuda 图

---

## 7. 结果摘要（待 fcloud 测试后填写）

| 配置 | 提交 | MCQ(20) | 归一化精度 | S1 (s) | S8 (s) | Smax (s) |
|------|------|---------|----------|---------|---------|----------|
| Stage2 cgraph 基准 | 46553947b | — | 80.11% | 118.28 | 43.87 | 35.75 |
| Stage3a GLA 修复 (EAGER=1) | 94f6ff6c6 | 65% | — | — | — | — |
| **Stage3a cgraph (EAGER=0)** | **dc7300710** | 待测 | 待测 | 待测 | 待测 | 待测 |

---

## 8. 回滚说明

回滚到 CHANGE_0155 状态（Stage 3a 使用 EAGER=1）：

```bash
# 选项 A：git revert
git revert dc7300710

# 选项 B：在 prepare_env.sh 中重新设置 SOAR_SPEC_MEDUSA_EAGER=1（快速临时方案）
# 在 benchmark/soar/demo_sala/prepare_env.sh 中修改：
#   SOAR_SPEC_MEDUSA_EAGER=0   →   SOAR_SPEC_MEDUSA_EAGER=1
```

---

## 9. 后续步骤

1. **fcloud 验证**：启动实例，同步，使用默认 `SOAR_SPEC_MEDUSA_EAGER=0` 重启服务器，
   运行精度 + 速度基准测试。
2. **Stage 3b**：训练真实的 Medusa 头（K=1，W1≠0）以获得实际草稿接受加速。
   `medusa_worker.py` 中的部分接受快照/恢复路径已经脚手架完成。
3. **K>1 扩展**：在验证训练头之后，扩展到 K=2+ 以获得更高的接受率。
