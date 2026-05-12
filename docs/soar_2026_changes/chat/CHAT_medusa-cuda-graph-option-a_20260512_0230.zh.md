# 对话日志：Medusa CUDA Graph 重新启用（选项 A）

**会话**: 2026-05-13（文件时间戳：20260512_0230）  
**主题标识**: `medusa-cuda-graph-option-a`  
**相关变更**: CHANGE_0157  
**提交**: `dc7300710`

---

## 1. 用户问题 / 请求

**从上一会话延续的上下文**：上一会话结束时，已找到 MEDUSA + cuda-graph 崩溃
（`kv_indptr` AttributeError）的根本原因，但修复代码尚未编写。用户已批准选项 A
方案。

**本会话开始**：（隐式延续——用户返回会话，期望选项 A 实现继续推进。）

**持续性指令**："请稍微打破规则，我希望你继续推进 medusa 功能直到可以在 fcloud
上测试，只需保持文档详细清晰，以便我们可以审查、参考和跟踪 medusa 实现。"

---

## 2. Agent 操作

### 2.1 上下文验证

- 读取 `medusa_worker._forward_verify_k1()`（第 80–307 行），确认调用流程：
  `target_worker.forward_batch_generation(model_worker_batch, is_verify=True)`
- 读取 `tp_worker.forward_batch_generation()`（第 425–495 行），理解 `is_verify=True`
  的作用：创建 ForwardBatch，调用 `model_runner.forward()`，跳过采样。
- 搜索 `_forward_raw()` 分发，确认 TARGET_VERIFY 没有分发分支（回退时会抛出
  ValueError）。
- 读取 `flashinfer_backend.init_forward_metadata()`（第 420–543 行），确认
  TARGET_VERIFY 使用 `prefill_wrappers_verify` → `generate_attn_arg_prefill()`。
- 验证 `piecewise_cuda_graph_runner.can_run()`——仅检查 `num_tokens`，但我们的设置
  中分段运行器为 `None`（`enable_piecewise_cuda_graph=False`，因为
  `enable_torch_compile=True` 会禁用它）。

### 2.2 修复实现（4 个文件编辑）

**修复 1：`cuda_graph_runner.can_run()`**（主版本和提交副本）

在 `is_ngram_supported` 之后添加 `is_medusa_verify_ok` 变量：
```python
is_medusa_verify_ok = not (
    self.model_runner.spec_algorithm.is_medusa()
    and forward_batch.forward_mode.is_target_verify()
)
```
在返回语句中添加 `and is_medusa_verify_ok`。

**修复 2：`model_runner._forward_raw()` extend 分发**（主版本和提交副本）

将：
```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
```
改为：
```python
elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) or (
    forward_batch.forward_mode.is_target_verify()
    and self.spec_algorithm.is_medusa()
):
```

### 2.3 提交和推送

```
[mixed_minicpm_cudagraph dc7300710] CHANGE_0157: 为 MEDUSA 重新启用 cuda-graph，verify 以 eager 模式运行
4 个文件更改，44 行新增，2 行删除
```
推送到 `minicpm-src`（从不推送到 `origin`）。

### 2.4 文档

- 创建 `docs/soar_2026_changes/CHANGE_0157_medusa_cuda_graph_verify_fix.en.md`
- 创建 `docs/soar_2026_changes/CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md`
- 更新 `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`，添加
  `Stage3a-cgraph (PLANNED)` 行（dc7300710，测试结果待填）。

---

## 3. 关键决策

| 决策 | 理由 |
|------|------|
| 选项 A（DECODE 使用 cuda-graph，verify 使用 eager）| 简洁：DECODE 路径快速（cuda graph），verify 正确（eager extend）。无需新的图捕获。 |
| 两点修复（can_run + _forward_raw）| `can_run()` 防止错误的 DECODE 图重放；`_forward_raw()` 防止回退时的 ValueError。两者都需要。 |
| 仅在 `_forward_raw` 中守卫 `is_medusa()` | 保守：NGRAM/EAGLE 不受影响（它们的 can_run 从不回退）。 |
| TARGET_VERIFY 使用 `forward_extend()` | `init_forward_metadata()` 已有 `is_target_verify()` 分支 → `prefill_wrappers_verify` → `generate_attn_arg_prefill()`。正确路径。 |

---

## 4. 结果

- **CHANGE_0157 已实现并提交**：`dc7300710` 在 `mixed_minicpm_cudagraph` 上。
- **已推送到 minicpm-src** ✅
- **文档已创建** ✅（中英双语，TEST_RESULTS_TRACKING 已更新）
- **fcloud 测试待进行** — 需要用户启动实例并批准测试运行。

---

## 5. 未解决问题 / 后续步骤

1. **需要 fcloud 验证**：
   - 启动实例：`python3 scripts/fcloud/fcloud_workflow.py start-instance`
   - 同步 + 重启：`python3 scripts/fcloud/fcloud_workflow.py sync && restart-server`
   - 服务器使用默认 `SOAR_SPEC_MEDUSA_EAGER=0`（无需覆盖）。
   - 快速 MCQ 测试：预期 ≥ 60%（≥ 12/20）。
   - 完整精度 + 速度：预期与 Stage 2 cgraph 基准相当。

2. **验证后**：Stage 3b（真实 Medusa 头，K=1 训练，W1≠0）。

3. **提交文档**：CHANGE_0157 文档和对话日志应提交到 `minicpm-src`。

---

## 6. 交叉引用

| 类型 | 引用 |
|------|------|
| 变更文档（英文）| [CHANGE_0157_medusa_cuda_graph_verify_fix.en.md](CHANGE_0157_medusa_cuda_graph_verify_fix.en.md) |
| 变更文档（中文）| [CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md](CHANGE_0157_medusa_cuda_graph_verify_fix.zh.md) |
| 前置变更 | [CHANGE_0155_stage3a_gla_fix.en.md](CHANGE_0155_stage3a_gla_fix.en.md) |
| 测试跟踪 | [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) 行 `Stage3a-cgraph (PLANNED)` |
| 提交 | `dc7300710` 在 `mixed_minicpm_cudagraph` 上 |
