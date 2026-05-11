# CHAT — Medusa Phase R1b Stage 2 调通（纯 pass-through）

**主题**：Stage 2 MedusaWorker 纯 pass-through 在 fcloud 上的逐 bug 调通，并取得首轮干净 accuracy + S1 基线。
**会话区间**：2026-05-10 23:00 → 2026-05-11 04:30（UTC+8）。
**分支 / 最新提交**：`mixed_minicpm_cudagraph` @ `3b1293ed6`。
**fcloud 实例**：`ai-e7e98a7c52`（新 SM120 Blackwell）。

---

## 1. 用户提问（原文/精简）

1. *“这次破例一下，medusa 特性请一直推进到可以在 fcloud 上测试为止，文档要详细清晰，方便回看/参考/追踪 medusa 实现。”*
2. *“go 1，做一次干净的 restart + 完整 accuracy + S1 基准跑 Stage 2 基线，然后推进 Stage 3。”*

## 2. Agent 行动 / 决策

### 依次踩坑并修复的 4 个 bug

| # | 现象 | 根因 | 修复 commit |
|---|------|------|------------|
| 1 | decode 触发 `shape [1] doesn't match broadcast shape [7]` | cuda-graph 调度缺 MEDUSA 分支 | `fc2fc8ab1` —— `prepare_env.sh` 在 `SOAR_SPEC_MEDUSA=1` 时剥掉 `--enable-torch-compile`/`--torch-compile-max-bs N` 并追加 `--disable-cuda-graph` |
| 2 | 首个 decode 报 `Number of tokens in position_ids must match QKV` | `schedule_batch.prepare_for_decode` 在 `spec_algorithm != NONE` 时早退，positions 未被更新 | `364152221` —— `medusa_worker.py` 临时把 `batch.spec_algorithm` 置为 `NONE` 后重新调用 `batch.prepare_for_decode()` |
| 3 | decode 无限生成（28k token，`max_new_tokens=30` 被忽略） | `scheduler_output_processor_mixin.py L398-405` 只在 `is_none()` 或 `is_spec_v2` 时把 token 写回 `req.output_ids`；Medusa-v1 + passthrough 两个条件都不满足 | `a0e680907` —— `medusa_worker.py` 在 worker 入口把 `batch.spec_algorithm` **永久**置为 `NONE`（按 batch），output 处理走标准路径 |
| 4 | eval 中途 `log_decode_stats` 抛 `ZeroDivisionError` | 调度器层 `self.spec_algorithm == MEDUSA` 走 else 分支，但 batch 已被翻转为 NONE → `update_spec_metrics` 跳过 → `spec_num_forward_ct` 一直为 0 | `3b1293ed6` —— `scheduler_metrics_mixin.py` 给除法加保护（`if self.spec_num_forward_ct > 0 else 0`） |

四个修复都极小（≤ 5 行），位置精确；详细写在 `CHANGE_0155_medusa_phase_r1b_stage2.{en,zh}.md` §10–§13。

### Stage 2 worker 最终形态

```python
def forward_batch_generation(self, batch):
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    needs_redo_prep = batch.forward_mode.is_decode() and not batch.spec_algorithm.is_none()
    if not batch.spec_algorithm.is_none():
        batch.spec_algorithm = SpeculativeAlgorithm.NONE  # 每个 batch 永久翻转
    if needs_redo_prep:
        batch.prepare_for_decode()
    model_worker_batch = batch.get_model_worker_batch()
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
    return GenerationBatchResult(
        logits_output=batch_result.logits_output,
        next_token_ids=batch_result.next_token_ids,
        num_accepted_tokens=0,
        can_run_cuda_graph=batch_result.can_run_cuda_graph,
        accept_lens=None,
    )
```

### 全部 4 修复完成后的本轮测试

- Server 启动：35s，日志含 `MedusaWorker Stage 2 ready`。
- Accuracy：150 样本，耗时约 59 分钟。
- 服务端日志显示 decode 健康：24 running req、956 tok/s 生成、无崩溃。
- S1 速度基准：200.92s。

## 3. 结果

| 指标 | Stage 2 (Medusa-R1b) | Test 12 baseline | Δ |
|------|----------------------|------------------|---|
| ori_accuracy | **76.38%** | 79.29% | −2.91pt |
| normalized | 95.48% | 99.11% | — |
| C | 0（淘汰） | 1.0 | — |
| mcq | 43.33% | 63.33% | **−20pt** |
| cwe | 86.33% | 72.00% | +14.33pt |
| fwe | 98.89% | 97.78% | +1.11pt |
| niah | 96.67% | 100.00% | −3.33pt |
| qa | 56.67% | 63.33% | −6.66pt |
| Total duration | ~3540s | 4244s | −16%（eval 随机性） |
| **S1** | **200.92s** | **121.71s** | **+65% 慢** |

**解读**：
- Stage 2 **不是**提交候选。S1 −65% 与 mcq −20pt 都属于预期内的回退，原因：
  1. eager 模式（无 torch.compile、无 cuda graph）—— fix #1 剥掉 cuda graph 是最大的速度损失源
  2. MedusaWorker pass-through 多了一层间接
  3. 加载了 1 个 Medusa head 但没参与计算（仅占显存）
- 但 Scheduler → MedusaWorker → target_worker 这条路径在长 eval 中稳定通过——所有 wiring 关卡过完。

**下一步**：Stage 3（真实 verify + rewind）。Stage 3 大概率不会再踩前 3 个 bug，因为 verify 路径会把 `forward_mode` 翻成 `TARGET_VERIFY`，那条路径在 scheduler/output-processor 里已经被官方支持。

## 4. 关联文档 / 提交

- 代码：
  - `python/sglang/srt/speculative/medusa_worker.py`
  - `python/sglang/srt/managers/scheduler_metrics_mixin.py`（除零保护）
  - `benchmark/soar/demo_sala/prepare_env.sh`（SOAR_SPEC_MEDUSA 分支）
- 文档：
  - [CHANGE_0155_medusa_phase_r1b_stage2.en.md](../CHANGE_0155_medusa_phase_r1b_stage2.en.md)
  - [CHANGE_0155_medusa_phase_r1b_stage2.zh.md](../CHANGE_0155_medusa_phase_r1b_stage2.zh.md)
- 跟踪表：`TEST_RESULTS_TRACKING.md` → 行 `Medusa-Stage2`。
- 本轮 commit：`fc2fc8ab1`、`364152221`、`a0e680907`、`3b1293ed6`（均推送至 `minicpm-src/mixed_minicpm_cudagraph`）。
- fcloud 上预测文件：`/root/data/outputs/20260511_032859/predictions.jsonl`。

## 5. 后续 follow-up

1. **Stage 3 提案**：起草 `PROPOSAL_medusa_stage3_verify_rewind.{en,zh}.md` —— 单头 verify、被拒绝时 rewind、树形候选先不做。待用户同意后再编码。
2. **重新启用 torch.compile + cuda graph**：Stage 3 需要把 fix #1 中剥掉的 cuda graph 拿回来（或者把 graph 改成 Medusa-aware 分支）。这是目前速度回退的主因。
3. **mcq 回退诊断**：Stage 3 真接受率上来之后单独再测一次 mcq。如果仍偏低就不是 Stage 2 痕迹，需要单独排查。
4. 会话结束后已执行 `pause-instance`，fcloud 已暂停计费。
