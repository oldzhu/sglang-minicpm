# CHAT — Medusa Phase R1b Stage 2 —— 重新启用 cuda-graph（续 _001）

**主题**：诊断 Stage 2 eager 基线 S1 +65% 的原因；把 eager 模式 strip 改成可选项，重新启用 cuda-graph + torch.compile；验证。
**会话区间**：2026-05-11 07:30 → 2026-05-11 09:00 本地。
**分支 / 最新提交**：`mixed_minicpm_cudagraph` @ `46553947b`（本轮）→ `<待定>`（本提交）。
**fcloud 实例**：`ai-e7e98a7c52`。

## 1. 用户提问

1. *"在去 stage3 之前，stage2 启用 cuda 时的崩溃怎么办？S1 +65% 慢，担心 medusa 自身的提速补不回 cuda 关掉的损失。"*
2. *"go"*（同意按 plan 推进；我的进一步分析显示 cuda-graph 崩溃与 bug #2 同源，因此实际改动只是去掉 eager strip）。

## 2. Agent 分析

复盘 addendum #1 在 `prepare_env.sh` 的注释：

> cuda_graph_runner 在 is_eagle/is_standalone/is_ngram 里都不认 MEDUSA → 按 num_tokens_per_bs=1 捕获。但 spec_algorithm != NONE，scheduler.run_batch 把 ScheduleBatch 直接送进 model_worker.forward_batch_generation，prefill batch 被 replay 到 [1]-shape buffer 上 → "shape [1] doesn't match broadcast shape [N]"。

这个描述其实和 bug #2 同源（首个 decode 用到了 prefill 残留 metadata）。补丁 #2 之后 worker 自己重跑 `prepare_for_decode`；补丁 #3 之后 `batch.spec_algorithm` 在调 `get_model_worker_batch()` 之前已经被翻成 NONE：

- `ForwardBatch.spec_algorithm = NONE`、`num_tokens_per_seq = 1`
- `cuda_graph_runner.get_spec_info()` 在 MEDUSA 下返回 None（不在 eagle/standalone/ngram 分支）→ 捕获的图与普通 decode（`num_tokens_per_bs=1`）字节相同
- replay：`forward_batch.batch_size=N` → `graph_key=N` → 命中

**结论**：addendum #1 的 eager strip 过度防御了。去掉它应该既不崩溃，也能恢复 Test 12 的速度。

## 3. 实现

`prepare_env.sh`：新增开关 `SOAR_SPEC_MEDUSA_EAGER`，默认 `0`。默认情况下 Medusa 路径保留 v22 基线 args（`--enable-torch-compile --torch-compile-max-bs 24`、cuda-graph 开）；`SOAR_SPEC_MEDUSA_EAGER=1` 时套用旧 strip 作为回滚开关。

单 commit：`46553947b`（`feat(medusa-r1b): re-enable cuda-graph + torch.compile for Medusa pass-through (Stage 2 §14)`）。

## 4. 结果

Server 启动 ~257s（16 个 bs 桶的 cuda-graph 捕获 + torch.compile autotune）。
首请求未崩溃，eval 全程跑完。

| 指标 | **Stage 2 (eager)** | **Stage 2 + cuda-graph + torch.compile** | Test 12 baseline | vs Test 12 |
|------|---------------------|--------------------------------------------|------------------|-------------|
| ori_accuracy | 76.38% | **80.11%** | 79.29% | **+0.82pt** |
| normalized | 95.48% | **100.14%** | 99.11% | — |
| C | 0 | **1.0** | 1.0 | — |
| mcq | 43.33% | 56.67% | 63.33% | −6.66pt（落在本地噪声带） |
| cwe | 86.33% | 85.00% | 72.00% | +13.00pt |
| fwe | 98.89% | 98.89% | 97.78% | +1.11pt |
| niah | 96.67% | 100.00% | 100.00% | = |
| qa | 56.67% | 60.00% | 63.33% | −3.33pt |
| Accuracy 总耗时 | ~3540s | **2845.05s** | 4244s | −33% |
| S1 | 200.92s | **118.28s** | 121.71s | **−2.8%** |
| S8 | n/a | **43.87s** | 44.09s | **−0.5%** |
| Smax | n/a | **35.75s** | 35.86s | **−0.3%** |
| TPOT (S1) | 11.73ms | **6.74ms** | — | — |

**假设成立。** Bug #1 与 bug #2 同源，#2/#3 修完之后 eager strip 已经多余。Stage 2 在三档速度上都基本追平/略优于 Test 12，accuracy 也略高一点点。

这就是干净的 **Stage 2 基线**——MedusaWorker 已经接好，cuda-graph + torch.compile 端到端可用，且通过 MedusaWorker 这层 pass-through 没有任何速度损耗。Stage 3（真 verify + rewind）现在可以在这个基线上做公平评测。

## 5. 关联文档 / 提交

- 代码：`benchmark/soar/demo_sala/prepare_env.sh`（本轮）。`python/sglang/srt/speculative/medusa_worker.py`（未改 —— bug #2/#3 修复已在上一轮搞定）。
- 文档：[CHANGE_0155_medusa_phase_r1b_stage2.en.md §14](../CHANGE_0155_medusa_phase_r1b_stage2.en.md) / [.zh.md §14](../CHANGE_0155_medusa_phase_r1b_stage2.zh.md)。
- 跟踪表：`TEST_RESULTS_TRACKING.md` → 行 `Medusa-Stage2-cgraph`（原 eager 行 `Medusa-Stage2` 标记为 superseded）。
- Commit：`46553947b`（re-enable 改动）+ 本文档 commit。
- fcloud 上预测：`/root/data/outputs/20260511_075837/predictions.jsonl`。

## 6. 下一步

1. **进 Stage 3**。Stage 3 = 真 Medusa verify（1 个 head → 每个 seq 1 个 draft token）+ 被拒时 rewind。在 Test 12 ≈ Stage 2 的现状下，Stage 3 带来的 S1/S8/Smax 改动可以干净归因到 Medusa 的接受率上。
2. Stage 3 设计约束（CHANGE_0154 已写）：
   - Draft = 1（Stage 3）；Stage 4 再扩到多 head 树。
   - Verify forward_mode = `TARGET_VERIFY`；这条路径 scheduler/output-processor 已经原生支持，R1b 这几个 bug 大概率不会重现。
   - 接受判定：draft head 的 argmax 与 target argmax 一致 → 接受，写两个 token、seq_lens += 2；否则只写 target token、seq_lens += 1。
3. **遗留问题**：Medusa head 当前未训练；Stage 3 接受率上限大约 50%（随机 head ≈ 随机 token）。真正想要的提速来自 (a) 用 MiniCPM-SALA 输出对 head 做微调，或 (b) 用 LM head + skip-connection 初始化 head（CHANGE_0154 §6）。
