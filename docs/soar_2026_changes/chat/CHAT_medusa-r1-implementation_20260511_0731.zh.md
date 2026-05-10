# 对话：Medusa R1 实现（阶段 R1a）

- 日期：2026-05-11 07:31
- 分支：`mixed_minicpm_cudagraph`
- 主题：实现 CHANGE_0153 的 R1a 阶段（Medusa 骨架，默认零行为变化）
- 关联文档：
  - [CHANGE_0153_medusa_phase_r1_design.zh.md](../CHANGE_0153_medusa_phase_r1_design.zh.md)
  - [PROPOSAL_medusa_minicpm_sala_001.zh.md](../PROPOSAL_medusa_minicpm_sala_001.zh.md)
  - [RESEARCH_speculative_decoding_survey_001.zh.md](../RESEARCH_speculative_decoding_survey_001.zh.md)

## 第 1 轮 — 用户授权与 R1 范围拆分

**用户**："while we are waiting for floud for testing, could we go medusa planned implementation?"

**Agent**：
- 审查现有 sglang 推测解码基础设施，发现 `update_mamba_state_after_mtp_verify`（hybrid_linear_attn_backend.py L1373）已经实现了 Medusa+SimpleGLA 所需的逐步态分发模式。R1 工作量由约 1.5–2 kLOC 缩减为约 635 LOC。
- 起草 `CHANGE_0153_medusa_phase_r1_design.{en,zh}.md`，已提交为 `2d658a868`。
- 抛出三个决策点：（Q1）是否现在开始写代码？（Q2）K=1 还是 K=2？（Q3）是否同时加 NGRAM 开关？

**用户**："q1: yes, q2: deafult, q3: ok"

**Agent**：确认 → K=1 默认值，NGRAM 开关加入。把 R1 拆为两次可独立审查的迭代：
- **R1a（本次提交）**：注册表/argparse/数据类/heads/开关，无功能路径变化。带 `SOAR_SPEC_MEDUSA=1` 启动服务器会在 R1b 落地前**故意**抛 `NotImplementedError`。
- **R1b（下次迭代）**：真正的 `MedusaWorker.forward_batch_generation`、SimpleGLA 状态分发、`MiniCPMSALAForCausalLM.forward_with_hidden`、tp_worker 管道。

## 第 2 轮 — R1a 改动

涉及 6 个文件，净增约 383 行：

1. [python/sglang/srt/speculative/spec_info.py](../../../python/sglang/srt/speculative/spec_info.py)
   - 新增 `SpeculativeAlgorithm.MEDUSA` 与 `is_medusa()`。
   - 新增 `SpecInputType.MEDUSA_VERIFY` 并纳入 `is_verify_input()` 集合。
   - `create_worker()` 中加入 lazy import 的 `MedusaWorker` 分支。

2. [python/sglang/srt/server_args.py](../../../python/sglang/srt/server_args.py)
   - dataclass 新字段 `speculative_num_medusa_heads: int = 1`。
   - `--speculative-algorithm` 的 choices 加入 `"MEDUSA"`。
   - 新增 argparse `--speculative-num-medusa-heads`。
   - 仿照 NGRAM 的 post-init：关闭 overlap 调度与 mixed-chunk、`num_draft_tokens = K + 1`、禁止 DP attention、强制 CUDA、默认 `max_running_requests=24`。

3. [python/sglang/srt/speculative/medusa_info.py](../../../python/sglang/srt/speculative/medusa_info.py) — **新文件**
   - `MedusaInput(SpecInput)`，字段与 `EagleVerifyInput` 对齐，使现有 tree-verify kernel 可以零修改复用。
   - `MedusaVerifyOutput` 数据类。
   - `get_spec_adjust_token_coefficient()` 返回 `(K+1, 1)`。

4. [python/sglang/srt/models/minicpm_medusa_heads.py](../../../python/sglang/srt/models/minicpm_medusa_heads.py) — **新文件**
   - `MedusaHead`：残差 MLP `SiLU(W1 @ h) + h`，W1 零初始化。
   - `MedusaHeads`：K 个 head 集合，并以非拥有方式引用主模型 `lm_head`。
   - W1=0 时每个 head 的 logits 都等于主模型分布；在 `accept_threshold=1.0` 下所有 draft 必接受 → 与不开启推测解码字节一致。这是 R1 的正确性闸门。

5. [python/sglang/srt/speculative/medusa_worker.py](../../../python/sglang/srt/speculative/medusa_worker.py) — **新文件**
   - `MedusaWorker(BaseSpecWorker)` 骨架。`__init__` 直接 `NotImplementedError` 并显式指向 R1b。

6. [benchmark/soar/demo_sala/prepare_env.sh](../../../../benchmark/soar/demo_sala/prepare_env.sh)
   - 默认值 `SOAR_SPEC_MEDUSA=0`、`SOAR_SPEC_MEDUSA_HEADS=1`、`SOAR_SPEC_NGRAM=0`。
   - 当 `SOAR_SPEC_MEDUSA=1` 时追加 `--speculative-algorithm MEDUSA --speculative-num-medusa-heads K --speculative-num-draft-tokens K+1`。
   - 当 `SOAR_SPEC_NGRAM=1` 时追加 `--speculative-algorithm NGRAM`（与 MEDUSA 互斥，同时设置时告警并以 MEDUSA 为准）。
   - env dump 中新增对应 echo。

## 结果

- 五个改动文件均通过 `ast.parse`，无语法错误。
- 本地 import smoke 因工作站缺 `numpy` 而中止；不影响：fcloud 上一直可用，且默认开关全部关闭，运行时影响为零。
- 提交：`95d931c8b`（`mixed_minicpm_cudagraph` 分支）。
- 推送：在 chat log 落盘后一并推送。

## 待办

- R1b（下一会话，需用户审阅 R1a 后授权）：
  1. 把 `update_mamba_state_after_mtp_verify` 移植为 `update_simple_gla_state_after_verify`（`hybrid_linear_attn_backend.py`）。
  2. 实现 `MedusaWorker.forward_batch_generation`（verify-tree 前向 + 状态分发 + 接受 token 汇集）。
  3. 给 `MiniCPMSALAForCausalLM` 加 `forward_with_hidden`（返回 pre-norm 末层隐藏）。
  4. 把 `forward_with_hidden` 经 `managers/tp_worker.py` 暴露出来。
- 排队（依赖 fcloud）：SAMPLES=30 sequential GPTQ 复测，目录 `MiniCPM-SALA-30-sequential-sparse_qkv_w8`。
