# CHANGE_0155 — Medusa 第 R1b 阶段 Stage 2(Heads-Shadow 接通烟囱测试)

- **状态**:已实现,等待 fcloud 验证。
- **分支**:`mixed_minicpm_cudagraph`
- **前置**:[CHANGE_0154_medusa_phase_r1b_design.zh.md](CHANGE_0154_medusa_phase_r1b_design.zh.md)(Stage 1 helpers)。
- **后继**:待定(Stage 3 = 完整 verify + 回退)。

## 1. 目的

CHANGE_0154 R1b 的 Stage 2。目标是**接通验证**:证明在
`SOAR_SPEC_MEDUSA=1` 下启动的 server 能够正常启动、在 GPU 上实例化
Medusa heads,并产生**与基线逐字节相同**的输出。

Stage 2 在 fcloud 通过后,Stage 3 将在同一接线上叠加真正的 verify +
回退路径。

## 2. 范围(刻意最小化)

Stage 2 **不**触碰 verify 路径:

- `forward_mode` 永远不切换到 `TARGET_VERIFY`。
- `capture_hidden_mode` 永远不被设置为 `LAST`。
- Stage 1 的 helpers(`snapshot_state_for_spec` / `restore_state_for_spec`)
  保持未被调用。
- 不产生 draft token;`num_accepted_tokens` 恒为 0。

这使 Stage 2 是**纯透传 worker** + 一次权重分配。如果精度不是逐字节
相同,Bug 必在 delegation/wiring(不在 Medusa 算法本身)。

## 3. 为什么实际改动比设计文档更精简

CHANGE_0154 §4 原先建议在六处调用点(scheduler、model_runner ×3、
cuda_graph_runner ×3)添加 `is_medusa()`。本次会话的代码巡查显示:
这些调用点只在 MEDUSA **确实**触发 TARGET_VERIFY capture 时才有意义。
Stage 2 保持在普通 DECODE 模式 → 不触发 TARGET_VERIFY capture →
无需新增分支。

Stage 2 实际改动文件:**2 个**(从 6 个降下来)。

## 4. 实现

### 4a. `python/sglang/srt/speculative/medusa_worker.py`(整体重写)

之前:R1a 留下的 stub,直接抛 `NotImplementedError`(CHANGE_0153)。

之后:~150 行实现,关键点:

- 普通类(对齐 `NGRAMWorker` 的风格,**不**继承 `ABC`),scheduler
  现有的 `draft_worker.forward_batch_generation(batch)` 调用路径直接可用。
- `__init__` 签名匹配 scheduler 的 `draft_worker_kwargs`:
  `(server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker)`。
- 用 `MedusaHeads(hidden_size, num_heads, lm_head, dtype)` 在 GPU
  上分配,`dtype` 取自 `model_runner.dtype`(SOAR 提交配置下为 bfloat16)。
- 校验 `target_model.lm_head` 存在(防御性 guardrail,仅会在不支持的
  模型类上触发)。
- `forward_batch_generation(batch)` 调用
  `target_worker.forward_batch_generation(batch.get_model_worker_batch())`,
  把结果包成 `GenerationBatchResult`(`num_accepted_tokens=0,
  accept_lens=None`)。
- 在 init 时记录一行汇总日志:`K`、`hidden`、`dtype`、
  `approx_weight_MiB`。K=1、hidden=4096、bf16 → 约 32 MiB。

### 4b. `python/sglang/srt/managers/scheduler.py` L866

在 `init_disaggregation` 中跳过 `draft_token_to_kv_pool` 的分支里追加
`or self.spec_algorithm.is_medusa()`。Medusa 没有独立 draft model,
必须走和 NGRAM 一样的路径。

这是防御性改动 — `init_disaggregation` 只在 PD 分离部署中调用
(SOAR 不用),但加守卫能保持契约正确。

### 4c. Stage 2 **没有**修改的文件

- `model_runner.py` L1702(`_is_flashinfer_available` 判断)— MEDUSA 走
  和 no-spec 相同的 target 路径,无需改动。
- `model_runner.py` L1730(`_dummy_run` 预热)— Stage 2 在 DECODE 模式,
  不触达 `TARGET_VERIFY` 分支。
- `model_runner.py` L1860(`get_spec_info`)— Stage 2 不在 batch 上放
  spec_info。
- `cuda_graph_runner.py` L281(capture forward 模式)— Stage 2 在 DECODE,
  现有默认 `num_tokens_per_bs=1` 正确。
- `cuda_graph_runner.py` L431(`is_ngram_supported`)— Medusa Stage 2 的
  `input_ids.numel() == batch_size * 1`;当算法为 MEDUSA 时
  `is_ngram_supported = True` 也满足。
- `cuda_graph_runner.py` L909(capture 端 `get_spec_info`)— Stage 2 没有
  spec_info 要 capture。

Stage 3 接入 verify 路径时,会逐一重新评估这些位置。

## 5. 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| `target_worker.forward_batch_generation` 的 `is_verify` 参数签名不匹配 | 低 | Stage 2 从不传 `is_verify=True`;调用方式与 `NGRAMWorker` 默认路径一致 |
| `MedusaHeads(lm_head=ref)` 在权重加载时被重复计数 | 低 | `_lm_head` 用 1 元素 list 包(非 submodule),已在 `minicpm_medusa_heads.py` 中确认 |
| `model.lm_head` 在边缘模型上不存在 | 低 | 显式 `hasattr` 守卫并抛清晰的 `RuntimeError` |
| Server 找不到 `--speculative-num-medusa-heads` | 低 | R1a 已在 `server_args.py` 中接入(CHANGE_0153) |
| 32 MiB 头部分配 OOM | 可忽略 | 占 84 GB GDDR7 的 0.04% |
| `init_disaggregation` 回归 | 低 | 单行守卫,单实例 SOAR 部署根本不进入这条路径 |

## 6. 验证计划(fcloud)

sync + 重启 server 之后:

```bash
# 1. 基线复测(应与之前测试一致):
SOAR_SPEC_MEDUSA=0 SOAR_SPEC_NGRAM=0 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # 预期 ≈79.29% ori_acc

# 2. Stage 2 烟囱测试:
SOAR_SPEC_MEDUSA=1 SOAR_SPEC_MEDUSA_HEADS=1 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # 必须与基线逐字节相同
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 200   # 验证 "MedusaWorker Stage 2 ready"
```

### 通过判据

- ✅ Server 启动不崩。
- ✅ Server 日志含 `MedusaWorker Stage 2 ready` 一行。
- ✅ 精度与基线相同。**逐字节相同的预测**是可接受标准;即使 <1% 的
  漂移也意味着接线 Bug。
- ✅ S1 延时回退 ≤ 3%。Heads 已分配但未被调用,唯一开销是
  `GenerationBatchResult` 构造 + 每步一次 Python 函数调用。

### 失败模式分诊表

| 现象 | 可能原因 | 对 Stage 3 的影响 |
|---|---|---|
| Server init 卡住 | `MedusaHeads` import 或权重分配 | 必须先修 Stage 2 |
| 精度等于基线但速度大幅退化 | 某处隐藏路径把 MEDUSA 当作 EAGLE/NGRAM 处理 | 复查 `is_*` 检查 |
| 精度不等于基线 | `forward_batch_generation` 包装时修改了 batch 状态 | 修 Stage 2 delegation |
| 启动崩,报 `KeyError: 'MEDUSA'` | server_args 调度回归 | 复核 CHANGE_0153 R1a |

## 7. 回退

- `prepare_env.sh` 中 `SOAR_SPEC_MEDUSA=0`(默认值)彻底关闭路径。
  Stage 2 worker 类只在选中 MEDUSA 时(经 `create_worker`)才会被导入。
- 源代码回退:`git revert <stage2-commit-sha>` 会恢复 R1a 的
  `NotImplementedError` stub,无依赖污染。

## 8. 移交 Stage 3

如果 fcloud 验证通过,Stage 3 将:

1. 设置 `mwb.capture_hidden_mode = CaptureHiddenMode.LAST`,在基线
   decode 之后读取 `logits_output.hidden_states`。
2. 运行 `self.medusa_heads(hidden)`,为每个请求产生 K 个 draft token。
3. 把 drafts 包装为 `NgramVerifyInput`(依 CHANGE_0154 §7 q2 的决策:
   **复用** 可省 ~200 LOC;K=1 chain 结构兼容)。
4. 通过 Stage 1 helpers 快照 SimpleGLA 状态。
5. 调用 `target_worker.forward_batch_generation(..., is_verify=True)`。
6. Verify、部分接受时恢复状态、重放已接受前缀。
7. 把 `is_medusa()` 接入 §4c 列出的六个调用点。

预计 Stage 3 改动量:`medusa_worker.py` 约 400 LOC,加上六处 1 行守卫。

## 9. 参考

- [CHANGE_0153_medusa_phase_r1_design.zh.md](CHANGE_0153_medusa_phase_r1_design.zh.md)
- [CHANGE_0154_medusa_phase_r1b_design.zh.md](CHANGE_0154_medusa_phase_r1b_design.zh.md)
- 参考实现:[ngram_worker.py L213](../../python/sglang/srt/speculative/ngram_worker.py#L213)(forward_batch_generation 模式)
