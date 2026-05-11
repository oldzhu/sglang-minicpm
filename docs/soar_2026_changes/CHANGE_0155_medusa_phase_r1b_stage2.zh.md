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

## 10. Stage 2 fcloud 启动补丁（2026-05-11）

**首次以 `SOAR_SPEC_MEDUSA=1` 启动服务器**确认 worker 初始化成功：

```
[2026-05-11 00:41:32] MedusaWorker Stage 2 ready: K=1, hidden=4096,
  dtype=torch.bfloat16, device=cuda:0, approx_weight_MiB=32.0
```

但 **第一个 prefill 请求崩溃**：

```
File .../medusa_worker.py L150 ... forward_batch_generation
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
File .../model_runner.py L2251 ... _forward_raw
    ret = self.graph_runner.replay(...)
File .../input_buffers.py L156 ... populate_from_forward_batch
    self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
RuntimeError: output with shape [1] doesn't match the broadcast shape [7]
```

**根因。** sglang 内部两段逻辑在 "已注册 MEDUSA 但没有真实 draft" 时相互矛盾：

1. [cuda_graph_runner.py L281](../../python/sglang/srt/model_executor/cuda_graph_runner.py#L281)
   只对 EAGLE / STANDALONE / NGRAM 把 `num_tokens_per_bs` 设为
   `speculative_num_draft_tokens`，MEDUSA 走 default 分支（保持 = 1），
   所以 16 张图（捕获耗时 936 s）都按普通 decode shape 录制。
2. 但 `--speculative-algorithm MEDUSA` 已被设置，[scheduler.py L2225](../../python/sglang/srt/managers/scheduler.py#L2225)
   走 spec-v1 分支，把 `ScheduleBatch` 直接传给
   `model_worker.forward_batch_generation`。叠加 `--enable-torch-compile` 后，
   prefill 请求被 dispatch 进 `graph_runner.replay`，而 `raw_num_token = 1 * 1 = 1`
   无法 broadcast 7 个 token 的 prefill `input_ids`。

**修复（已选定路径）。** CHANGE_0154 §2 已明确 **R1b 全程纯 eager**，verify 路径
的 CUDA graph capture 推迟到 R1c。因此只需在 `prepare_env.sh` 里：
当 `SOAR_SPEC_MEDUSA=1` 时剥离 `--enable-torch-compile` /
`--torch-compile-max-bs N`，并追加 `--disable-cuda-graph`：

```bash
if [[ "$SOAR_SPEC_MEDUSA" == "1" ... ]]; then
    NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))
    export SGLANG_SERVER_ARGS="... --speculative-algorithm MEDUSA \
        --speculative-num-medusa-heads ${SOAR_SPEC_MEDUSA_HEADS} \
        --speculative-num-draft-tokens ${NUM_DRAFT_TOKENS}"
    # Stage 2/3 全程 eager；图捕获在 R1c 处理。
    export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--enable-torch-compile/}"
    export SGLANG_SERVER_ARGS="$(echo "$SGLANG_SERVER_ARGS" | sed -E 's/--torch-compile-max-bs [0-9]+//g')"
    export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --disable-cuda-graph"
fi
```

**对基准测试的影响。** 关闭 CUDA graph + torch.compile 后，Stage 2 比 v22 baseline
会明显变慢（粗估 S1/S8/Smax 各退化 1.3–2 倍）。**Stage 2 接受这一退化**：
本阶段目标是字节级精度对齐 + dispatch 路径打通；速度退化将在 **R1c**（在
`cuda_graph_runner.py` 中为 MEDUSA 增加 TARGET_VERIFY graph capture 分支）后消除。

**重测预期。** 精度：与 v22 baseline 字节级一致。速度：明显退化（eager 模式），
仅记录，不作为通过门槛。

**本次改动文件（delta）。**
- [benchmark/soar/demo_sala/prepare_env.sh](../../benchmark/soar/demo_sala/prepare_env.sh)
  — 在 MEDUSA 分支剥离 torch-compile 并追加 `--disable-cuda-graph`。

## 11. Stage 2 fcloud 启动补丁 #2 — decode-prep bug（2026-05-11）

**eager 模式重测**越过了 cuda-graph 崩溃：**第一个 prefill 请求成功**
（Marlin GEMM 在 M=7 跑通）。但**第二次 forward 调用**（第一个 decode step）
立即崩溃：

```
RuntimeError: Number of tokens in position_ids must match QKV
```

位于 `fused_qk_norm_rope`，M=7（即 decode batch 仍带着 prefill 阶段的 7 个 input_ids）。

**根因。** [schedule_batch.py L1948](../../python/sglang/srt/managers/schedule_batch.py#L1948)：

```python
def prepare_for_decode(self):
    self.forward_mode = ForwardMode.DECODE
    ...
    if not self.spec_algorithm.is_none():
        # 投机解码下，decode batch 由 spec worker 自己准备
        return   # ← 提前返回
```

scheduler 在两次迭代之间会调用 `batch.prepare_for_decode()`。当
`spec_algorithm != NONE` 时，它**只翻转 forward_mode 到 DECODE，但跳过所有
字段更新**（`input_ids = output_ids`、`seq_lens += 1`、`alloc_for_decode` 等），
因为 EAGLE/NGRAM worker 自己会在 `_prepare_for_speculative_decoding` →
`prepare_for_verify` 内做这些。

Stage 2 MedusaWorker 是**纯透传**，没有 `_prepare_for_speculative_decoding`。
于是 decode batch 到达我们 worker 时，`forward_mode = DECODE` 但 `input_ids`
仍保留 prefill 的 [7 tokens]。下游 `_forward_raw` 调度到 `forward_decode`，
但 `positions = clamp_position(seq_lens=[7])` 长度只有 1，与 input_ids 长度 7
不匹配 → fused_qk_norm_rope 断言失败。

**修复。** 在 `MedusaWorker.forward_batch_generation` 中，当 batch 进入 DECODE
模式时，临时把 `batch.spec_algorithm` 设为 NONE 并再次调用 `prepare_for_decode()`
来补完被跳过的准备工作，随后恢复原 spec_algorithm：

```python
if batch.forward_mode.is_decode():
    saved_spec_algo = batch.spec_algorithm
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    try:
        batch.prepare_for_decode()
    finally:
        batch.spec_algorithm = saved_spec_algo
```

证据是临时调试日志（commit `db839a4b1`，修复 commit 中已移除）抓到的两条
打印：

```
[MedusaWorker.fbg] batch.forward_mode=EXTEND ... input_ids_len=7   ← prefill 正常
[MedusaWorker.fbg] batch.forward_mode=DECODE ... input_ids_len=7   ← 字段陈旧，崩溃
```

**对 Stage 3 安全。** Stage 3 实现完整的 verify+rewind 后，
`_prepare_for_speculative_decoding` 会在 dispatch 前把 `forward_mode` 翻转
到 `TARGET_VERIFY`。本补丁中的 `is_decode()` 分支自然会跳过，因为那时
`forward_mode == TARGET_VERIFY`。所以 Stage 2 的这个补丁不会阻塞 Stage 3。

**本次改动文件（delta #2）。**
- [python/sglang/srt/speculative/medusa_worker.py](../../python/sglang/srt/speculative/medusa_worker.py)
  — 当 batch 处于 decode 模式时，补完 scheduler 跳过的 `prepare_for_decode`。

## 9. 参考

- [CHANGE_0153_medusa_phase_r1_design.zh.md](CHANGE_0153_medusa_phase_r1_design.zh.md)
- [CHANGE_0154_medusa_phase_r1b_design.zh.md](CHANGE_0154_medusa_phase_r1b_design.zh.md)
- 参考实现:[ngram_worker.py L213](../../python/sglang/srt/speculative/ngram_worker.py#L213)(forward_batch_generation 模式)
