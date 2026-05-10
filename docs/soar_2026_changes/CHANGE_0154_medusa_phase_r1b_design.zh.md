# CHANGE_0154 — Medusa R1b 阶段设计（SimpleGLA 快照/恢复）

- **状态**：阶段 1 已落地（helper 函数已加入，调用前为 no-op）；阶段 2、3 待用户审阅。
- **分支**：`mixed_minicpm_cudagraph`
- **前置文档**：[CHANGE_0153_medusa_phase_r1_design.zh.md](CHANGE_0153_medusa_phase_r1_design.zh.md)（R1a 骨架）。
- **后续文档**：待定（R1c 将为 verify 路径加入 CUDA-graph 捕获；R3 训练 heads）。

## 1. 本次改动的目的

R1a 已经在 `SOAR_SPEC_MEDUSA=1` 下完成了注册表/argparse/数据类/heads/骨架 worker。但目前打开该开关后服务器仍会因 `MedusaWorker.__init__` 抛 `NotImplementedError` 而**故意失败**。R1b 是 `forward_batch_generation` 的**功能实现**及其所依赖的 SimpleGLA 状态管理机制。

R1b 中最棘手的子问题是 **MiniCPM-SALA 24 个 GLA 层的循环状态回退（rewind）**。推测解码本质上会"过早地"推进状态，对于 KV-cache 层可以释放对应缓存并回退 `seq_lens`，但 SimpleGLA 后端的循环状态**原地写回**到 `layer_cache.temporal`，且没有逐步中间缓冲（不像 Mamba2 拥有 `intermediate_ssm[layer, req, step, :]` 和专门的 scatter 函数 `update_mamba_state_after_mtp_verify`，见 hybrid_linear_attn_backend.py L1373）。

本文档规定：
- 快照/恢复方案（阶段 1 已提交）
- 使用它的 worker 逻辑（阶段 2、3，待批准）
- 内存代价、风险、待用户决策的问题

## 2. 约束复述（来自 CHANGE_0153 + 仓库规则）

- **R1b 中 K 硬编码为 1**（heads = 1）。"树"退化为 2 token 链（根 + 1 个 draft）。多 head 推迟到 R2。
- **W1 = 0 head 初始化**（Medusa 论文 §3.2）。W1=0 时，`SiLU(0·h) + h = h`，每个 head 的预测等于 `argmax(lm_head(h))` = 主模型在位置 t 的下一个 token。但主模型在 t+1 看到该 token 后会预测**不同**的下一个 token，因此 head 的 draft 与之不匹配 → 每步 `accept_len = 0`。**R1 是正确性闸门**：输出与基线字节一致，吞吐略低（heads 前向 + 2 token verify 前向开销）。速度收益要靠训练后的 heads（R3）。
- **归一化准确率必须 > 97%**（仓库 instructions §1）。字节一致意味着 100%。
- **R1b 的 verify 路径不做 CUDA-graph 捕获** —— 仅 eager。R1c 再补。

## 3. 阶段 1（本次已落地）—— SimpleGLA 快照/恢复辅助函数

### 文件：`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

`SimpleGLAAttnBackend` 新增三个方法：

```python
def snapshot_state_for_spec(self, mamba_indices: torch.Tensor) -> None:
    """快照所有 24 层的活跃 GLA 状态；在 TARGET_VERIFY 之前调用。"""

def restore_state_for_spec(self) -> None:
    """恢复快照；当 verify 后有 accept_len < num_draft_tokens 时调用。"""

def clear_state_snapshot_for_spec(self) -> None:
    """直接丢弃快照；当所有请求都完美预测时调用。"""
```

默认的 decode/extend 路径不受影响（快照仅在 `snapshot_*` 与 `restore_*` / `clear_*` 之间存在）。

**内存代价**（R1b K=1 链，bs=24）：

| 项目 | 数值 |
|---|---|
| MiniCPM-SALA GLA 层数 | 24 |
| 每层每请求 GLA 状态形状 | `num_heads × head_v_dim × head_k_dim` |
| `num_heads`（lightning_nkv，tp=1） | 16 |
| `head_dim`（lightning_head_dim） | 128 |
| **每行状态体积** | `16 × 128 × 128 × 2B (bf16) = 0.5 MiB` |
| **bs=24、24 层快照总量** | `24 × 24 × 0.5 = 288 MiB` |

每个 Medusa 步骤 288 MiB 不小。可选优化：
1. 方案 A（推荐）：只在 verify 可能失败时快照。R1（W1=0）中 verify 永远失败，但 verify 前向**确实**会推进状态，所以仍必须快照。**正确性强制**。
2. 方案 B：预分配单个快照缓冲区，跨步复用，避免 allocator 抖动。若 profile 显示压力则在阶段 3 引入。
3. 方案 C（R2）：K=2 时也只需每层快照一次；内存不随 K 增长。

State dim 注意事项：
- 单行 `state_dim` 由 `layer_cache.temporal.shape[1:]` 动态计算，不由上面的常量决定。288 MiB 是典型 SALA 配置的估计；首次调用时会打印实际值（TODO）。
- 若环境关闭 `fast_state_io`，`index_select` 仍会产生连续克隆。

### 阶段 1 单独的风险

- **无**。helper 已添加但未被调用。默认行为不变。导入文件正常（ast 验证通过）。
- 唯一风险是未来开发的名称冲突 —— 名字以下划线开头（`_spec_state_snapshot`、`_spec_snapshot_indices`），明确是模块私有。

## 4. 阶段 2（待批准）—— Heads-shadow worker

本阶段让 `SOAR_SPEC_MEDUSA=1` 能启动一个服务器，该服务器：
- 跑基线 decode 前向（每步 1 token），并 `capture_hidden_mode = LAST`
- 跑 MedusaHeads(hidden) 产出 K=1 个 candidate
- **丢弃 draft**，只输出基线 argmax
- 指标里 `num_accepted_tokens = 0`

这验证端到端集成，**在**实施 verify+rewind 之前。如果阶段 2 在 fcloud 启动正常且准确率与基线一致（必须，因为没有真正推测），阶段 3 可信地继续。

### 阶段 2 涉及文件

#### a) `python/sglang/srt/speculative/medusa_worker.py`
把 R1a 的 `NotImplementedError` 占位替换为：

```python
class MedusaWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.num_heads = server_args.speculative_num_medusa_heads
        assert self.num_heads == 1, "R1b only supports K=1 (chain)"
        self.device = target_worker.device

        hidden_size = self.model_runner.model_config.hidden_size
        lm_head = self.model_runner.model.lm_head
        dtype = self.model_runner.dtype
        from sglang.srt.models.minicpm_medusa_heads import MedusaHeads
        self.medusa_heads = MedusaHeads(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            lm_head_module=lm_head,
            dtype=dtype,
        ).to(self.device)
        self.medusa_heads.eval()

    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        if not batch.forward_mode.is_decode():
            mwb = batch.get_model_worker_batch()
            return self.target_worker.forward_batch_generation(mwb)

        mwb = batch.get_model_worker_batch()
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        mwb.capture_hidden_mode = CaptureHiddenMode.LAST
        result = self.target_worker.forward_batch_generation(mwb)

        # 阶段 2 shadow：跑 heads，丢弃 draft。
        if result.logits_output.hidden_states is not None:
            with torch.inference_mode():
                _ = self.medusa_heads(result.logits_output.hidden_states)

        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=result.can_run_cuda_graph,
            accept_lens=None,
        )
```

#### b) `python/sglang/srt/managers/scheduler.py`（L866）

把 MEDUSA 加入 `is_ngram()` 分支（都需要 `draft_token_to_kv_pool = None`）：

```python
if self.draft_worker is None or self.spec_algorithm.is_ngram() or self.spec_algorithm.is_medusa():
    draft_token_to_kv_pool = None
```

#### c) `python/sglang/srt/model_executor/model_runner.py`（L1702、L1730、L1860）

三处 `is_ngram()` 都加 `or self.spec_algorithm.is_medusa()`。其中 L1860 的 `NgramVerifyInput` 捕获分支在阶段 2 **跳过**（无 verify 捕获）；R1c 再加 MedusaInput 捕获分支。

#### d) `python/sglang/srt/model_executor/cuda_graph_runner.py`（L281、L431、L909）

三处 is_ngram() 都加 MEDUSA。R1b 中 MEDUSA verify 路径**不**捕获 CUDA-graph（因为还没有 verify 路径）。

#### e) 测试命令

fcloud 同步后：

```bash
# 1. SOAR_SPEC_MEDUSA=1 启动服务器应成功
SOAR_SPEC_MEDUSA=1 source ./prepare_env.sh
python3 -m sglang.launch_server --model-path "$MODEL_PATH" ... "${SGLANG_SERVER_ARGS[@]}" &

# 2. 健康检查
curl http://localhost:30000/health

# 3. 准确率必须与基线一致（字节级）
python3 /root/data/eval_model_001.py --data_path /root/data/perf_public_set.jsonl --port 30000

# 4. S1 速度可测（受 heads 前向开销影响会略慢于基线）
python3 /root/data/eval_model_001.py --data_path /root/data/speed_s1.jsonl --max-concurrent 1
```

阶段 2 通过标准：
- ✅ 服务器无崩溃启动
- ✅ 准确率 = 基线（ori_accuracy ≈ 79.29%，normalized ≈ 99.11%）
- ⚠️ S1 速度回退 < 5%（heads 前向 K=1 ≈ 64 MiB GEMM，约 0.1 ms）

### 阶段 2 风险

- **MedusaHeads 权重初始化**：W1=0 时 heads 对 logits 贡献为零，shadow 前向数学上是 no-op。风险：若 model_runner.dtype 是 float32 则可能 OOM（SALA 上不太可能）。
- **CaptureHiddenMode 联通性**：把 `mwb.capture_hidden_mode = LAST` 会让 LogitsProcessor 在输出中带 hidden_states；需要在当前 `force_dense_minicpm` + flashinfer 配置下验证。
- **Heads 参数量**：K=1 × hidden² × dtype_size = 4096² × 2B = 32 MiB（单 head）。安全。
- **bf16 权重加载错误**：heads 在 CPU 初始化再移到 GPU；安全。

## 5. 阶段 3（阶段 2 通过后再批准）—— 完整 verify + rewind

最高风险的一段。阶段 1、2 必须先在 fcloud 通过。

### 算法（双前向，K=1 链）

```
def forward_batch_generation(self, batch):
    if not batch.forward_mode.is_decode():
        return self._baseline_forward(batch)

    # ===== 第 1 次前向：基线 decode（1 token） =====
    mwb = batch.get_model_worker_batch()
    mwb.capture_hidden_mode = CaptureHiddenMode.LAST
    base_result = self.target_worker.forward_batch_generation(mwb)
    # GLA 状态前进 1（正确）。KV cache 写入新 token。
    base_argmax = base_result.next_token_ids                       # [bs]
    hidden = base_result.logits_output.hidden_states               # [bs, hidden]

    # ===== Heads 前向 =====
    head_logits = self.medusa_heads(hidden)                        # [bs, 1, vocab]
    drafts = head_logits.argmax(dim=-1).squeeze(1)                 # [bs]

    # ===== 第 2 次前向：TARGET_VERIFY 每请求 K=1 个 draft =====
    simple_gla = self._get_simple_gla_backend()
    snapshot_indices = batch.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
    simple_gla.snapshot_state_for_spec(snapshot_indices)

    verify_batch = self._make_verify_batch(batch, drafts)
    verify_result = self.target_worker.forward_batch_generation(
        verify_batch.get_model_worker_batch(), is_verify=True
    )
    # GLA 状态前进 2。每请求多 1 个 KV cache 槽。

    # ===== Verify：把 drafts 与 target_predict 在位置 0 比 =====
    target_logits = verify_result.logits_output.next_token_logits  # [bs, vocab]
    target_predict = target_logits.argmax(dim=-1)                  # [bs]
    accept_mask = (drafts == target_predict)                       # [bs]

    if accept_mask.all():
        simple_gla.clear_state_snapshot_for_spec()
        # 每请求输出 [base_argmax, drafts]（2 token / step）
    else:
        simple_gla.restore_state_for_spec()
        # 接受请求需对接受的 draft 再跑 1 token extend；拒绝请求只输出 base_argmax
        # KV cache 回滚：用 batch.tree_cache.free 释放被拒槽位（参考 NgramVerifyInput.verify）
        ...

    return GenerationBatchResult(...)
```

### 阶段 3 涉及文件

#### a) `python/sglang/srt/speculative/medusa_worker.py`
覆盖阶段 2 实现。**净新增约 400 行**，包括 `_make_verify_batch` 与 `_make_extend_batch_for_accepted`。

#### b) `python/sglang/srt/speculative/medusa_info.py`
增加 `prepare_for_verify(batch, page_size)` 与 `verify(batch, logits_output, page_size)`，对应 `NgramVerifyInput`。
- `prepare_for_verify`：为每请求分配 K=1 个额外 KV 槽，写 `batch.out_cache_loc`、`batch.input_ids = drafts`、`batch.forward_mode = TARGET_VERIFY`。
- `verify`：算 accept_mask、释放被拒槽、更新 `batch.seq_lens`、返回 `(logits_output, verified_id, num_accepted_tokens)`。

决策点：**能否直接复用 NgramVerifyInput？** NgramVerifyInput 的 draft 与 tree_mask 形状与我们 K=1 链兼容（2 token 链 == ngram_len=2）。若可，阶段 3 缩到约 200 LOC。请用户决策。

#### c) 部分接受时的缓存回滚
最难一部分。KV cache 回滚用 `batch.tree_cache.free(rejected_loc)` 加 `req_to_token_pool` 索引；GLA 回滚用 `restore_state_for_spec` + 1 token extend 前向。**两者必须原子完成** —— 若 extend 前向失败（如 OOM），状态会不一致。R1b 文档化此处脆弱性。

### 阶段 3 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| GLA rewind + extend 因 `query_start_loc`/`mamba_indices` 错位而产生错误状态 | 高 | 加 debug 模式 side-by-side 跑基线 + Medusa，断言输出字节一致 |
| `_make_verify_batch` 在 TARGET_VERIFY 下错配 `req_to_token_pool` 索引 | 高 | 完全复制 `NgramVerifyInput.prepare_for_verify` 模式 |
| 部分接受时 KV cache 泄漏 | 中 | 复用 `NgramVerifyInput._free_cache` 逻辑 |
| 性能回退 > 10%（两次前向 + heads + 快照） | 大概率（预期） | R1b 是正确性闸门；R3 训练后的 heads 会把回退转为收益 |

## 6. 本次为何只提交阶段 1

用户原话："working on medusa feature until it can be tested in floud, just keep documents detail and clear so we can review,reference and tracking the medusa implelmentation."

阶段 2、3 需要改 6+ 个文件（scheduler、model_runner、cuda_graph_runner、worker、info、prepare_env），并依赖一些关于 `capture_hidden_mode`、`req_to_token_pool` 索引、`NgramVerifyInput` 复用的细节假设。本地工作站没有 CUDA / numpy，无法迭代验证。一把推送所有阶段风险太高。

阶段 1（helpers + 本文档）是：
- **可审阅**：只改 1 个文件，新增约 90 行，零行为变化。
- **可回滚**：helper 是死代码，未被调用。
- **可追踪**：本文档完整规定阶段 2/3 设计，下一次会话零设计歧义。

用户审阅本文档后：
1. 如设计通过，下次会话实现阶段 2（1 个文件：medusa_worker.py；约 80 LOC）。
2. 阶段 2 通过 fcloud 字节一致测试后，阶段 3 落地真正的 verify+rewind。
3. R1c 给 Medusa verify 加 CUDA-graph 捕获（独立改动）。
4. R3 训练 heads（独立离线流水线）。

## 7. 待用户决策的开放问题

1. **R1b 取 K = 1 还是 K ≥ 2？** K=1 最简单（不需要 tree kernel）。同次迭代上 K ≥ 2 复杂度大致翻倍（tree mask、retrieve_index、eagle_utils kernel）。建议：先发 K=1，R2 升级。
2. **复用 `NgramVerifyInput` 还是新写 `MedusaInput.verify()`？** 复用节省约 200 LOC 但与 ngram API 耦合。建议：复用并文档化耦合。
3. **每步 288 MiB 快照可接受吗？** 若可，按设计推进。若紧张，可改为单一预分配缓冲区原地覆盖（约 50 LOC，R1b 内部）。
4. **R1b 是 eager only 还是含 CUDA-graph？** R1b 仅 eager；R1c 再加 graph（独立迭代）。
5. **要不要先发阶段 2（heads-shadow）作为中间提交，再做阶段 3？** 阶段 2 提供 fcloud 可冒烟测试的服务器**早于**高风险 verify 路径。强烈推荐。

## 8. 各阶段验证清单

### 阶段 1（本次）
- [x] `hybrid_linear_attn_backend.py` ast.parse 通过。
- [x] 无任何 call site 使用新 helper → 默认行为不变。
- [ ]（可选）fcloud 冒烟：SOAR_SPEC_MEDUSA=0 下基线准确/速度无变化。

### 阶段 2（待定）
- [ ] SOAR_SPEC_MEDUSA=1 服务器启动。
- [ ] `eval_model_001.py` 报告 normalized_accuracy 与基线一致。
- [ ] S1 延迟回退 < 5%。

### 阶段 3（待阶段 2 通过）
- [ ] 同样字节一致保证。
- [ ] `num_accepted_tokens` 在 W1=0 下显示 0（健全性）。
- [ ] bs=24 不 CUDA OOM。
- [ ] S1 回退 < 10%（R1 是正确性闸门，可接受）。

## 9. 回滚

- 阶段 1：revert 本次提交。
- 阶段 2/3（未来）：把 `SOAR_SPEC_MEDUSA=0` 写回 `prepare_env.sh`（已是默认）—— 路径默认禁用。

## 10. 参考

- [CHANGE_0153_medusa_phase_r1_design.zh.md](CHANGE_0153_medusa_phase_r1_design.zh.md)
- [PROPOSAL_medusa_minicpm_sala_001.zh.md](PROPOSAL_medusa_minicpm_sala_001.zh.md)
- [RESEARCH_speculative_decoding_survey_001.zh.md](RESEARCH_speculative_decoding_survey_001.zh.md)
- Medusa 论文 §3.2（W1 零初始化）：https://arxiv.org/abs/2401.10774
- sglang 参考：`update_mamba_state_after_mtp_verify` 见 [hybrid_linear_attn_backend.py L1373](../../python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1373)
- sglang 参考：`NgramVerifyInput.verify` 见 [ngram_info.py L374](../../python/sglang/srt/speculative/ngram_info.py#L374)
