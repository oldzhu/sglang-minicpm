# CHANGE_0153 — Medusa Phase R1 详细设计（plumbing spike）

状态：**设计稿 — 改代码前需批准**
日期：2026-05-10
分支：`mixed_minicpm_cudagraph`
父文档：[PROPOSAL_medusa_minicpm_sala_001.zh.md](PROPOSAL_medusa_minicpm_sala_001.zh.md)
配套：[RESEARCH_speculative_decoding_survey_001.zh.md](RESEARCH_speculative_decoding_survey_001.zh.md)

## 0. TL;DR

通读 sglang 推测解码子系统后，两个发现**显著降低 R1 工程量**：

1. **"GLA-fork 问题"在 sglang 上游对 Mamba2/GDN 已经解决。** `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify`（hybrid_linear_attn_backend.py L1373–1440）在 verify 时把每步 state 写入 `mamba_pool.intermediate_ssm[layer, request, step]`；`verify_tree_greedy` 返回每个请求的 accepted-prefix 长度后，再 scatter 对应 step 的 state 回到持久化的 `ssm_states`。我们要做的是**把这个机制移植到 `SimpleGLAAttnBackend`**——不是从零设计。原提案 R2 中最大的设计风险因此消除。
2. **Tree-verify 基础设施完全可复用。** `eagle_utils.build_tree_kernel_efficient` + `verify_tree_greedy_func` 与 draft 来源无关。Medusa 的树（一根、K 层、每层 top-s）是 EAGLE 树的一种特例。

净效应：R1 从原本的 1500–2000 LOC 缩到 ~600–800 LOC，工程焦点收敛到一个新机制——**`SimpleGLAAttnBackend.update_simple_gla_state_after_verify`**——加少量胶水。

## 1. 架构（R1）

```
                  ┌─────────────────────────────────────┐
                  │ MedusaWorker（继承 TpModelWorker） │
                  └──────────────────┬──────────────────┘
                                     │ forward_batch_generation()
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ 1. 主模型 fwd → hidden + base logits│
                  │ 2. K 个 Medusa head → K 个候选集    │
                  │ 3. build_tree_kernel_efficient      │
                  │    （复用 eagle_utils）             │
                  │ 4. 主模型 verify-fwd 走树           │
                  │ 5. verify_tree_greedy → 接受长度 L  │
                  │ 6. update_simple_gla_state_after... │
                  │    （新：scatter accepted state）   │
                  └─────────────────────────────────────┘
```

R1 故意**取 K=1, top_s=1, num_draft_tokens=2**（1 base + 1 draft）→ 树退化成链。这把 state-fork 机制从 tree-verify 边界条件中隔离出来。R2 会在 R1 byte-identity 通过后扩 K 与 top_s。

## 2. 新增 / 修改文件（具体 diff 计划）

### 2.1 新文件

```
python/sglang/srt/speculative/medusa_info.py                   ~150 LOC
python/sglang/srt/speculative/medusa_worker.py                 ~350 LOC
python/sglang/srt/models/minicpm_medusa_heads.py               ~120 LOC
```

**`medusa_info.py`** — 数据结构：
```python
class MedusaInput(SpecInput):
    """K 个 Medusa head 的输出，准备进 tree-verify。"""
    spec_input_type = SpecInputType.MEDUSA_VERIFY  # 新枚举值

    draft_token_ids: torch.Tensor      # (bs, num_draft_tokens)
    parent_index: torch.Tensor          # (bs, num_draft_tokens) — 父节点
    retrieve_index: torch.Tensor        # (bs, num_draft_tokens) — flatten 索引
    tree_mask: torch.Tensor             # (bs, num_draft_tokens, num_draft_tokens)
    positions: torch.Tensor             # (bs, num_draft_tokens) — 绝对位置
    accept_threshold: float = 1.0       # R1 默认 = byte-identity gate

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (self.draft_token_ids.shape[1], 1)


@dataclass
class MedusaVerifyOutput:
    verified_id: torch.Tensor                    # (sum_accepted_per_req,)
    accept_length_per_req_cpu: List[int]
    last_hidden_state: torch.Tensor              # (bs, hidden) 给下一步 head 用
```

**`medusa_worker.py`** — 每步生成的控制流：
```python
class MedusaWorker(TpModelWorker):
    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        # 1. 主模型在当前输入上 forward
        logits, hidden = self.target_worker.forward_with_hidden(batch)
        # 2. K 个 head → 每个 head 取 top-s
        head_logits = self.heads(hidden[:, -1])     # (bs, K, vocab)
        # 3. 用 eagle_utils 建树（K=1 时是链）
        spec = self._build_tree(head_logits, hidden)
        # 4. 主模型 verify forward（带 tree mask 与 positions）
        verify_logits, verify_hidden = self.target_worker.verify(batch, spec)
        # 5. 贪心 verify（R1：accept_threshold=1.0 byte-identity）
        verified_id, accept_len = verify_tree_greedy_func(...)
        # 6. 把接受前缀的 GLA state 回写到持久化 cache
        self.attn_backend.update_simple_gla_state_after_verify(
            accepted_steps=accept_len, ...
        )
        return GenerationBatchResult(next_token_ids=verified_id, ...)
```

**`minicpm_medusa_heads.py`** — head 模块：
```python
class MedusaHeads(nn.Module):
    """K 个残差 MLP，分类器复用主模型 lm_head。

    每个 head：  p^(k) = softmax( lm_head( SiLU(W1^k · h) + h ) )
    W1 零初始化（论文 §3.2）后随机初始化下每个 head 等价于 base 模型，
    accept_rate = 1.0 平凡成立 → R1 byte-identity 闸自动通过。
    """
    def __init__(self, hidden_size, num_heads, lm_head_module):
        super().__init__()
        self.num_heads = num_heads
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(num_heads)
        ])
        self.lm_head = lm_head_module       # 共享权重，不单独存

    def forward(self, h):                   # h: (B, hidden)
        outs = []
        for k in range(self.num_heads):
            outs.append(self.lm_head(F.silu(h @ self.W1[k]) + h))
        return torch.stack(outs, dim=1)     # (B, K, vocab)
```

**为什么 R1 平凡正确：** `W1=0` 时 `SiLU(0·h) + h = h`，所以每个 head 输出严格等于 `lm_head(h)`——与 base 模型 argmax 完全一致。`accept_threshold=1.0` 下 verifier 当且仅当 draft == base argmax 时接受 → **总是接受**。这就是 R1 正确性立足的 byte-identity 闸。

### 2.2 修改文件

| 文件 | 改动 | LOC |
|---|---|---|
| `python/sglang/srt/speculative/spec_info.py` | `SpeculativeAlgorithm` 加 `MEDUSA = auto()`；`is_medusa()`；`create_worker()` 新分支；`SpecInputType` 加 `MEDUSA_VERIFY` | +20 |
| `python/sglang/srt/server_args.py` | 接受 `--speculative-algorithm MEDUSA`；`--speculative-num-medusa-heads K`（默认 1） | +15 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | `SimpleGLAAttnBackend` 加 `update_simple_gla_state_after_verify`（镜像 `update_mamba_state_after_mtp_verify`）；为 SimpleGLA 在 `init_cuda_graph_state` 里初始化 `intermediate_ssm` | +110 |
| `python/sglang/srt/models/minicpm.py` | `SOAR_SPEC_MEDUSA=1` 时在 `MiniCPMSALAForCausalLM` 上挂 `MedusaHeads`；暴露 `forward_with_hidden` 返回 pre-norm last hidden | +40 |
| `python/sglang/srt/managers/tp_worker.py` | 把 `forward_with_hidden` 通到 worker 层 | +15 |
| `benchmark/soar/demo_sala/prepare_env.sh` | opt-in env：`SOAR_SPEC_MEDUSA=0`（默认），为 1 时追加 `--speculative-algorithm MEDUSA --speculative-num-draft-tokens 2 --speculative-num-medusa-heads 1` | +15 |
| `benchmark/soar/demo_sala/preprocess_model.py` | 模型目录中存在 head 权重文件时拷到输出目录；否则 no-op | +20 |

**合计**：~635 LOC，新 3 文件 + 改 7 文件。

## 3. State-scatter 机制（R1 核心技术工作）

### 3.1 已有参考：`update_mamba_state_after_mtp_verify`（Mamba2/GDN）

每层：
- **verify forward** 时 backend 把每步 state 写到 `intermediate_ssm[layer, request, step]`，shape `(num_layers, max_requests, max_draft_tokens, state_dim)`。
- `verify_tree_greedy` 产 `accepted_steps[req] ∈ {-1, 0, 1, ..., K-1}` 后：
  - `accepted_steps[req] >= 0` 的请求：scatter `intermediate_ssm[:, req, accepted_steps[req]]` → `ssm_states[:, req_pool_idx[req]]`。
  - 全拒（steps == -1）的请求保持原 state（draft 之前那个）——天然正确，因为我们从未覆盖它。

### 3.2 R1 移植：`update_simple_gla_state_after_verify`

```python
class SimpleGLAAttnBackend(MambaAttnBackendBase):
    def update_simple_gla_state_after_verify(
        self,
        accepted_steps: torch.Tensor,      # (bs,) int，-1 表示无接受
        layer_id: int,
    ):
        """verify 后把接受前缀对应的 GLA state 写回 temporal cache。"""
        request_number = accepted_steps.shape[0]
        valid = accepted_steps >= 0
        dst = self.forward_metadata.mamba_cache_indices[:request_number][valid]
        src = torch.arange(request_number, device=dst.device)[valid]
        steps = accepted_steps[valid].to(torch.int64)

        layer_cache = self._get_layer_cache(layer_id)
        # `intermediate_ssm` 是 layer-private，索引 [request, step, ...]
        layer_cache.temporal[dst.to(torch.int64)] = (
            layer_cache.intermediate_ssm[src, steps].to(layer_cache.temporal.dtype)
        )
```

`forward()` 中：
```python
if forward_batch.spec_input is not None and forward_batch.spec_input.is_medusa_verify():
    # 不写 final_state 到 layer_cache.temporal；改写 per-step state 到 intermediate_ssm
    layer_cache.intermediate_ssm[:request_number, step_idx] = final_state
else:
    self._store_final_state(layer_cache, mamba_indices, final_state)
```

### 3.3 `intermediate_ssm` 分配

给 `MambaPool` 加 SimpleGLA 分支。Buffer shape：`(num_simple_gla_layers, max_speculative_bs, max_draft_tokens, num_heads, head_dim_qk, head_dim_v)`。SALA `lightning_nkv=16`、head_dim=128（qk 与 v 同维）、max_bs=24、K=2 → ~24·24·2·16·128·128 ≈ 600 MB BF16。**太大。**

缓解：R1 K=1，`max_draft_tokens=2` → 节省 50% → ~300 MB。仍偏高，预算可容但需测。R2 会量化中间 buffer 或并入 `ssm_states` 多预留几个 slot。

行动项：spike 前先核对该 buffer 与 `mem-fraction-static=0.84` 余量。

## 4. 正确性闸（R1 退出条件）

`SOAR_SPEC_MEDUSA=1` 且 `accept_threshold=1.0` 启动 server：

1. 发 50 prompts（每类任务 10 条：mcq/qa/niah/cwe/fwe）。
2. 每 prompt 取 spec 模式下的 `predictions.jsonl`。
3. 与 `SOAR_SPEC_MEDUSA=0` 同 seed、同 temp 的 `predictions.jsonl` 逐 token 对比。
4. **通过条件**：50 prompts 全部 token 序列 byte-identical。

不通过：GLA state-scatter 有 bug。R1 停，进调查。

## 5. 测试命令

本地语法 / import：
```bash
cd /home/oldzhu/sglang/python
python -c "from sglang.srt.speculative.medusa_worker import MedusaWorker"
python -c "from sglang.srt.speculative.spec_info import SpeculativeAlgorithm; assert SpeculativeAlgorithm.from_string('MEDUSA').is_medusa()"
```

fcloud spike（需要 fcloud 恢复）：
```bash
cd /root/submission_sim
SOAR_SPEC_MEDUSA=1 source prepare_env.sh
python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" "${SGLANG_SERVER_ARGS[@]}" &
# 等健康
curl -s -X POST http://localhost:30000/generate \
  -d '{"text": "Hello world", "sampling_params": {"temperature": 0, "max_new_tokens": 32}}'
# 期望：与 SOAR_SPEC_MEDUSA=0 baseline 输出完全一致（byte-identity 闸）
```

单 prompt 烟囱通过后：
```bash
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --env SOAR_SPEC_MEDUSA=1 \
  --speed-cap 5     # 快速子集；全量留 R2
```

## 6. R1 特有风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| `intermediate_ssm` 吃显存太多 | 中 | R1 限 max_speculative_bs；放量前测 |
| `verify_tree_greedy_func` 期望 EAGLE 数据 layout 与 Medusa 链 K=1 不同 | 低 | 显式构链（parent_index = [-1, 0]）——sglang NEXTN 已支持 |
| 给 head 的 hidden 是 post-norm 而非 pre-norm（或反） → head 看到"错"的 h | 中 | W1=0 时无所谓——pre/post-norm 都塌缩到 lm_head(h)。R3 head 训练时才有意义。R1 用注释固定约定 |
| `--enable-torch-compile` 给 verify 与 decode 编出不同图 | 中 | R1 在 `SOAR_SPEC_MEDUSA=1` 时关 torch.compile。R3 R2 稳定后再开 |

## 7. 回滚

`SOAR_SPEC_MEDUSA=0` 默认 → 新代码路径都不走。最坏 = `git revert <commit>`。

## 8. 决策点（请你定）

**Q1.** 是直接进 R1 编码，还是先复审本设计？
- (A) 直接进：下 1–2 个回合写 ~635 LOC、铺到 10 个文件；fcloud 恢复后测。
- (B) 先复审：我停在这里，你读完再继续。

**Q2.** R1 默认 `K`（Medusa head 数）：
- (A) `K=1`（链，爆炸半径最小——推荐）
- (B) `K=2`（一上来就是树——代码与风险更大）

**Q3.** 是否同时给 `prepare_env.sh` 加 `SOAR_SPEC_NGRAM=1` 的 opt-in 作为"免费保险"，与 Medusa 并行测？纯 server-arg 改动，零风险。详见 [RESEARCH_speculative_decoding_survey_001.zh.md](RESEARCH_speculative_decoding_survey_001.zh.md) §5.2。
- (Y) 是
- (N) 推迟

## 9. 参考

- 冠军原文：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- State-scatter 参考：`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` L1373–1440（`update_mamba_state_after_mtp_verify`）
- Tree build/verify 参考：`python/sglang/srt/speculative/eagle_utils.py` L41–199（`build_tree_kernel_efficient`、`verify_tree_greedy_func`）
- Spec worker 模式参考：`python/sglang/srt/speculative/eagle_worker.py` L79–328
