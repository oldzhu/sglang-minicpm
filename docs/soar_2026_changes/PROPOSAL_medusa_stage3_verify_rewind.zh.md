# PROPOSAL —— Medusa Phase R1b Stage 3：真 verify + rewind

**状态**：提案，本轮 session 内自验证。
**分支**：`mixed_minicpm_cudagraph`。
**前置文档**：[CHANGE_0153 (R1 设计)](CHANGE_0153_medusa_phase_r1_design.zh.md)、[CHANGE_0154 (R1a 脚手架)](CHANGE_0154_medusa_phase_r1a_scaffolding.zh.md)、[CHANGE_0155 (R1b Stage 2 pass-through)](CHANGE_0155_medusa_phase_r1b_stage2.zh.md)。

## 1. 背景与动机

Stage 2 完成后（commit `46553947b` → `1e5cd15a8` → `469e5815f`），`MedusaWorker` 是一个纯 pass-through：worker 接好了，heads 已分配，scheduler dispatch 走通，运行时与 v22 基线一致（acc 80.11%，S1 118.28s ≈ Test 12）。submission tarball `minicpm_sala_submit_v23.tar.gz` 用于在官方硬件上验证这一点。

Stage 3 的任务是：让 worker 真的产 + 验 draft token，但保持 zero-init head 的字节同构性 —— 在 K=1、`W1=0` 时，每个 draft token 都等于 target 的 argmax，verify 永远 accept，模型输出与基线**字节一致**。

## 2. 合规声明

- 提交大小不变（heads 已在 Stage 2 计入；本轮只改 host 代码）。
- 现场量化不变。
- zero-init heads 下输出字节一致 → 归一化 accuracy ≥ 基线（预期 ≥ 99% → C=1.0）。
- 不依赖任何禁用技巧 —— 同一 eval harness、同一 `--max-concurrent`、不重新打开 prefix-cache。

## 3. 目标与上限估算

| 阶段 | 范围 | 期望精度 | 期望速度 |
|------|------|----------|----------|
| **3a**（本提案、eager） | heads + verify 接通；verify 路径关 cuda-graph | 字节一致（acc ≈ 80.11%，C=1.0） | 大概率**比基线慢**（eager + verify 开销）—— 只验证正确性 |
| **3b**（后续、cuda-graph） | 同样逻辑、cuda-graph 捕获 TARGET_VERIFY | 精度不变 | **乐观估计 S1 −30~40%**（100% accept 时一次 forward 出 2 个 token，扣除 head 与 verify 开销）。**现阶段实际收益更可能 −10~20%**，因为 head 未训练。 |

**理论上限**：K=1、100% accept → ~1.7-1.8× decode 提速。**当前阶段实际上限**：受 verify 开销和 head forward 成本约束。真正提交可见的收益要等 head 训练（CHANGE_0154 §6，延后）。

## 4. 可复用的现有基础设施

`ngram_worker.py` + `ngram_info.py` 在 **完全相同的 server 配置**（`--attention-backend minicpm_flashinfer`）上已实现 K 个线性 draft token 的整套 verify 管道：

| 组件 | NGRAM 文件 | 我们复用？ |
|------|-----------|-----------|
| `NgramVerifyInput.prepare_for_verify`（verify KV slot 分配，`out_cache_loc`，`seq_lens` 扩展） | `ngram_info.py:74` | 是，逻辑通用 —— 直接复用或包一层。 |
| `NgramVerifyInput.verify`（贪心 accept walk、KV rewind、`next_token_ids` 提取） | `ngram_info.py:374` | 是。 |
| `generate_attn_arg_prefill`（flashinfer KV-indices、verify custom_mask） | `ngram_info.py:124` | 是。 |
| `minicpm_backend.py:521` 的 `is_target_verify()` 分支 | 已存在 | 是，证明 backend 已经兼容。 |
| `cuda_graph_runner.py:286` TARGET_VERIFY 捕获路径 | 已为 eagle 存在 | 仅 Stage 3b 复用。 |

**结论**：不重发明 verify；把 Medusa head forward 插入到 NGRAM 做 cache lookup 的位置即可。

## 5. 实现计划（Stage 3a）

### 5.1 Hidden state 捕获

Medusa heads 需要**上一步的 last hidden state** 作为输入，才能产下一步 draft。所以 verify 跑之前必须：(a) 在上一次 forward 设置 `spec_info.capture_hidden_mode = CaptureHiddenMode.LAST`；(b) 把返回的 `hidden_states` 回传到 worker。

序列**首次** decode（extend 后第一次）没有上一步的 hidden，所以这一步走 pass-through（无 draft、无 verify），从第 2 步开始才 spec。

### 5.2 Draft 生成

```python
# MedusaWorker 内：
def _generate_draft(self, prev_hidden: torch.Tensor) -> torch.Tensor:
    # prev_hidden: (bs, hidden_size) —— 每个 seq 上一步的 last hidden
    # MedusaHeads.forward 返回 (bs, K, vocab)
    logits = self.medusa_heads(prev_hidden)
    draft_tokens = logits.argmax(dim=-1)  # (bs, K)
    return draft_tokens  # int64
```

### 5.3 Verify input 构造

K=1 时极简：每条线就是上一刚采样的 target token 后挂一个 draft token。

- `draft_token` = `[t_prev_target, d_1]` per seq（长 2）
- `tree_mask` = 2-token 链的恒等因果掩码
- `positions` = `[seq_len-1, seq_len]`
- 喂给 `NgramVerifyInput`（类型 tag 无所谓 —— 复用 `SpecInputType.NGRAM_VERIFY` 即可；如果下游对类型敏感再加一个 `MedusaVerifyInput` 子类）。

### 5.4 Worker forward 流（Stage 3a）

```
def forward_batch_generation(batch):
    if batch.forward_mode.is_extend():
        # 首轮 prefill —— pass-through，但打开 LAST capture，让第 2 步能拿到 hidden
        ... target_worker.forward_batch_generation(batch) with capture_hidden_mode=LAST
        save self.last_hidden[req_id] = h_per_seq
        return result (no spec)

    if batch.forward_mode.is_decode():
        if 本 batch 内任一 req 没有缓存 hidden:
            # 冷启动 —— pass-through，同时为下一步捕获 hidden
            ... pass-through with LAST capture
            return result

        else:
            # 热路径 —— draft、verify、rewind
            drafts = self._generate_draft(self.last_hidden_for_batch(batch))
            self._build_verify_input(batch, drafts)
            batch.forward_mode = TARGET_VERIFY
            batch.spec_info.capture_hidden_mode = LAST  # 给下一步用
            res = target_worker.forward_batch_generation(batch_mwb, is_verify=True)
            logits_output, next_token_ids, num_accepted = verify_input.verify(...)
            # 从 logits_output.hidden_states 取被 accept 索引的 hidden 更新 self.last_hidden
            return GenerationBatchResult(...)
```

### 5.5 正确性校验（zero-init 不变量）

当 `W1 = 0` 时：
- `MedusaHead.forward(h) = SiLU(0) + h = h`
- `MedusaHeads.forward(h)[k] = lm_head(h)` = target lm_head 在 `h` 上的同样 logits
- `argmax(MedusaHeads(h)) = argmax(target_lm_head(h)) = t_prev_target`（即下一 token argmax）

所以 `d_1 = step_N 的 t_prev_target_argmax == 位置 N 处 target_argmax` —— verify 永远 accept，`next_token_ids` 与 pass-through 一致。**输出字节同构。**

## 6. 风险评估

| 风险 | 概率 | 缓解 |
|------|------|------|
| 在 eager 模式下 hidden state 捕获默默失效 | 低 | NGRAM 的 `is_verify=True` 路径本来就开 hidden 捕获，我们照抄。 |
| 复用 `NgramVerifyInput` 引起下游类型检查失败 | 低 | `SpecInputType.NGRAM_VERIFY` 在下游是 duck-typed；坏了就加 `MedusaVerifyInput` 子类。 |
| `minicpm_flashinfer` 的 TARGET_VERIFY 分支在 hybrid GLA 层上跑不通 | 中 | NGRAM 历史上在同一模型可用 —— 但没在 FP8 KV + dense 路径上跑过。**缓解**：Stage 3a 走 eager 模式（CHANGE_0155 §14 加的 `SOAR_SPEC_MEDUSA_EAGER=1` 开关已经支持）。 |
| KV rewind 与 SimpleGLA recurrent state 冲突 | 中-高 | CHANGE_0153 §3 已经预警过。Stage 3a 因为永远 accept，**rewind 实际不触发**，规避了这一问题。后续训完 head 的 3b/3c 才需要真验证。 |
| `minicpm_flashinfer` 上 TARGET_VERIFY 的 cuda-graph 捕获坏 | 中 | 推到 Stage 3b；Stage 3a 一律 eager。 |

## 7. 验证计划

### Stage 3a sanity（local）
1. Server 用 `SOAR_SPEC_MEDUSA=1 SOAR_SPEC_MEDUSA_EAGER=1` 启动。
2. Health check 过。
3. 3-sample accuracy smoke test → 答案与 v22 基线字节一致（zero-init heads + 贪心解码）。

### Stage 3a fcloud
1. `sync` → `restart-server` → `wait-server` → `accuracy` 全量。
2. **通过条件**：`ori_accuracy ≥ 80.0%`（落在 Stage 2 cuda-graph 80.11% 的本地噪声带内）。
3. 若通过，加跑一次 S1 quick（S8/Smax 不必，因为 eager 模式下预期更慢）。

### Stage 3b（3a 通过后下一轮）
1. 关掉 EAGER 开关 → cuda-graph 重启。
2. 验证 cuda-graph 能正确捕获 TARGET_VERIFY 形状。
3. accuracy + S1/S8/Smax 完整跑。

## 8. 修改文件

| 文件 | 修改 |
|------|------|
| `python/sglang/srt/speculative/medusa_worker.py` | Stage 2 pass-through 替换为 Stage 3a 的 draft+verify。增加 hidden-state cache。 |
| `python/sglang/srt/models/minicpm_medusa_heads.py` | 不动（forward 已正确）。 |
| `benchmark/soar/demo_sala/prepare_env.sh` | 暂不动 —— Stage 3a 通过 `SOAR_SPEC_MEDUSA_EAGER=1` 手动 export 切换。3b 通过后再翻回 cuda-graph 默认。 |
| `docs/soar_2026_changes/CHANGE_0156_medusa_phase_r1b_stage3.{en,zh}.md` | 进 Stage 3 时新建（本提案 §1-7 作为其 §1-7）。 |

## 9. 回滚

- Stage 3a：`SOAR_SPEC_MEDUSA=0` → MedusaWorker 不实例化 → 行为 = v22 基线。永远安全。
- Worker 内部：若 3a 逻辑本身坏掉，把 `medusa_worker.py` 回到 `46553947b`（Stage 2 pass-through）。

## 10. 3a 通过后的下一步

若 3a 通过（正确性验证完成）：
- **Stage 3b**：为 TARGET_VERIFY + DECODE 重启 cuda-graph；测量真正的速度收益上限。
- **Stage 3c**（延后，可能要模型重训）：用 `lm_head` 初始化 heads（CHANGE_0154 §6），或在 eval 分布上训 heads —— 让接受率突破"100% on zero-init"这种平凡情况（只有 head 等价于 target 时才成立）。

## 11. 后端探路勘误（2026-05-11）

Stage 3a 编码中我读到 `minicpm_backend.py:521` 的 `NotImplementedError` 即下结论 Stage 3 被阻。**该结论错误。** 用户指出 `prepare_env.sh` 第 199 行 `SOAR_BACKEND_VARIANT=flashinfer` 默认会把 launch 参改为 `--attention-backend flashinfer`（stock 后端，不是 custom 的 `minicpm_flashinfer`）。上面那个 `NotImplementedError` 只有在显式 override 为 `SOAR_BACKEND_VARIANT=minicpm_flashinfer` 时才会触发。

**Stock flashinfer 完整支持 `TARGET_VERIFY`** —— 是 sglang 上 eagle 和 ngram 路径的正规后端。所以 Stage 3 并未被阻，原本 §1-10 的计划仍然有效，§6 里那条「minicpm_flashinfer TARGET_VERIFY 未在 hybrid GLA 上验证」的风险根本不适用，因为我们本就不用该后端。

### 经验记录

看到源文件的 NotImplementedError 就下结论「被阻」，却不先确认运行时用的是哪个 backend —— 典型的假警报。以后任何「后端不支持」的结论之前，必须先查 `prepare_env.sh` 的默认值。

继续 Stage 3a 实现，针对 stock flashinfer 后端。

## 12. NGRAM 探路发现（2026-05-11）—— 提前定位 Stage 3b cuda-graph 阻塞点

在写任何 Medusa Stage 3a 代码之前，我们在 fcloud 上跑了一次「零代码改动」的 NGRAM speculative 探路，用来验证我们这套配置（stock flashinfer + GPTQ + FP8 KV + dense + mixed-chunk + torch.compile + 16 个 bs 的 cuda-graph buckets）能否端到端承载任何 TARGET_VERIFY 工作。

### 探路结果

服务启动跑到 cuda-graph capture，**在第一个 verify-shape bucket 就崩了**：

```
File "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 515, in _capture_metadata
    if forward_mode.is_target_verify() and spec_info.topk > 1:
AttributeError: 'NgramVerifyInput' object has no attribute 'topk'
```

第 575 行 `_replay_metadata` 里也直接用了 `spec_info.topk` 和 `spec_info.draft_token_num`。

### 对 Medusa 的影响

- `MedusaInput`（`python/sglang/srt/speculative/medusa_info.py`）**既没有 `topk` 也没有 `draft_token_num`** —— grep 已确认。
- hybrid GLA backend 的 `_capture_metadata` / `_replay_metadata` 写的时候默认 spec_info 是 Eagle 形状（tree-mask 带 `topk`、`draft_token_num`、`retrive_next_token` 等）。
- 所以 **Stage 3b（Medusa + cuda-graph）一旦把 cuda-graph 重新开起来跑 TARGET_VERIFY，就会撞到同一个 AttributeError**。这个崩溃不是 Medusa 专属的 —— 它是 hybrid backend 的一个缺口，影响所有非 Eagle 的 speculative 算法（NGRAM、Medusa、未来的 standalone draft）。

### 含义

1. **Stage 3a（eager）仍然可以放心写** —— 崩溃发生在 cuda-graph capture 里，Stage 3a 用 `SOAR_SPEC_MEDUSA_EAGER=1` 关掉 cuda-graph，所以不会触发，可以单独验证正确性。
2. **Stage 3b（cuda-graph）现在多了一个已知前置修复点**，要在 `hybrid_linear_attn_backend.py` 里：
   - 第 515 / 575 行：把 eagle-tree-mask 分支用 `getattr(spec_info, "topk", 1) > 1` 守起来（线性 K-token verify 等价 topk=1，无 tree）。
   - 第 570 行：把 `spec_info.draft_token_num` 改成 `getattr(spec_info, "draft_token_num", None) or self.speculative_num_draft_tokens`（或从 worker 静态配置读）。
   - 这是个约 5 行的容错补丁，风险低，NGRAM + Medusa + 未来任意 linear-verify 算法都受益。
3. **对 v23 提交包没有影响**：v23 默认开 `SOAR_SPEC_MEDUSA=1`，但 Stage 2 medusa_worker.py 是纯 pass-through（`get_model_worker_batch` 前把 `spec_algorithm` 翻成 NONE），运行时根本进不到 TARGET_VERIFY —— 上面那个 hybrid-backend 缺口对 v23 不生效。v23 安全。

### 探路结论

**ROI 非常高。** 一次 5 分钟、零新代码的探路，准确定位了将来 Stage 3b cuda-graph 速度收益的拦路石，同时也证实了 Stage 3a（eager-only）路径上模型加载、KV 分配、hybrid pool 初始化、KV dtype fp8_e5m2 等所有基础设施都正常。再次印证 §11 的教训：「写 speculative 代码前先跑便宜的端到端探路」能省下数小时的假阳性调试。

### Stage 计划更新

| Stage | 原计划 | 现计划 |
|-------|--------|--------|
| 3a | 写 Medusa worker，跑 `SOAR_SPEC_MEDUSA_EAGER=1` | 不变 |
| 3b | 关闭 EAGER → cuda-graph 重新开 → 测速 | **前置**：先给 `hybrid_linear_attn_backend.py` 的 `_capture_metadata` + `_replay_metadata` 打容错补丁，让它能吃 Medusa 形状的 spec_info，然后再关 EAGER 测速 |

Stage 3 启动时，本节将转入 `CHANGE_0156`。
