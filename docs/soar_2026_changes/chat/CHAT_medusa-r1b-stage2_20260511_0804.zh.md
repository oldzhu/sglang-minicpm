# CHAT — Medusa R1b Stage 2 实现 — 2026-05-11 08:04

## 用户本轮提问

> 1. k=1 只预测 1 个 token?k>=2 会预测多于 2 个?如果是,倾向 >=2
> 2. ngramverifyinput 和 medusainput.verify 的区别?
> 3. 288M snapshot 是干嘛的?每步 288M,有多少步?总共会有 288M*N 步的额外
>    内存开销吗?
> 4. eager-only 和坚持用 CUDA graphs 的区别?
> 5. 不如把 Stage 2 落成中间烟囱测试 commit,先评估再去 stage3,你倾向哪个?
>
> go

## Agent 回复

### Q&A 摘要

1. **K=1 vs K≥2**:K=1 每步最多接受 2 个 token;K=N 最多 N+1 个。R1
   的 heads 零初始化 → 接受率预期为 0,与 K 无关。建议:R1b 保持 K=1,
   R3 训练后再提升。
2. **NgramVerifyInput vs MedusaInput.verify**:两者都构造 verify tree,
   结构相同。K=1 Medusa 是 2-token chain,与 ngram_len=2 结构相同。
   复用可省 ~200 LOC。建议:复用。
3. **288 MiB snapshot**:一次 snapshot 在 snapshot+restore 之间存活,
   每步一次。总开销恒为 ~288 MiB,**不**随步数线性累加。
4. **Eager vs CUDA graphs**:Eager 启动开销大但无形状约束。R1b 走
   eager;R1c 再为 verify shape 加 graph capture。
5. **Stage 2 作为中间烟囱测试 commit**:强烈推荐。可隔离接线 Bug 与算法
   Bug。~80 LOC vs Stage 3 的 ~400 LOC。

### 用户 "go" 后采纳的决定

- K=1、Stage 3 复用 `NgramVerifyInput`、R1b 走 eager、本轮发 Stage 2。

### 实施结果

**Stage 2 实际比 CHANGE_0154 §4 描述的还要简单。** 代码巡查显示六处
`is_ngram()` 调用点只在 MEDUSA 主动触发 TARGET_VERIFY capture 时才起作用。
Stage 2 保持在普通 DECODE 模式,六处中有五处无需改动。最终改动:

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/speculative/medusa_worker.py` | 整体重写 R1a stub → ~150 LOC Stage 2 worker(透传 + heads 实例化) |
| `python/sglang/srt/managers/scheduler.py` L866 | 追加 `or self.spec_algorithm.is_medusa()` 守卫(防御性,SOAR 路径不会触达) |
| `docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.{en,zh}.md` | 新建 |
| `docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.{en,zh}.md` | 本文档 |

### Stage 2 worker 行为

- 在 GPU 上实例化 `MedusaHeads(K, hidden, lm_head, dtype)`(K=1、
  hidden=4096、BF16 → 约 32 MiB)。R1a 的 W1 零初始化被保留。
- 启动日志:`MedusaWorker Stage 2 ready: K=1, hidden=4096,
  dtype=torch.bfloat16, device=cuda:0, approx_weight_MiB=32.0`。
- `forward_batch_generation(batch)` 为**纯透传**:取
  `batch.get_model_worker_batch()`,调用
  `target_worker.forward_batch_generation(mwb)`,包装结果返回。
- 不修改 `capture_hidden_mode`,不切换 `forward_mode`,不产生 draft,
  不挂 spec_info。输出与基线逐字节相同。

### 验证判据(在 fcloud 上跑)

- `SOAR_SPEC_MEDUSA=1` server 启动正常。
- Server 日志含 `MedusaWorker Stage 2 ready`。
- 精度 = 基线(逐字节相同的预测,不是接近相同)。
- S1 回退 ≤ 3%(heads 已分配但未调用)。

## 结果

- Stage 2 实现已准备好上 fcloud 验证。
- Stage 3 接力文档见 CHANGE_0155 §8。
- 等待用户批准跑 fcloud workflow(`start-instance` → `sync` →
  `restart-server` → `accuracy` → `speed` → `pause-instance`)。

## 交叉引用

- [CHANGE_0155_medusa_phase_r1b_stage2.zh.md](../CHANGE_0155_medusa_phase_r1b_stage2.zh.md)
- [CHANGE_0154_medusa_phase_r1b_design.zh.md](../CHANGE_0154_medusa_phase_r1b_design.zh.md)
- [CHANGE_0153_medusa_phase_r1_design.zh.md](../CHANGE_0153_medusa_phase_r1_design.zh.md)
- Commit: 推送后补
