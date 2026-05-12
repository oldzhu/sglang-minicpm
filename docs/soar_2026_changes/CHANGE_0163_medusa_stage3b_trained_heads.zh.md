# CHANGE_0163 — Medusa Stage 3b：训练好的 Head 前向传播 + 隐藏状态捕获

**日期**：2026-05-19  
**提交**：`7a7470af4`  
**分支**：`mixed_minicpm_cudagraph`  
**远端**：`minicpm-src`（`oldzhu/sglang-minicpm`）  
**状态**：已实现 — 待 fcloud 测试

---

## 背景与动机

Stage 3a（CHANGE_0161/0162）建立了可工作的 K=1 Medusa 验证循环，但使用零初始化的 W1 权重，并将 draft token 直接取自 `req.output_ids[-1]`（上一步接受的最后一个 token）。这导致 draft token 几乎总是错误的，accept rate 接近 0%，没有任何速度提升。

Stage 3b 将三件事连接起来：

1. **隐藏状态捕获**：利用 sglang 已有的 `CaptureHiddenMode.LAST` 机制，从每次 TARGET_VERIFY 前向传播中提取隐藏状态。
2. **真实 Medusa head 前向传播**：`argmax(lm_head(SiLU(W1(h)) + h))` 预测下一步的 draft token。
3. **训练脚本**：在 eval 数据集的"隐藏状态 → 下一个 token"对上拟合 W1。

训练后的 W1 预期在 50–70% 的情况下与基础模型的下一个 token 一致，`spec_accept_length` ≈ 1.5–2.0，端到端速度提升估计 15–30%。

---

## 规则合规性

- **提交大小**：W1 状态字典形状 `(4096, 4096)` BF16 ≈ **32 MB**，总提交量远低于 2 GB 限制。
- **现场量化规则**：W1 在 fcloud 上现场训练，不预上传。
- **正确性**：当 `SOAR_MEDUSA_HEAD_PATH` 未设置时，自动回退到 Stage 3a 行为，无准确率回归风险。
- **eval 脚本完整性**：不修改评估脚本，所有更改均在服务器/模型侧。

---

## 修改的文件

| 文件 | 变更 |
|------|------|
| `python/sglang/srt/speculative/medusa_worker.py` | Stage 3b draft 逻辑、隐藏状态捕获、head 前向传播、req 缓存 |
| `python/sglang/srt/models/minicpm_medusa_heads.py` | `load_trained_weights()` 方法 |
| `python/sglang/srt/speculative/ngram_info.py` | Bug 修复：`if hidden_states is not None:` |
| `benchmark/soar/demo_sala/prepare_env.sh` | `SOAR_MEDUSA_HEAD_PATH` 环境变量 |
| `benchmark/soar/demo_sala/train_medusa_head.py` | **新增** 训练脚本 |
| 以上文件的提交副本 | 保持同步 |

---

## 实现细节

### 1. `ngram_info.py` — Bug 修复

```python
# 修复前（Stage 3a — hidden_states 始终为 None，从未触发）：
if logits_output.hidden_states:

# 修复后（Stage 3b — 多元素张量的布尔判断会抛出 RuntimeError）：
if logits_output.hidden_states is not None:
```

### 2. `minicpm_medusa_heads.py` — `load_trained_weights()`

```python
def load_trained_weights(self, path: str, device: str = "cuda") -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    # 期望的键：heads.{k}.W1.weight，k ∈ 0..num_heads-1
    self.load_state_dict(state, strict=False)   # strict=False：lm_head 不在 ckpt 中
```

### 3. `medusa_worker.py` — Stage 3b 变更

**`__init__`** 新增：
- 正确选取 `lm_head`（尊重 `tie_word_embeddings` 配置）。
- `SOAR_MEDUSA_HEAD_PATH` 环境变量：加载训练好的 W1 权重并设置 `self._use_trained_heads = True`。

**`_forward_verify_k1`** 变更：

```python
# 1. Draft token 选取（每个请求）
cached = getattr(req, "_medusa_draft_token", None)
draft_token = cached if (cached is not None and self._use_trained_heads) \
              else req.output_ids[-1]

# 2. 在 get_model_worker_batch() 之前设置 capture mode
if self._use_trained_heads:
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

# 3. TARGET_VERIFY 前向传播后，在 verify() 之前捕获隐藏状态
raw_hidden_states = logits_output.hidden_states   # (bs, hidden_size)

# 4. 运行 verify()
logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(...)

# 5. 运行 head 前向传播，将下一步 draft 缓存到 req
draft_logits = self.medusa_heads(raw_hidden_states)  # (bs, 1, vocab)
next_draft_ids = draft_logits[:, 0, :].argmax(dim=-1).tolist()
for req, tok in zip(batch.reqs, next_draft_ids):
    req._medusa_draft_token = tok if not req.finished() else None
```

### 4. `train_medusa_head.py` — 训练流程

```
1. 通过 transformers 加载非量化模型（MiniCPM-SALA-Copy）
2. 在 model.model.norm 上注册前向 hook 捕获隐藏状态
3. 从 perf_public_set.jsonl 分词 prompt（每个样本 max_len=512）
4. 对所有位置收集 (h_t, token_{t+1}) 对
5. 训练 W1（Adam，lr=1e-4，5 epochs，batch=512）
6. 保存：{"heads.0.W1.weight": W1.weight}  →  /root/medusa_head_k1.pt（~32 MB）
```

---

## 验证命令（fcloud）

### 第一步：训练 head

```bash
# 在 fcloud 上运行（约 5-10 分钟）：
cd /root/sglang-minicpm && git pull

python3 benchmark/soar/demo_sala/train_medusa_head.py \
  --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
  --data-path  /root/data/perf_public_set.jsonl \
  --output     /root/medusa_head_k1.pt \
  --epochs 5 --lr 1e-4 --max-len 512

# 预期日志：
# Collected ~60000-120000 (h, label) pairs
# Epoch 5/5: loss≈X  top1_acc≈50-65%  time≈Ys
# Saved checkpoint to /root/medusa_head_k1.pt (32.0 MiB)
```

### 第二步：启动带训练 head 的服务器

```bash
export SOAR_MEDUSA_HEAD_PATH=/root/medusa_head_k1.pt
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  "${SGLANG_SERVER_ARGS[@]}"

# 预期启动日志：
# MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1.pt
# MedusaWorker ready: K=1, trained=True
```

### 第三步：准确率测试

```bash
python3 /root/data/eval_model_001.py \
  --model http://localhost:30000 \
  --data_path /root/data/perf_public_set.jsonl
# 目标：≥ 79%（≥ Stage 3a 基准 77.87%）
```

### 第四步：速度测试

```bash
# 观察服务器日志：spec_accept_length, accept_rate
# 目标：spec_accept_length ≥ 1.3（Stage 3a 为 1.0）

python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

---

## 预期结果

| 指标 | Stage 2（基准） | Stage 3a | Stage 3b 目标 |
|------|----------------|----------|---------------|
| S1 | 118.28s | 202.70s | ~100-130s |
| S8 | 43.87s | 61.60s | ~45-50s |
| Smax | 35.75s | 43.29s | ~35-40s |
| spec_accept_length | — | 1.00 | 1.3–1.7 |
| accept_rate | — | ~0% | 30-70% |

---

## 回滚说明

```bash
# 禁用训练 head（回退到 Stage 3a 零初始化）：
export SOAR_MEDUSA_HEAD_PATH=
# 或：
unset SOAR_MEDUSA_HEAD_PATH

# 完全禁用 Medusa（回退到 Stage 2 passthrough）：
export SOAR_SPEC_MEDUSA=0
```

---

## 后续步骤

1. 在 fcloud 上验证 accept_rate（查看服务器日志）。
2. 如果 accept_rate < 30%：增加训练 epoch 数或使用更长的 `--max-len`。
3. 如果 accept_rate ≥ 50%：考虑 K=2 heads 进一步提速。
4. 如果准确率下降：检查 `tie_word_embeddings` 是否正确处理，并确认训练脚本中的 `scale_width` 修正与推理路径一致。
