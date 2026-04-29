# 研究 — `flashinfer` vs `minicpm_flashinfer` 后端代码流对比

**日期**: 2026-04-29
**触发**: Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`) 显示出 ~7% 真实速度
收益,但准确率下降 2.4pt(76.91% vs Test 12 79.29% → C=0)。用户提问:
*这究竟是 SALA 混合 sparse/dense 层的不兼容,还是有可调旋钮把它救回成下一个
baseline?*

本文档完整梳理两个后端字符串的实际代码路径,定位损失 2.4pt 的行为差。

## 速览

1. 两个字符串注册的是**不同的后端类**,不是 kernel 切换。
2. `--attention-backend minicpm_flashinfer` →
   `attention_registry.create_minicpm_flashinfer_backend` → 实例化
   `MiniCPMSparseBackend`(自定义、稀疏感知)。
3. `--attention-backend flashinfer` → 实例化原版
   `FlashInferAttnBackend`(无稀疏路由,无 compress KV cache)。
4. 加上 `--force-dense-minicpm` 时,**两个**字符串都会落到 `FlashInferAttnBackend`
   — `server_args.py:1525` 把 `minicpm_flashinfer → flashinfer` 重写。**这就是
   Test 12 baseline 的实际运行状态。**
5. 因此 Test 12(79.29%)和 Round 13f-1(76.91%)**std-attn kernel 完全相同**
   (都是 stock FlashInfer)。两者差异**不在** std-attn kernel,而在与
   `--force-dense-minicpm` 一起切换的**周边配置**:
   - `model_config.has_sparse_attention`: True(无 force) ↔ False(有 force)。
   - `model_config.sparse_layer_ids`: 非空 ↔ 空。
   - Lightning mixer `recurrent_threshold`: 64(无 force) ↔ 128(有 force)。
   - `--dense-as-sparse`: Round 13f-1 单独丢弃。

## 后端注册(分歧起点)

`python/sglang/srt/layers/attention/attention_registry.py`:

```python
@register_attention_backend("minicpm_flashattn")     # line 180
def create_minicpm_flashattn_backend(runner):
    from sglang.srt.layers.attention.minicpm_backend import MiniCPMSparseBackend
    return MiniCPMSparseBackend(runner)

@register_attention_backend("minicpm_flashinfer")    # line 190
def create_minicpm_flashinfer_backend(runner):
    from sglang.srt.layers.attention.minicpm_backend import MiniCPMSparseBackend
    return MiniCPMSparseBackend(runner)
```

```python
@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    return FlashInferAttnBackend(runner)
```

`MiniCPMSparseBackend.__init__` 中再次读取**字符串**,在内部选择 dense FA
具体实现(flash_attn vs flashinfer):

```python
elif attention_backend == "minicpm_flashinfer":      # minicpm_backend.py:366
    self.use_flashinfer = True
```

所以两个 `minicpm_*` 字符串只在**稀疏后端内部**的 kernel 选择上不同。
而不带 `minicpm_` 的 `flashinfer` 字符串结构上是另一个后端类。

## 模型如何接线

`MiniCPMSALAForCausalLM` 是混合模型:

| 层族 | 构造(minicpm.py) | 运行时使用的后端 |
|------|-------------------|------------------|
| 标准 attention(`MiniCPMAttention`,line 246) | `RadixAttention(...)` | `forward_batch.attn_backend.full_attn_backend.forward_extend / forward_decode` |
| Lightning mixer(`MiniCPMLightningMixer`,line 569) | 直接 CUDA kernel,经 `SimpleGLAAttnBackend` | `forward_batch.attn_backend.linear_attn_backend.*` |

`forward_batch.attn_backend` 是 `HybridLinearAttnBackend`,由
`attn_backend_wrapper` 构造,持有两个子 backend:

- `full_attn_backend` ← wrapper 传入,即 `attention_backend` 注册的后端
  (`MiniCPMSparseBackend` 或 `FlashInferAttnBackend`)。
- `linear_attn_backend = SimpleGLAAttnBackend(...)`。

模型代码本身(`minicpm.py:279`)是**无关**的:只调用
`self.attn(q, k, v, forward_batch)`。这条 std-attn 层在 SALA 里是不是
"sparse" 层,**完全由后端决定**(后端在构造时读
`model_runner.model_config.sparse_layer_ids`)。

## 端到端 forward 路径

### 路径 A — `--attention-backend minicpm_flashinfer`,无 `--force-dense-minicpm`

```
Scheduler → ModelRunner.forward
  → MiniCPMSALAForCausalLM.forward
      → MiniCPMDecoderLayer.forward
          ├── self_attn = MiniCPMAttention (std)        ─┐
          │     RadixAttention.forward                   │
          │       forward_batch.attn_backend.forward     │
          │         = HybridLinearAttnBackend.forward    │
          │             按 layer_type dispatch          │
          │               → full_attn_backend.forward    │
          │                  = MiniCPMSparseBackend      │
          │                       逐请求决策:           │
          │                         seq_lens >= dense_len?
          │                           是 → top-k 稀疏 FA  ◄──── 稀疏路由
          │                                  + compress k1/k2
          │                                  + sparse_page_table
          │                           否 → dense FA fallback ◄──── 全 attn
          │                                  via flashinfer kernel
          │                                  (sparse_page_table 拷贝 → BUG #0137)
          │
          └── self_attn = MiniCPMLightningMixer          ─┐
                forward_batch.attn_backend.linear_attn… ─┘
```

仅此路径独有的特征:
- `compress_k1/k2` 缓存(均值池化的 K)用于 top-k page 选择。
- 每条请求各自的 `sparse_page_table`。
- cudagraph 按 (decode bs, head_group_num, sparse_topk*block_size) shape 捕获,
  与 stock flashinfer 的 capture 不同。
- HEAD 上由潜伏 bug(CHANGE_0133、上文 CHANGE_0137)暴露问题。

### 路径 B — `--attention-backend flashinfer`(Round 13f-1)

```
Scheduler → ModelRunner.forward
  → MiniCPMSALAForCausalLM.forward
      → MiniCPMDecoderLayer.forward
          ├── self_attn = MiniCPMAttention (std)        ─┐
          │     RadixAttention.forward                   │
          │       forward_batch.attn_backend.forward     │
          │         = HybridLinearAttnBackend.forward    │
          │             → full_attn_backend.forward      │
          │                  = stock FlashInferAttnBackend
          │                     prefill: BatchPrefillWith*KVCacheKernel
          │                     decode:  BatchDecodeWithPagedKVCacheKernel
          │                     **无稀疏路由,无 top-k**
          │                     **每一层 = 完整 dense attention**
          │
          └── self_attn = MiniCPMLightningMixer          ─┐
                forward_batch.attn_backend.linear_attn… ─┘
```

相对路径 A 的关键特征:
- 不加载 `MiniCPMSparseBackend`,不分配 `compress_k1/k2`。
- 不存在 `sparse_page_table`。
- 所有 std-attn 层运行**完整** dense attention。SALA 的稀疏训练层看到的是
  无 mask 的注意力分数(无 top-k 过滤)。
- 这数学上是 top-k 稀疏注意力的**超集**,信息论上**多**了上下文,而非更少。
  那么准确率为何下降? 见下文讨论。

### 路径 C — `--attention-backend minicpm_flashinfer --force-dense-minicpm`(Test 12 baseline)

```
server_args.py:1521-1525:
    if force_dense_minicpm and attention_backend == "minicpm_flashinfer":
        attention_backend = "flashinfer"

→ 运行时实际就是路径 B。
```

但额外有:
- `model_config.has_sparse_attention` → `False`。
- `model_config.sparse_layer_ids` → `[]`。
- `default_recurrent_threshold = 128`(路径 B 里是 64)。

也就是说 **路径 B 与路径 C 使用相同的 std-attn kernel**(stock flashinfer 的
`BatchPrefillWith*KVCacheKernel`),但**差异**在于:
1. Lightning mixer recurrent vs chunk 模式阈值(64 vs 128)。
2. `model_config` 是否暴露 `sparse_layer_ids`。(stock flashinfer 不读;但其他
   代码路径可能读 — 例如 KV cache 布局、权重加载器。)
3. `SGLANG_SERVER_ARGS` 中是否带 `--dense-as-sparse`。Test 12 保留;Round 13f-1
   丢弃。stock flashinfer 不消费此 flag,但仍由 `server_args` 解析,可能流入
   其他子系统。

## 路径 B(Round 13f-1)与路径 C(Test 12)运行同一 kernel,为何路径 B 损失 2.4pt

std-attn kernel 完全相同。准确率差只能来自与 `force_dense_minicpm` 一起切换
的周边旋钮:

### 假设 1(最可能)— Lightning mixer recurrent 阈值

`hybrid_linear_attn_backend.py:1484`:
```python
default_recurrent_threshold = 128 if self.force_dense_minicpm else 64
```

Lightning mixer 有两套实现:
- **chunk 模式**(seq_len ≥ threshold)— 分块 SimpleGLA,长序列更快,但通过
  分块 causal kernel 累积状态。
- **recurrent 模式**(seq_len < threshold)— 逐元素递推状态更新,与训练数学
  完全一致。

路径 B 阈值 = 64,路径 C 阈值 = 128。许多 eval prompt 的 extend_seq_len 落在
[64, 128) 区间(短 mcq ~60-100 tokens、qa 短答、running > 64 的 decode 步)。
路径 B 走 **chunk** 模式,路径 C 走 **recurrent** 模式。两者在 float32 下
等价,但**低精度下偏离**,因为 chunk 模式的分块 reduction 累积 FP 误差不同。

这是 150 样本 benchmark 上损失 2.4pt 的**最强候选**,尤其对 cwe(损失最严重:
路径 B 82.33% vs 期望 ≥87%)和 qa(56.67% vs Test 20 63.33%)这类对注意力
状态精度极敏感的检索任务。

### 假设 2 — `has_sparse_attention=True` 副作用

若 `has_sparse_attention=True`,模型加载器 / KV cache 可能:
- 为稀疏层分配不同 KV 布局(例如为 compress_k1/k2 留出额外空间)。stock
  flashinfer 不消费 compress,缓存只是闲置 — 应无害。
- 触发不同的 rope / scaling 路径。需在模型构造代码里 grep
  `sparse_layer_ids` 用法。

**次级**候选 — 可通过保持 `attention_backend=flashinfer` 但强制
`has_sparse_attention=False` 来便宜验证。

### 假设 3 — `--dense-as-sparse` 移除暴露配置漂移

`--dense-as-sparse` 由 `server_args` 解析,影响
`MiniCPMSparseBackend.__init__`(强制 dense_len=0)。stock flashinfer 不使用。
路径 B 上移除应当无副作用。**大概不是元凶。**

### 假设 4 — KV cache pool 布局差异

`attention_backend=flashinfer` 时(路径 B 和路径 C 都)KV pool 类型由
`kv_cache_dtype=fp8_e5m2` 与后端首选布局决定。两路径选同一 pool。
**大概不是元凶。**

## 建议实验(便宜、可并行)

用户明确希望把 Round 13f-1 演化为可用的提交 baseline。前进路径是在路径 B
之上找最小配置 delta,既保住速度又恢复 Test 12 级准确率。

| # | 实验 | 验证 | 预期 |
|---|------|------|------|
| **A** | 路径 B + `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128` | 假设 1 | 若准确率回升 ≥1.5pt → 阈值是主因,应聚焦该旋钮。 |
| **B** | 路径 B + `--force-dense-minicpm` 但保留 stock flashinfer 字符串 | 把 force-dense 所有副作用(config + 阈值)叠加,而无需重写 backend 名 | 若准确率回到 Test 12 → 速度收益与 backend 切换无关;直接出新 baseline。 |
| **C** | 路径 C 把 `--max-running-requests` 降至 16(更接近 Round 13f-1 的有效并发) | 排除调度器影响 | 准确率应不动;反向印证 kernel/阈值答案。 |
| **D** | 路径 B 重新启用 `--enable-torch-compile --torch-compile-max-bs 8` | Round 13f-1 丢了 compile;Test 12 保留。compile 路径或可稳定数值 | 若 compile 救回准确率 → 数值稳定性是核心。 |

推荐先跑**实验 B**。如果 `--force-dense-minicpm` + stock flashinfer(实际上
就是 Test 12 已有的同一服务器配置)能复现 79.29%,那 Round 13f-1 的"速度
收益"是错觉(同 backend、同 kernel),7% 差异必来自 {compile, lightning 阈值,
调度器} 之一。在显式 `flashinfer` 字符串下复现该状态 → 已经就是新 baseline。

如果实验 B 速度回归到 Test 12 水平,说明 Round 13f-1 的速度增益恰好来自
**移除** `--force-dense-minicpm`,即 stock flashinfer 在 `has_sparse_attention=True`
状态下运行。该状态正是损失 2.4pt 的源头,需进一步刻画。

## 回答原始问题

> *flashinfer backend 真的处理不了 minicpm 的混合层(sparse + lightning)吗?*

- **Lightning 层**:两个路径都通过 `SimpleGLAAttnBackend` **正确**处理 — 与
  std-attn 后端无关。
- **稀疏 std-attn 层**:stock flashinfer **会算**,只是按**完整 dense
  attention**(top-k 稀疏的超集)算,不会崩,会输出数值合法的结果。准确率
  能否对齐训练取决于 SALA 微调对"看到无 mask 注意力"的鲁棒性。实证 Round
  13f-1 损失 2.4pt — **大概率不是 sparse-vs-dense 计算本身**,而是路径 B
  vs 路径 C 之间一同切换的周边配置(lightning 阈值、`has_sparse_attention`
  flag)。
- **结论**:stock flashinfer **功能上与 SALA 兼容**。它**当前未配置成与
  minicpm_flashinfer + force-dense 路径准确率等价**的状态。修复方向是配置
  对齐,不是替换 kernel。上面四个实验可定位哪个旋钮拥有那 2.4pt。

## 交叉引用

- 测试行: TEST_RESULTS_TRACKING.md `R13f1-flashinfer`。
- Round 13f-1 chat: [chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md](chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md)。
- CHANGE_0136 暴露的 off-by-one(仅路径 A): [CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md](CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md)。
- 代码锚点:
  - [python/sglang/srt/layers/attention/attention_registry.py](../../python/sglang/srt/layers/attention/attention_registry.py#L180-L220)
  - [python/sglang/srt/server_args.py](../../python/sglang/srt/server_args.py#L1521-L1525)
  - [python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py](../../python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1484)
  - [python/sglang/srt/layers/attention/minicpm_backend.py](../../python/sglang/srt/layers/attention/minicpm_backend.py#L366)
