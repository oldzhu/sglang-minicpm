# 澄清说明 — MiniCPM-SALA 提交中的量化层级与运行时验证方法

**日期**: 2026-04-27 08:11
**背景**: 用户提问：(1) 既然已经用 GPTQ 做了 INT4 量化、启用 Marlin、加上 FP8 KV，是否已经是 W4A8？(2) `--kv-cache-dtype fp8_e5m2` 是否也让 lightning 递归 state 变成 FP8？(3) GPTQ INT4 是不是 NVFP4？(4) 除了代码审查，运行时如何验证这些说法？
**状态**: 仅讨论 / 参考。无代码修改。延续 `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md` 中的分析。

---

## 1. 三个关键误解（以及为什么各自都错）

| 说法 | 真相 |
|---|---|
| "GPTQ INT4 权重 + Marlin = W4A8" | ❌ Marlin 在 kernel 内部把 INT4 反量化成 BF16，再做 **BF16 × BF16** tensor core 运算。激活精度始终是 **BF16**，不是 8-bit。 |
| "`--kv-cache-dtype fp8_e5m2` 让注意力以 FP8 精度计算" | ❌ 它只改 K/V 在层之间的**存储**格式。读取时立刻 cast 回 BF16 再送入 attention。 |
| "`--kv-cache-dtype fp8_e5m2` 也覆盖 lightning state" | ❌ Lightning state 由另一个池（`MambaPool`）分配，依然是 BF16。该 flag 完全不影响它。 |
| "GPTQ INT4 ≈ NVFP4" | ❌ 完全不同的数字格式，完全不同的硬件执行路径。 |

下面把这四点逐一展开，并提供**运行时验证方法**——你不必只靠代码审查就能确认这些事实。

---

## 2. 我们今天实际在跑的（数据流视角）

### 2.1 GEMM（qkv_proj、o_proj、MLP up/down/gate、lightning Q/K/V/O）

```
输入激活 (BF16) ───┐
                   │
                   ▼
            ┌───────────────────────────────┐
            │ Marlin GEMM kernel            │
GPTQ INT4 ─►│  1. 加载 INT4 权重块           │── BF16 输出激活
            │  2. 反量化到 BF16             │
            │     (按 group_size=128 用 BF16 │
            │      scale)                   │
            │  3. BF16 × BF16 在 BF16        │
            │     tensor core (148 TF)       │
            └───────────────────────────────┘
```

- 权重**存储**: INT4（4 bit/值），节省 DRAM 带宽与显存。
- 权重**反量化后的计算精度**: BF16。
- 激活**存储与计算精度**: BF16。
- Tensor core 路径: **BF16**，SM120 上 148 TFLOPS。
- **完全没有用到 FP8 tensor core (296 TFLOPS)。**

GPTQ + Marlin 的设计决定了这条路径就是这样。要进入 FP8 tensor core 路径，必须换 kernel（W4A8 / mxfp8 / nvfp4-MMA），不是改一个配置 flag 就能切换的。

### 2.2 标准注意力 KV cache（8 个 `minicpm4` 层）

代码定位 `python/sglang/srt/mem_cache/memory_pool.py`：
- 第 668-670 行: `if dtype is FP8: store_dtype = torch.uint8 else store_dtype = dtype`
- 第 1019-1021 行: 读取时 `cache_k = cache_k.view(self.store_dtype)`，再 **cast 回** 后送入注意力计算。

具体流程：

```
注意力层输出 BF16 的 k, v
      │
      ▼
    cast / pack 到 FP8 e5m2（用 uint8 视图存储）
      │
      ▼
   存入 MHATokenToKVPool
      │
      ▼  (下一步注意力读取时)
   加载 FP8 字节
      │
      ▼
   cast 回 BF16
      │
      ▼
   FlashInfer / FA backend 用 BF16 算 softmax(QKᵀ)V
```

所以 `--kv-cache-dtype fp8_e5m2` 是**带宽 + 显存**优化。计算数学始终是 BF16。它仍然非常有价值（KV 读取在 decode 主导带宽），只是不是 "compute precision" 优化。

### 2.3 Lightning 递归 state（24 个 `lightning` 层）

代码定位：
- `python/sglang/srt/mem_cache/memory_pool.py:128` — `class MambaPool`（与 `MHATokenToKVPool` 不同的类）。
- 池基于 `model_config` 的 linear-attn 参数创建，与 `--kv-cache-dtype` 无关。
- 分配 dtype 是 BF16（或 FP32 累加），不论 KV flag 怎么设。

`fused_recurrent_simple_gla` 与 `chunk_simple_gla` 都以 BF16 读写 state。

所以今天：
- 24 个 lightning 层 × 每个请求 × `(num_kv_heads, d, d)` BF16 state 在 `MambaPool` 中。
- 每生成一个 token，每个 lightning 层都要读+写一次 BF16 state。
- 这与 `--kv-cache-dtype fp8_e5m2` **完全无关**。

### 2.4 当前数据布局总览

| 组件 | 存储 dtype | 计算 dtype | 硬件单元 | 控制方 |
|---|---|---|---|---|
| 多数权重（MLP，lightning Q/K/V/O，std-attn 非 sparse QKV） | INT4 (GPTQ) | BF16（反量化后） | BF16 tensor core 148 TF | preprocess_model.py + Marlin |
| 8 标准层 sparse QKV 权重 | INT8 (GPTQ) | BF16（反量化后） | BF16 tensor core | `SOAR_GPTQ_SPARSE_QKV_BITS=8` |
| 所有输入/输出激活 | BF16 | BF16 | BF16 tensor core | model dtype |
| 标准注意力 KV cache | **FP8 e5m2（仅存储）** | BF16（读时 cast） | BF16 attention 数学 | `--kv-cache-dtype fp8_e5m2` |
| Lightning 递归 state | **BF16** | BF16 | Triton kernel BF16 | 目前没有相关 flag |
| FlashInfer / FA backend (8 层) | n/a | BF16 | BF16 | backend 默认 |
| Triton SimpleGLA kernel (24 层) | n/a | BF16 | BF16 | kernel 默认 |

### 2.5 真正的 "W4A8 + FP4-KV + FP8-state" 应该是什么样

| 组件 | 存储 dtype | 计算 dtype | 硬件单元 |
|---|---|---|---|
| 多数权重 | INT4 (GPTQ) | **FP8**（反量化后） | **FP8 QMMA 296 TF** |
| 激活 | **FP8 e4m3** | FP8 | FP8 QMMA |
| 标准注意力 KV cache（混合） | FP8 / 内部层 **FP4 E2M1** | BF16 cast（或走 FP4 QMMA 路径） | BF16 / FP4 attention |
| Lightning 递归 state | **FP8** | FP8（或 BF16 累加） | FP8 in Triton |

这是**三个独立的工程改动**，没有一个能通过当前已有 flag 自动获得。

---

## 3. GPTQ INT4 ≠ NVFP4

完全不同的数字系统，完全不同的 SM120 硬件路径。

| 格式 | 编码 | 硬件执行 | 我们的用法 |
|---|---|---|---|
| **GPTQ INT4** | 均匀整数: `value = scale × (q − zero_point)`，group_size=128，每组 BF16 scale | Marlin 自定义反量化 → BF16 tensor core (148 TF) | 当前权重 |
| **NVFP4 (E2M1)** | 浮点: 1 sign + 2 exp + 1 mantissa，共享 block-scale（通常 16 元素/块） | **QMMA tensor core 直接以 FP4 输入运算**，SM120 上 593 TFLOPS | Test 21（失败：精度 ~12 %） |

一个模型可以是 "4-bit"，但：
- INT4 是均匀网格整数；接近 0 区域容易饱和，远离 0 区域聚集化。
- FP4 是对数式；动态范围大，但靠近 0 时精度差。
- 大多数 LLM 权重均值为 0、近似 Laplace 分布 → 配 per-group scale 的 INT4 很合身；FP4 有时需要 QAT 才行。

结论: 我们当前用的是 **GPTQ INT4**，不是 NVFP4。两者不可互换。（NVFP4 实验在独立分支上做过，Test 21 灾难性精度回退。）

---

## 4. 运行时验证方法（本文档的核心）

你的问题是: "除了代码审查，能否在运行时验证以上说法？" — 可以。每条都给出具体方法、可执行命令、预期输出。

### V1 — 检查模型对象内的 tensor dtype

服务起来之后，在 SGLang 的 Python 进程里加一段一次性 debug 打印（**只在本地用，不要进入提交包**）。例如改 `python/sglang/srt/models/minicpm.py` 的 std-attn 与 lightning 各一个 `__init__`：

```python
# MiniCPMAttention.__init__ (std-attn) 内
print(f"[VERIFY] layer={layer_id} type=std qkv_proj.qweight.dtype={self.qkv_proj.qweight.dtype} (expect torch.int32 packed)")
print(f"[VERIFY] layer={layer_id} type=std qkv_proj.scales.dtype={self.qkv_proj.scales.dtype} (expect bfloat16)")

# MiniCPMLightningMixer.__init__ 内
print(f"[VERIFY] layer={layer_id} type=lightning qkv_proj.qweight.dtype={self.qkv_proj.qweight.dtype}")
```

预期输出:
```
[VERIFY] layer=0 type=lightning qkv_proj.qweight.dtype=torch.int32 (4-bit packed)
[VERIFY] layer=2 type=std qkv_proj.scales.dtype=torch.bfloat16
```

`int32` packed weight (Marlin 存储格式) + BF16 scale 证明：**权重 INT4 存储，BF16 反量化**。GEMM 路径里没有任何 FP8 scale。

### V2 — 确认 KV pool 存储 dtype（FP8）+ cast-back

服务起来后，在同一进程拿到 `model_runner.token_to_kv_pool`：

```python
pool = engine.runner.token_to_kv_pool
print(type(pool).__name__)            # MHATokenToKVPool
print("dtype     =", pool.dtype)      # 逻辑 dtype；预期 torch.bfloat16
print("store_dtype=", pool.store_dtype)  # 预期 torch.uint8 (FP8 字节存储)
print("k_buffer dtype=", pool.k_buffer[0].dtype)  # torch.uint8
```

如果 `dtype == bfloat16` 且 `store_dtype == uint8`，**证明** 池子按 FP8 字节存，但对外宣告 BF16 → 每次读取都要 cast。这正是 `memory_pool.py:1019-1021` 干的事。

### V3 — 确认 lightning state dtype

类似 V2，但针对 mamba pool：

```python
mamba_pool = engine.runner.mamba_pool
print(type(mamba_pool).__name__)        # MambaPool
for i, st in enumerate(mamba_pool.state_buffers[:3]):
    print(f"layer {i} state dtype = {st.dtype}, shape = {tuple(st.shape)}")
```

预期输出:
```
layer 0 state dtype = torch.bfloat16, shape = (max_bs, num_kv_heads, d, d)
```

看到 `torch.bfloat16` → **确认 lightning state 当前是 BF16**，不论 KV flag 怎么设。要让它变 FP8，必须改池子和 kernel（提案 #3）。

### V4 — 确认硬件单元真的是 BF16 还是 FP8 tensor core

这是你最关心的运行时检查 — "FP8 tensor core 是否真的在跑"。

用 **Nsight Compute**：

```bash
# fcloud 上，服务空闲只跑一条请求
nsys profile --trace=cuda,nvtx -o /tmp/trace_decode \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python /root/data/eval_model_001.py --num_samples 1 ...

# 或用 ncu 看每个 kernel 的 SM 单元信息
ncu --target-processes all --section ComputeWorkloadAnalysis \
    --launch-skip 200 --launch-count 30 \
    -o /tmp/ncu_marlin python /root/data/eval_model_001.py --num_samples 1 ...
```

打开 `.ncu-rep`，看 GEMM kernel 这几行的 counter：

- `sm__inst_executed_pipe_tensor_op_hmma` — **BF16/FP16 tensor core (HMMA)**
- `sm__inst_executed_pipe_tensor_op_qmma` — Blackwell 的 **FP8 tensor core (QMMA)**

如果 Marlin kernel 上只有 `hmma` 在跑、`qmma` 是 0，**这就是运行时铁证：我们没有用 FP8 tensor core**。等 W4A8 实现后，应能看到 `qmma` 起来。

当前提交预期: `hmma` >> 0，`qmma` ≈ 0。

### V5 — 通过显存带宽快速 sanity check

Decode 受显存带宽约束。如果 "FP8 KV" 真的只是存储侧，那么改成 BF16 KV 之后应该：
- KV 读取流量大致 **翻倍**（从 1 字节/元素到 2 字节/元素）。
- Decode TPS 下降一定幅度（**不会** 减半，因为权重也占带宽；预计 ~10-20 %，长上下文更明显）。
- GEMM 延迟**不变**（因为两种模式下 GEMM 激活都是 BF16）。

并行跑两组 benchmark：

```bash
# 基线（FP8 KV）
SGLANG_SERVER_ARGS=(... --kv-cache-dtype fp8_e5m2 ...)
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# 对照（BF16 KV）
SGLANG_SERVER_ARGS=(... --kv-cache-dtype auto ...)
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

差别落在 ~10-20 % → 确认 FP8 KV 仅带宽优化。如果是计算精度变化，差别会更大（且 Marlin GEMM 也会快——但实际上不会）。这个对照其实之前跑过：Test 13 (BF16 KV，Smax 更慢) vs Test 12 (FP8 KV)。

### V6 — Lightning state dtype 的快速 sanity check

如果 lightning state 真的是 BF16，把 `max-running-requests` 翻倍应该让 MambaPool 在 L2 中的占用 ~翻倍。如果它私下是 FP8，翻倍量就只有一半。

实操: 服务起来后立刻读 `nvidia-smi --query-gpu=memory.used`，分别在两种 `--max-running-requests` 配置：

```
--max-running-requests 12 → memory.used = X MB
--max-running-requests 24 → memory.used = Y MB
```

`Y - X` 应该等于 `12 × (24 层 × token/req × num_kv_heads × d × d × 字节/元素)`。BF16 → 2 字节/元素 → 与 BF16 假设吻合即确认。

### V7 — 用 SGLang 内置 profiler

SGLang 集成 PyTorch profiler。启用方式：

```bash
curl -X POST http://127.0.0.1:30000/start_profile
# 跑一次推理
curl -X POST http://127.0.0.1:30000/stop_profile
```

trace JSON 含每个 kernel 的元数据。过滤这些 kernel 名：
- `marlin_gemm_*` — 输入 dtype 应是 BF16。
- `flash_attention_v2_kernel_*` — 输入 dtype 应是 BF16。
- `simple_gla_*`（Triton 自动生成名）— 输入 dtype 应是 BF16。

这是端到端检查 kernel 签名最便宜的方式。

---

## 5. 一句话汇总

| 问题 | 回答 | 运行时验证 |
|---|---|---|
| 我们已经是 W4A8 了吗？ | 不是。权重 INT4 存储，但计算是 BF16 × BF16，跑 BF16 tensor core。 | V1 + V4 (ncu 显示 `hmma`，无 `qmma`) |
| FP8 KV cache 是否意味着 FP8 注意力计算？ | 不是。仅存储；读取时 cast 回 BF16。 | V2 (`pool.dtype=bf16, store_dtype=uint8`) |
| FP8 KV cache 是否覆盖 lightning state？ | 不。Lightning state 在 `MambaPool`，是 BF16。 | V3 (`mamba_pool.state_buffers[i].dtype=bf16`) |
| GPTQ INT4 是 NVFP4 吗？ | 不是。编码不同，硬件路径不同。NVFP4 走 QMMA 直接处理 FP4 输入。 | V4 (在 ncu 输出里搜 `qmma_e4m3` 或 `qmma_e2m1`) |
| Lightning state FP8 是否能从现有 flag 自动获得？ | 不能。需要 kernel 改动（提案 #3）。 | V3 提供今日基准 |

---

## 6. 待用户决定的开放问题（下一轮提案前要解决）

1. 是否要在 v18 baseline 上跑一次 V2+V3+V4，把 "今天" 的画面打印出来作为参考？（成本: ~10 分钟 fcloud 时间。）
2. 上一轮研究备忘里的 #1/#3/#4/#6 中，下一轮想让我正式起草哪一个？
3. #1（W4A8）走 GPTQ Marlin 扩展还是 upstream `mxfp8 / nvfp4` 路径？后者集成更顺，但要放弃我们的 GPTQ-INT4 校准。
4. #3（lightning state FP8）希望 per-head 静态 scale 还是 per-token 动态 scale？静态快但风险高。
