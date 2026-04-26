# 研究备忘 — MiniCPM-SALA 混合架构的速度优化路径

**日期**: 2026-04-26 15:26
**背景**: 用户提出三个问题：(1) 4-bit 量化现状，(2) 4-bit KV cache 适用范围，(3) 24 层 lightning（线性注意力）相对于 8 层标准注意力的优化方向。
**状态**: 仅讨论 / 分析，无代码修改。作为下一轮迭代提案选型的输入。

---

## 0. 架构事实（讨论起点）

| 层类型 | 数量 | 计算模型 | KV / state | 复杂度（序列长度 `N`） |
|---|---|---|---|---|
| **`minicpm4`（标准注意力）** | **8** | `softmax(QKᵀ)V`，paged KV cache，RoPE，GQA | Paged KV（当前 FP8 e5m2；上限受 84 GB GDDR7 约束） | Prefill **O(N²·d)** 计算密集；Decode **O(N·d)** 受 KV 带宽约束 |
| **`lightning`（SimpleGLA / 线性注意力）** | **24** | Prefill 用 `chunk_simple_gla`，Decode 用 `fused_recurrent_simple_gla`；递归状态更新 `S = g·S + kᵀv`，输出 `o = q·S` | **固定大小** state `(num_kv_heads, head_dim, head_dim)`，**无 token 维 KV cache** | Prefill **O(N·d²)** ≈ 线性；Decode **O(d²) 每 token，与 N 无关** |

**最重要的一条结论**：序列越长，8 个标准层占总成本越大（KV 随 N 线性增长；lightning state 恒定）。在长上下文（即官方真实评测）场景下，优化这 8 层杠杆最大。但 24 层 lightning 在小上下文 / 短历史 decode 时仍然占主导，因为它们占层数 75 %，每 token 成本是常数 `d²`。

代码定位（2026-04-26 验证）：
- 层分发：`python/sglang/srt/models/minicpm.py:530`（`self.mixer_type = config.mixer_types[layer_id]`）。
- Lightning 内核选择：`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:1632-1672`，根据 `forward_mode.is_decode()` 或 `_select_mode(forward_batch)` 在 `chunk_simple_gla` 与 `fused_recurrent_simple_gla` 间切换。
- Lightning state 形状：`python/sglang/srt/models/minicpm.py:407`。

---

## 1. 用户问题 #1 — "4-bit 量化（已经是 W4A8 了？）"

**并非如此。** 我们当前实际使用（已通过 `benchmark/soar/demo_sala/preprocess_model.py` 与 Marlin 路径校核）：

- 大多数 linear 层：GPTQ **W4A16**（权重 INT4，激活 BF16）。
- 8 个标准注意力层的 QKV：GPTQ **W8A16**（`SOAR_GPTQ_SPARSE_QKV_BITS=8`，`group_size=128`）。
- KV cache：**FP8 e5m2**（与权重量化无关）。
- GEMM 中**激活全部为 BF16**（Marlin W4A16 / W8A16 内核）。

也就是说我们只有 **W4 / W8 weight-only**，**不是** W4A8，也**不是** W4A4。真正的 W4A8（INT8 激活 + INT4 权重，QQQ 风格或 Marlin-W4A8）和 W4A4（NVFP4）是独立的优化方向：

| 方案 | 权重 | 激活 | SM120 预期收益 | 现状 |
|---|---|---|---|---|
| W4A16（当前） | INT4 | BF16 | 1×（基线） | ✅ 已上线 |
| W8A16（sparse QKV） | INT8 | BF16 | 0.6×（比 W4A16 慢） | ✅ 为了精度保留 |
| **W4A8（INT8 激活）** | INT4 | INT8 | ~1.4×（FP8 GEMM 296 TFLOPS） | ❌ 未尝试 — 需要激活校准 + Marlin W4A8 内核路径 |
| **W4A8（FP8 激活，mxfp8 / QMMA）** | INT4 | FP8 | ~1.5-1.8×（SM120 QMMA） | ❌ 未尝试 — Blackwell 上最有前景 |
| **NVFP4 W4A4** | FP4 | FP4 | ~2.5×（理论 593 TFLOPS） | ❌ Test 21 灾难性精度 ~12 % — 摧毁推理能力 |

**新优化机会**：**W4A8 with FP8 激活**。SM120 硬件支持 mxfp8 QMMA。在保留 GPTQ INT4 权重的基础上把激活换为 FP8，相比当前 W4A16 大约能让 GEMM 吞吐翻倍，且预期精度损失远小于 NVFP4 W4A4（FP8 e4m3 激活通常能保留 BF16 99 % 的精度——我们也已经在 KV cache 里验证过 FP8 是安全的）。

---

## 2. 用户问题 #2 — "4-bit KV cache（只对 8 层用？）"

**两点都对。** Lightning 层没有 token 索引型 KV cache，只有固定的 `d×d` 递归状态，无法做 "per-token 量化"。所以 4-bit KV 只能作用于 **8 个标准层**。

为什么 SM120 上 4-bit KV 值得做：
- KV 带宽是 decode 主瓶颈（RTX PRO 6000 = 1398 GB/s）。
- 把 KV 大小从 FP8 减半到 FP4，带宽压力直接减半。
- SGLang upstream 已自带 `MHATokenToKVPoolFP4` 与 `--kv-cache-dtype fp4_e2m1`（`python/sglang/srt/mem_cache/memory_pool.py:1085`，自动创建在 `model_runner_kv_cache_mixin.py:583`）。M2.0 ablation 提案已起草。

**为什么 "混合" 很关键**（冠军方案 W5 经验，结合我们模型再表述）：
- 首层 / 末层精度敏感（注意力 sink、输出决策）。
- 中间层意外地能容忍 FP4 KV。
- 我们 8 个标准层并非连续，散布在 `mixer_types` 中。所以应该按 "8 层中的相对位置" 而不是 "32 层中的绝对 layer_id" 做敏感度判断。

**M2.0 之外的开放方向**：即便在这 8 层内部，也可以尝试 **per-head / per-channel 分级 FP4 KV**：低熵头用 FP4，高熵头保留 FP8。理论上能拿到 95 % 的 FP4 带宽收益和 99 % 的 FP8 精度。比 M2.0 复杂，但属于同一代码路径。

---

## 3. 用户问题 #3 — 线性注意力层的优化（深度问题）

是的，lightning ≈ Mamba/GLA 家族 — 复杂度对 N **线性**而非平方。算法结构：

```
S_t = g_t · S_{t-1} + k_tᵀ v_t      # O(d²) state 更新
o_t = q_t · S_t                       # O(d²) 输出读取
```

`S ∈ R^(d×d)` 大小固定。**没有可以压缩的 seq-len 轴。**

### 3a. 这意味着两种不同的优化方向

| 阶段 | 瓶颈 | 对 lightning 有效的优化 |
|---|---|---|
| **Prefill（计算密集）** | 分块算法 `chunk_simple_gla` — Triton kernel 做 block-sparse 矩阵乘；成本 ≈ `O(N · d²)` 等价 GEMM | (i) 增大 chunk 提升 tensor core 利用率（已有 `SGLANG_FLA_CHUNK_SIZE`，CHANGE_0080），(ii) 在 chunk kernel 内部使用 FP8/FP4 GEMM，(iii) 融合 q/k norm + RoPE + GLA 前置步骤（已有 `fused_qk_norm_rope`） |
| **Decode（权重带宽受限，state 不一定是瓶颈）** | `fused_recurrent_simple_gla` — 每 token 加载 `S`（d² 个字），做 2 个小矩阵乘，再写回 `S`。每层 O(d²) 读写。叠加 24 层 + batch，**state load/store 流量 + 权重加载**主导 | (i) 整层把 `S` 放在寄存器/SMEM（fused_recurrent 已做），(ii) **量化 state 自身**（BF16→FP8 → state 带宽减半），(iii) 把 QKV proj + 递归更新 + O proj 融合到一个 kernel launch |

### 3b. 线性注意力专属的算法优化方向

下面这些杠杆在标准 softmax 注意力上**不存在**：

1. **State 量化（BF16 state → FP8 state）** — 最大且未触碰的杠杆。
   - 每层维护 `S ∈ R^(num_kv_heads × d × d)`，BF16/FP32 存储。`d=128` 时约 64 KB / 层 / 请求。24 层 × bs=24 ≈ 36 MB，能装进 L2（112 MB），但 state load/store 仍主导递归 kernel 内层循环。
   - FP8 state 直接砍半带宽。精度影响未知，但因为 state 受几何衰减 `g` 高度平均化，预期影响很小。
   - 实现：修改 `fused_recurrent_simple_gla` 接受 per-head 量化 scale。kernel 改动 1-2 天。

2. **跨请求 state 重物化 / 流式 SSM 技巧** — 长上下文 decode 中，state 重置周期与重算之间的权衡。我们上下文上限 128k，可能影响有限。

3. **Chunk-size 自动调优** — `chunk_simple_gla` 性能高度依赖 chunk size 与 head_dim、SM 数（SM120 上 96 SMs）的关系。CHANGE_0080 把 chunk size 做成了开关，应该用**官方风格的长上下文速度数据集**（不是我们本地的短上下文集）扫一遍。预期 5-15 % prefill 收益，零精度成本。

4. **Chunk vs 递归 切换阈值调优** — `_select_mode` 当前规则是：prefill 用 chunk，decode 用 fused_recurrent。但 chunked-prefill 中 extend 块很小（64-256 token）时，chunk launch 成本可能超过递归成本，需要运行时阈值切换。

5. **Kernel 融合：QKV-proj + GLA + O-proj 单一 Triton kernel** — 小 batch decode 时，launch + register 填充开销主导。一个融合 kernel `compute_lightning_layer(x_in, W_qkv, W_o, S_inout) -> x_out` 可把 3 次 launch 砍到 1 次。重写工作量大（~1-2 周），但 lightning 占 24/32 层，回报值得。

6. **矩阵形式并行 decode（speculative decoding 的天然朋友）** — 当 spec draft 一次提交 `g` 个候选 token，lightning 层可以用一次 `chunk_simple_gla` 而不是 `g` 次递归调用就消化完。线性注意力是**最理想的 speculative decoding 目标** — 验证成本天然摊销。这把 spec dec 从 "难赢" 变成 "易赢"，正是我们这套架构的特殊优势。

7. **Output gate / RMSNorm 融合** — 已有 `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE`。需确认 v18 路径上确实生效，是另一个小常数收益。

### 3c. Lightning 层不要尝试的方向（避免浪费时间）

- ❌ Sparse attention / topk attention — 只对平方 softmax 注意力有意义。
- ❌ KV cache 压缩 — 不存在 token-indexed KV。
- ❌ Flash-attention v2/v3 — 是 softmax 注意力的内核。
- ❌ Sliding window — seq-len 已经被 `S` 折叠掉。

---

## 4. 当前代码库的具体优先级排序

按 `(预期收益 × 概率) / 工作量` 排序，全部基于 v18 baseline：

| # | 优化项 | 影响层 | 预期收益 | 工作量 | 风险 | 时机 |
|---|---|---|---|---|---|---|
| **1** | **W4A8 FP8 激活（mxfp8 QMMA）** | 所有 linear 层 | **GEMM 吞吐 +30-50 %** | 1-2 周（新 Marlin kernel 路径） | 中 | SM120 硬件支持；当前 296 TFLOPS FP8 未利用 |
| **2** | **M2.0 → M2.1 混合 FP8/FP4 KV** | 8 标准层 | Decode TPS +10-15 %，KV 内存下降 | 低（已有提案，~50 LOC） | 低（per-layer ablation 找安全层） | 已起草 |
| **3** | **Lightning state FP8 量化** | 24 lightning | 全模型 decode TPS +5-10 % | 中（kernel 改动） | 中-低 | 未触碰；线性路径上最大收益 |
| **4** | **Lightning 层融合（QKV + GLA + O）Triton kernel** | 24 lightning | 小 bs 时延迟 +5-15 % | 高（~2 周） | 中 | 减少小 bs launch 开销 |
| **5** | **Chunk size 扫描 + 切换阈值** | 24 lightning | Prefill +3-7 % | 低（环境变量扫一遍） | 低 | 廉价探索 |
| **6** | **Speculative decoding (n=2-3) 利用线性注意力低成本验证** | 全部 | 低 bs decode TPS +20-40 % | 中（draft 模型 + 集成） | 中 | 与 mcq 长尾问题部分重叠 |
| **7** | **Per-head / per-channel 分级 FP4 KV** | 8 标准层 | 比统一 FP4 再 +5-10 % | 中 | 中 | M2.0 落地后再做 |

**建议**：先完成 M2.0（#2，已起草），再以 **#1（W4A8 FP8）和 #3（lightning state FP8）** 作为两条并行工作线推进。两者结合现实地看能把我们推进到排行榜中段。**#6（speculative decoding）** 是不做大架构改动的前提下进入 top-5 的唯一路径，而且我们架构对它格外友好。

---

## 5. 必须跟踪的约束与风险

- **本地 vs 官方精度差**：本地准确率目标 ≥ 80 % 留安全边际（私有集未知）。
- **本地 vs 官方速度差**：官方速度集长上下文样本更多；W4A8 / lightning-state-FP8 / spec dec 都对长上下文 decode 帮助最大。
- **提交约束**：≤ 2 GB 总量、上交后量化、≤ 5h 总时间。新 W4A8 内核会增大 wheel 体积，每次实现后重新测量。
- **精度系数悬崖**：始终保 `C ≥ 0.96`；绝不为 < 5 % 速度收益牺牲 > 1pt 本地精度。

---

## 6. 待用户决定的开放问题（下一轮提案前要解决）

1. #1 / #3 / #4 / #6 中，下一轮想让我正式起草哪一个？
2. #1（W4A8）应基于现有 GPTQ Marlin 路径扩展，还是走 upstream `mxfp8` / `nvfp4` 路径（`sgl-kernel/`）？
3. #3（lightning state FP8）希望用 per-head 静态 scale 还是 per-token 动态 scale？静态快得多但精度风险高。
4. #6（spec dec）：用 MiniCPM-SALA 自身的小 draft（n-gram 或局部层子集）还是独立 draft 模型？后者占提交体积。
