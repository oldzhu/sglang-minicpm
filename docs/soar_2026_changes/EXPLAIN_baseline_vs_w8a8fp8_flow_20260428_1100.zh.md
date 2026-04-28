# 深度解析 — Baseline (W4A16 BF16) 与 W4 模型上 W8A8 (FP8) 执行流的对比

**日期**：2026-04-28
**状态**：解释文档。无代码改动。回答用户关于两条路径的代码级/指令级/硬件级差异问题。

本文档与 `ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md` 互补。请**先读**那篇做高层成本分析，**再读**本篇做逐行执行追踪。

---

## 0. 路线图

端到端对比两条路径：
- **Path A（v18 baseline，生产中）**：GPTQ W4 + Marlin GEMM + BF16 激活 + FP8 e5m2 KV
- **Path B（W4A8 #1 误标签测试，env-gated 关闭）**：同样的 GPTQ W4 checkpoint，但加载器把权重重新量化为 FP8 e4m3 → CUTLASS FP8×FP8 blockwise GEMM + FP8 e4m3 per-token 激活 +（仍是）FP8 e5m2 KV

每条路径追踪：**加载 → forward GEMM → KV 路径**，附文件:行引用、实际 MMA 指令、寄存器/smem 占用。

---

## 1. Path A — Baseline v18：GPTQ W4A16 + Marlin

### 1.1 加载时（无权重膨胀）

[gptq.py](python/sglang/srt/layers/quantization/gptq.py) `GPTQMarlinLinearMethod.create_weights` —— 第 ~545 行：

```python
qweight = PackedvLLMParameter(
    data=torch.empty(
        input_size_per_partition // self.quant_config.pack_factor,  # K // 8
        output_size_per_partition,                                    # N
        dtype=torch.int32,            # 每 int32 打包 8 个 INT4
    ),
    ...
)
scales = ChannelQuantScaleParameter(
    data=torch.empty(
        scales_and_zp_size,           # K // group_size = K // 128
        output_size_per_partition,    # N
        dtype=params_dtype,           # torch.bfloat16
    ),
    ...
)
qzeros = PackedColumnParameter(
    data=torch.empty(
        scales_and_zp_size,            # K // 128
        output_size_per_partition // 8,
        dtype=torch.int32,             # 每 int32 打包 8 个 INT4 zero point
    ),
)
```

加载后 `process_weights_after_loading` 调用 `gptq_marlin_repack`（~720 行）把 int32 打包重排为 Marlin tile-friendly 布局——**存储仍为打包 INT4**，只改字节 layout。

**HBM 占用每层**（示例：K=11008、N=4096、group_size=128）：

| 张量 | 形状 | 类型 | 字节 |
|---|---|---|---|
| qweight | (K/8, N) | int32 | (1376 × 4096) × 4 = **~22.5 MB** |
| scales | (K/128, N) | bf16 | (86 × 4096) × 2 = **704 KB** |
| qzeros | (K/128, N/8) | int32 | (86 × 512) × 4 = **176 KB** |
| **HBM 合计** | | | **~23.4 MB** |

### 1.2 "fp16/bf16 group scales kept" 是什么意思——以及资源开销

"group scales" = 与 INT4 权重一起存的**每 128 元素组的 fp16/bf16 乘子**。它们是**常驻 HBM 张量**（不是寄存器），加载时分配一次。

**HBM 开销（每层，示例）：**704 KB。整个模型（~24 std-attn linear + 32 MLP linear，K/N 各异）：合计几十 MB——相对总权重 ~6 GB 可忽略。

**寄存器开销（GEMM 期间，瞬时）：**Marlin K 循环的每次迭代中，每个 warp 通过 `ldmatrix` 或 `cp.async` 加载当前组的部分 scales 到共享内存，然后广播到线程寄存器（每线程每 K-tile 通常 ~4–8 个 fp16/bf16 值）。这是噪声级；Marlin 的热点寄存器使用由累加器和权重 fragment 主导，不是 scales。

**为什么需要 scales**：GPTQ INT4 每组只编码 16 个等级。fp16/bf16 scale 告诉你这 16 个等级覆盖 `[−scale × 8, scale × 7]` 范围。没 scale，int4 值无意义。Per-group 而非 per-channel 是因为相同压缩比下 per-group 精度好得多（GPTQ 的贡献）。

### 1.3 Forward —— 调用链

```
Linear.forward(x)               # x: (M, K) bf16, M = batch*seq
  → GPTQMarlinLinearMethod.apply(layer, x, bias)         # gptq.py L871
    → apply_gptq_marlin_linear(...)                      # marlin_utils.py L464
      → gptq_marlin_gemm(...)                            # sgl-kernel binding
        → CUDA kernel in sgl-kernel/csrc/gemm/marlin/    # 实际 GEMM
```

kernel 在 K 循环内做**融合的 INT4→BF16 反量化**：

```cuda
// sgl-kernel/csrc/gemm/marlin/* （简化）
for (int ki = 0; ki < K; ki += 16) {
    // 1. 从共享内存加载打包 INT4（来自 HBM via cp.async）
    uint32_t w_packed = *(weight_smem + ki/8);

    // 2. 解包 8 个 int4；减零点；乘 bf16 scale
    //    结果作为 bf16 fragment 存于寄存器（每个打包 int32 含 8 个）
    bf16_w_frag = (int4_unpack(w_packed) - zero) * bf16_scale;

    // 3. 把 bf16 权重 fragment + bf16 激活 fragment 喂给张量核心
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ...");
}
```

**MMA 指令**在 [marlin_template.h](sgl-kernel/csrc/gemm/marlin/marlin_template.h) 第 57–68 行：
```cuda
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
  : "=f"(c[0..3])    // 每 warp slot 4× FP32 累加器
  : "r"(a[0..3]),    // 4× uint32 (各含 2× bf16) —— A frag
    "r"(b[0..1]),    // 2× uint32 (各含 2× bf16) —— B frag
    "f"(c[0..3]));   // 初始累加器（FP32）
```

**关键**：FP32 累加器是 BF16 MMA 的硬件约束——PTX 没有 BF16 累加器选项。MMA 始终在 FP32 中累加。这就是为什么需要"epilogue"。

### 1.4 "BF16 epilogue" 是什么意思

K 循环结束后，每个 warp 的累加器是 **FP32**（每线程每输出 fragment 4 个 fp32 值）。Epilogue 是 MMA 后的小后处理：

1. 从寄存器读 FP32 累加器。
2. 可选地加 bias（广播 bf16 → fp32 → fmadd）。
3. 通过 `__float2bfloat16` 内置（每元素一条 PTX `cvt.rn.bf16.f32`）把 FP32 → BF16。
4. 把 BF16 结果写到 global 内存。

所以是的，"BF16 epilogue" = **FP32 累加器在写回前被 cast 成 BF16**。张量核心上无法跳过 FP32→BF16；问题只是它发生在 kernel 内融合（Marlin 是这样）还是作为单独 kernel（不是）。

### 1.5 KV 缓存（FP8 e5m2）

KV 分配为 `torch.float8_e5m2`（每值 1 字节）—— [model_runner.py](python/sglang/srt/model_executor/model_runner.py) 第 ~1541 行。在 FlashAttention 后端 [minicpm_backend.py](python/sglang/srt/layers/attention/minicpm_backend.py) 第 ~834 行：

```python
if self.kv_cache_dtype_str.startswith("fp8"):
    if query_layer.dtype != torch.bfloat16:
        query_layer = query_layer.to(torch.bfloat16)  # 隐式反量化 FP8 → BF16
    # 实际注意力矩阵乘运行在 BF16
```

所以**注意力矩阵乘（Q·K^T 然后 softmax-times-V）使用 BF16 张量核心**，即便 K/V 在 HBM 中以 FP8 存储。FP8 存储纯粹是内存/带宽优化；数学运算在 BF16 中。

---

## 2. Path B —— W4A8 #1 误标签（W8A8 FP8 blockwise）

### 2.1 加载时——为什么做 INT4 → BF16 → FP8 而不是像 baseline 一样保留 INT4

[utils_w4a8_fp8.py](python/sglang/srt/layers/quantization/utils_w4a8_fp8.py) `gptq_int4_dequantize` 与 `fp8_blockwise_quantize`，由 [gptq.py](python/sglang/srt/layers/quantization/gptq.py) 第 ~796 行的 `_soar_maybe_setup_w4a8_fp8` 调用：

```python
# 1. 解包 INT4 → INT32
q_unpacked = (qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF  # (K, N) int32
# 2. 反量化：(q - zero) * scale → BF16 稠密
w = (q_unpacked - zeros_full).to(scales.dtype) * scales_full  # (K, N) bf16
# 3. Per-block FP8 量化（128×128 块）
block_amax = w_blocked.abs().amax(dim=(1,3))                  # (N/128, K/128) fp32
weight_fp8_scale = (block_amax / 448.0).to(torch.float32)
w_scaled = (w_f32 / scale_full).clamp(-448, 448)
weight_fp8 = w_scaled.to(torch.float8_e4m3fn)                 # (N, K) FP8 e4m3
```

**为什么这样做**（而不是保留 INT4）：运行时 GEMM 目标是 sgl-kernel 的 `cutlass_w8a8_block_fp8_linear`，它**要求两个输入都已经是 FP8**。SM120 上没有任何稠密 kernel 直接接 INT4 权重 + FP8 激活（这正是要建的 kernel——见 PROPOSAL_W4A8_REAL_001 / 选项 A）。所以加载器**预转换** INT4 权重为现有 FP8 kernel 能消费的格式。

这是**误标签的根本原因**：用 W4 源 checkpoint，反量化，再量化为 FP8 存储仅仅为了喂 FP8 GEMM。kernel 看不到原始 INT4 表示，所以 W4 打包优势在加载时就被破坏。

**转换后的 HBM 占用**（同样 K, N）：

| 张量 | 形状 | 类型 | 字节 |
|---|---|---|---|
| weight_fp8 | (N, K) | float8_e4m3fn | 4096 × 11008 = **~44 MB** |
| weight_fp8_scale | (N/128, K/128) | float32 | 32 × 86 × 4 = **~11 KB** |
| **HBM 合计** | | | **~44 MB** |

对比 Path A：**44 MB vs 23.4 MB。权重 HBM 几乎翻倍。** 这就是导致回退的带宽惩罚。

### 2.2 Forward —— 调用链

```
Linear.forward(x)
  → GPTQMarlinLinearMethod.apply(layer, x, bias)             # gptq.py L871
    if layer._soar_w4a8_active:                              # 门控分支
      → cutlass_w8a8_block_fp8_linear_with_fallback(...)     # fp8_utils.py L343
        → per_token_group_quant_fp8(input_2d, 128, ...)      # 量化激活
        → fp8_blockwise_scaled_mm(q_input, weight.T, x_scale, weight_scale.T, ...)
          → CUDA kernel (CUTLASS FP8 GEMM)
```

forward 中的两个独立步骤：

**A. 激活量化** —— Triton kernel，**每次 forward 都跑**，[fp8_kernel.py](python/sglang/srt/layers/quantization/fp8_kernel.py) 第 ~464 行：
```python
sgl_per_token_group_quant_fp8(
    x,                # (M, K) bf16 输入
    x_q,              # (M, K) FP8 e4m3 输出（已分配）
    x_s,              # (M, K/128) fp32 scales（已分配）
    group_size=128, eps=1e-10, fp8_min=-224, fp8_max=224,
)
```
对每行的每 128 元素组：计算 amax、推 fp32 scale、除、夹紧、cast 到 FP8。这是**单独的 kernel 启动**，自带激活张量的 load/store。

**B. FP8×FP8 GEMM** —— CUTLASS，A（激活）和 B（权重）都已是 FP8：
```cuda
asm volatile(
  "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ..."
  //                                ^^^^^^^^^^^^^^^
  //                                FP8 输入，FP32 累加器
);
```
FP32 累加器 → 在 epilogue 里 cast 到 BF16（与 Path A 同样的 FP32→BF16 步；FP8 核心仍在 FP32 中累加）。

### 2.3 用户提问 —— Marlin INT4→BF16 反量化 vs Path B 的 BF16→FP8 量化：哪个更高效？

虽然两者都涉及类型转换，但**不是同一类操作**：

| | Marlin INT4 → BF16 (Path A) | Path B 激活 BF16 → FP8 |
|---|---|---|
| 何时 | GEMM K 循环**内部** | GEMM **之前**，独立 kernel 启动 |
| 操作数 | 权重（推理期间不变） | 激活（每个 token 都不同） |
| 操作 | 解包 int4 + 减零点 + 乘 bf16 scale | 128 元素的 reduce-max + 除 + clamp + cast |
| 在哪做 | 寄存器，与 MMA 同 SM，完全融合 | 单独 Triton kernel = 独立的 kernel-launch + HBM I/O |
| 每元素成本 | ~3 个简单 int/fp 操作，无 global reduce | reduce + 除 + clamp = ~5 op + 跨 128 元素的 reduction |
| HBM 流量 | 读 INT4（GEMM 本就需要） | 读 bf16 输入（M·K·2 字节）、写 FP8（M·K·1 字节）+ scales（M·K/128·4 字节） |
| 延迟掩藏 | 与 MMA 同 warp 内重叠 | 不重叠——作为独立 kernel 在 GEMM 前跑 |

**结论：Path A 的反量化便宜得多。** 它免费搭乘 K 循环的现有内存流量，并与 MMA 重叠。Path B 的激活量化是**单独 kernel**，付出独立的 HBM round-trip（读 bf16 输入，写 FP8 + scales）和 kernel-launch 延迟。decode bs=1 时 M=1，M·K 激活量化数据量小但仍是非零开销；prefill 时是激活张量上的一次额外 kernel 遍。

这还在 Path B 的 **2× 权重 HBM 带宽**问题之上（Path B 的 FP8 权重 44 MB vs Marlin 的 23 MB）。decode（带宽受限）时，2× 权重带宽是主导惩罚。

### 2.4 用户提问 —— "FP8×FP8 to BF16，没有额外的 FP32→BF16，对吗？"

**不对，FP32→BF16 cast 仍然发生**。硬件 FP8 MMA 始终产出 FP32 累加器；不能从 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` 直接得到 BF16。`row.col` 后面的 `f32` 是**累加器类型**，无法协商。所以 epilogue 仍然做 FP32 → BF16（`cvt.rn.bf16.f32`）再写到 global 内存，与 Path A 完全一样。

GEMM 自身的输出在寄存器中是 FP32；只有**写出**的结果是 BF16。任何路径（BF16 MMA 或 FP8 MMA）都不会跳过 FP32 累加器。

### 2.5 用户提问 —— "对于 KV，由于我们做 FP8×FP8 GEMM，FlashAttention 内不需要 FP8 转 BF16，对吗？"

**不对。** KV 缓存与 linear 层 GEMM 是**针对不同张量的不同操作**：

- **W8A8 FP8 GEMM** 是 linear 投影（如 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj）。其输入是 `linear_input × linear_weight`。这里是 FP8×FP8。
- **注意力**是 `softmax(Q · K^T) · V`。Q、K、V 是 q/k/v 投影**输出的**激活。K 和 V 张量被**存入** KV 缓存。下一个 token 的 Q 与存的 K 做 matmul，再 softmax，再与 V 做 matmul。
- **注意力 kernel**（FlashAttention 后端）是不同于 linear 层 GEMM 的另一个 kernel。它做自己的 MMA。我们的 build 里 FlashAttention 走 **BF16** 的 Q/K/V matmul。
- 所以当 KV 在 HBM 中以 FP8 e5m2 存储时（节省内存），注意力 kernel 在自己的 BF16 matmul 之前仍要**反量化 FP8 → BF16**。这里没有端到端的 FP8 注意力路径。

Path B 保留 `--kv-cache-dtype fp8_e5m2` 不变。W4A8 #1 实现只动了 linear 层，不动注意力 kernel。所以 FlashAttention 内的 FP8→BF16 反量化在 **Path A 和 Path B 中完全一样**。

如果想让注意力真正运行在 FP8 核心上，需要 **FP8 FlashAttention** kernel（如 Hopper 上 cuDNN flash-attention-3 的 FP8 路径，或 Blackwell 变体）。我们目前不用。NVFP4 KV 缓存也不解决这个问题——它仍是仅存储压缩，注意力计算时反量化回 BF16。

---

## 3. 指令级时间线并排对比

一次 linear 层调用，形状 M × K → M × N（如 M=1 decode token、K=14336、N=4096）：

| 步骤 | Path A (Marlin) | Path B (W8A8 FP8) |
|---|---|---|
| 1. 读输入 | (M, K) bf16 from HBM (28 KB) | (M, K) bf16 from HBM (28 KB) |
| 2. 激活预处理 | 无——直通 | **独立 kernel**：读 bf16 (28 KB)、每 128 元素 reduce-max、计算 fp32 scale、除+夹紧、cast 到 FP8、写 FP8 (14 KB) + scales (448 B) → **+~85 KB HBM** |
| 3. 读权重 | Marlin 打包 INT4 (~22.5 MB) + bf16 scales (~700 KB) | FP8 (~44 MB) + fp32 scales (~11 KB) |
| 4. K 循环反量化 | 寄存器内 inline INT4→bf16（免费，与 cp.async 融合） | 无（权重已是 FP8） |
| 5. MMA | `m16n8k16.f32.bf16.bf16.f32` × ⌈K/16⌉ × tile，峰值 148 TF | `m16n8k32.f32.e4m3.e4m3.f32` × ⌈K/32⌉ × tile，峰值 281 TF |
| 6. Epilogue | FP32 累加 → bf16 cast → 写 (M, N) bf16 (8 KB) | FP32 累加 → bf16 cast → 写 (M, N) bf16 (8 KB) |
| M=1 主导成本 | 从 HBM 读 22.5 MB 权重 | 从 HBM 读 **44 MB** 权重（~2× 带宽） |
| M=large 主导成本 | 计算（MMA 吞吐） | 计算（MMA 吞吐，理论 2× 峰值） |

**为什么 Path B 在 decode (M=1) 回退**：带宽受限场景，权重在 HBM 大 2× → GEMM 墙钟约 2× 长。2× MMA 峰值无关，因为 MMA 不是瓶颈。

**为什么 Path B *可能* 在大 M 时赢**：算力受限场景，FP8 MMA 快 2×。但 SOAR 中即便 Smax 也少进入"FP8 MMA 收益能压过 2× 权重带宽劣势"的区间，因为 GPTQ-Marlin 的 W4 权重已经很高效，我们很少完全算力受限。

**为什么真实 W4A8 FP8（PROPOSAL_W4A8_REAL_001 选项 A）会赢**：它会在 HBM 中保持权重为 INT4（无膨胀）**且**用 FP8 MMA。这要求把反量化（INT4 → FP8）写在 K 循环内，就像 Marlin 对 INT4 → BF16 那样，但末尾 FP8 编码而不是停在 bf16。这就是 kernel 项目——3–4 周（见 ANALYSIS_w4a8_fp8_kernel_feasibility）。

---

## 4. 直接回答你的问题（汇总）

**Q1：W8A8 加载为什么做 INT4→BF16→FP8 而不是像 baseline 那样保留 INT4？**
因为现有 FP8 GEMM kernel（`cutlass_w8a8_block_fp8_linear`）要求两个输入都已是 FP8 格式。SM120 上没有任何稠密 kernel 直接接 INT4 权重 + FP8 激活。加载器做转换是变通做法，会破坏 W4 打包优势。真实 W4A8 kernel 会在 K 循环内做 int4→FP8 转换（保留 INT4 HBM 存储）。

**Q2："fp16/bf16 group scales kept" 是什么意思？是否占寄存器/内存？**
意思是 GPTQ checkpoint 的每 128 元素组的 bf16 scale 张量与 int4 权重一起常驻 HBM。HBM 成本：每层 ~700 KB（可忽略）。GEMM 期间寄存器成本：每线程每 K-tile 几个值（可忽略）。保留它们是因为 INT4 量化没有 per-group scale 把它映射回原始实数范围就毫无意义。

**Q3："BF16 epilogue" 是什么意思——把 FP32 转成 BF16？**
是的。张量核心 MMA 始终在 FP32 中累加（硬件约束）。Epilogue 是 MMA 后的步骤，把 FP32 → BF16（如有 bias 也加上）再写到 HBM。

**Q4：Marlin INT4→BF16 反量化 vs Path B 的 BF16→FP8 激活量化——哪个更高效？**
Marlin 的，差距很大。Marlin 的反量化**融合在 GEMM K 循环内**，跑在寄存器中，与 MMA 重叠，免费搭乘已经发生的权重内存流量。Path B 的激活量化是**单独 Triton kernel**，自带 kernel-launch 开销、激活张量的 HBM round-trip、每组 reduce-max + scale-计算步骤。Path B 在 HBM 层面还有额外的 2× 权重带宽惩罚。

**Q5：FP8×FP8 → BF16，没有额外 FP32→BF16，对吗？**
不对——FP32→BF16 cast 仍然发生。PTX MMA 指令是 `mma.sync...f32.e4m3.e4m3.f32`，意思是 **FP8 输入 + FP32 累加器**。硬件不会直接产出 BF16。"BF16 输出"来自 epilogue 的 FP32→BF16 cast，与 BF16 路径完全一样。

**Q6：对于 KV，由于我们做 FP8×FP8，注意力中不需要 FP8 转 BF16，对吗？**
两点都不对：
1. "FP8×FP8" 是 **linear 层** GEMM，不是注意力。注意力 kernel 是另一个 kernel，做 Q·K^T 和 softmax-times-V。
2. 我们的注意力后端不论 KV 存储类型都跑 **BF16**，不是 FP8。所以 FP8 KV 在 FlashAttention 内被反量化到 BF16。要跳过这一步需要 FP8 FlashAttention kernel——我们没有。

---

## 5. 交叉引用

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md) —— 高层成本分析
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md) —— 硬件上限测量
- [PROPOSAL_W4A8_REAL_001.zh.md](PROPOSAL_W4A8_REAL_001.zh.md) —— 真实 W4A8 kernel 应该长什么样
- [gptq.py](../../python/sglang/srt/layers/quantization/gptq.py) 第 520–860 行 —— 两条加载路径
- [utils_w4a8_fp8.py](../../python/sglang/srt/layers/quantization/utils_w4a8_fp8.py) —— 引发误标签的 dequant→requant 代码
- [marlin_template.h](../../sgl-kernel/csrc/gemm/marlin/marlin_template.h) 第 50–120 行 —— Marlin BF16 MMA 指令
- [model_runner.py](../../python/sglang/srt/model_executor/model_runner.py) 第 ~1541 行 —— KV dtype 配置
- [minicpm_backend.py](../../python/sglang/srt/layers/attention/minicpm_backend.py) 第 830–900 行 —— 注意力前的 KV 反量化
