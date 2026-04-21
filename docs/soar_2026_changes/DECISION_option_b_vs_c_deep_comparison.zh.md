# 决策文档：选项B vs 选项C — 深度完整对比
## MiniCPM-SALA GPTQ 的 SM120 FP8 GEMM 路径分析

**日期**: 2026-04-21  
**状态**: 选项B已选定立即执行；选项C文档化供未来深度优化参考  
**基准**: GPTQ (sparse_qkv_w8) + FP8 KV cache + dense 模式 — S1=121.71s, S8=44.09s, Smax=35.86s  
**参考硬件**: SM120 RTX PRO 6000 — 593 TFLOPS FP8, 296 TFLOPS BF16, 1398 GB/s

---

## 背景：我们为何来到这里

性能分析（CHANGE_0120）揭示：
- **GEMM 占 prefill 时间的 85.3%，decode 时间的 63.5%**
- 当前 Marlin GPTQ W4 内核使用 SM80 `mma.sync.aligned.m16n8k16` — 无法访问 SM120 FP8/FP4 TFLOPS
- SM120 FP8 硬件提供的 **TFLOPS 是 BF16 的 2 倍**，是 INT8 模拟路径的 **4 倍**
- 关键瓶颈：compute-bound 的 prefill 浪费了 400+ TFLOPS 的 SM120 硬件算力

第一阶段调研（PROPOSAL_fp8_w4_dequant_gemm.md）评估了 TRT-LLM，确认 CUTLASS 路径可行。决策矩阵（DECISION_fp8_w4_implementation_options.md）否决了选项A（TRT-LLM wheel 太大），选定了 B 和 C。

### 关键发现：sgl-kernel 中已存在 SM120 FP8 GEMM

在调研选项B时，我们发现 **sgl-kernel 已经有完整可用的 SM120 FP8 blockwise GEMM**：

```
文件: sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu
函数: sm120_fp8_blockwise_dispatch_shape<OutType>(out, a, b, scales_a, scales_b)
宏保护: #if defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
Python接口: fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype)
```

SM120 内核使用 CUTLASS 3.x：
- `cutlass::arch::Sm120` 架构标签
- `MmaTileShape = Shape<128, 128, 128>`（使用 SM120 UMMA warp 级 MMA）
- `ScalesPerTile = Shape<128, 1, 1>` → A 矩阵每行一个 scale（每128个K元素），B 矩阵每 128×128 块一个 scale
- **A 的 scale 格式**: `(M, K/128)` — 每行每 K-block-128 一个 scale
- **B 的 scale 格式**: `(K/128, N/128)` — 每 K-block 每 N-block 一个 scale

这意味着选项B **无需编写任何 CUDA 内核** — 只需做权重转换和模型加载更改。

---

## 两条路径对比

### 选项B：使用现有 SM120 内核的 FP8 Blockwise GEMM

**核心思路**：
1. `preprocess_model.py`：加载 GPTQ W4 模型 → 反量化 W4 为 FP16/BF16 → 重新量化为 FP8 blockwise 格式（M×128-K-块，128×128-N-块）
2. 以新格式存储 FP8 权重 + 块 scale
3. `linear.py`（模型代码）：检测到 FP8 权重 → BF16 激活量化为 FP8（每行每 K-block）→ 调用 `fp8_blockwise_scaled_mm` → 输出 BF16
4. 无需修改 sgl-kernel — 内核已为 SM120 编译

**变更内容**：
| 组件 | 变更 |
|------|------|
| `preprocess_model.py` | 添加 `gptq_to_fp8_blockwise()` 线性层转换 |
| `python/sglang/srt/layers/linear.py` | 添加 FP8 blockwise 分发路径（`MiniCPMFP8Linear`） |
| `python/sglang/srt/models/minicpm.py` | 检测到 FP8 权重时使用 `MiniCPMFP8Linear` |
| `benchmark/soar/demo_sala/prepare_env.sh` | 添加 FP8 线性路径的服务器参数 |
| 无 CUDA 文件改动 | — |

**scale 转换（GPTQ group_size=128 → blockwise 128×128）**：
- GPTQ 权重 scale：形状 `(K/128, N)`（每组每列）
- 内核需要：`scales_b` 形状 `(K/128, N/128)`（每 K-block 每 N-block）
- 转换方法：对每个 K-组内的每 128 列 N-block：
  - 将 128×128 的 W4 块反量化为 FP16
  - 找最大绝对值 → 计算 FP8 scale = max_abs / 448.0
  - 量化为 FP8 E4M3 = block / scale
- 内存：W4（0.5B/参数）→ FP8（1B/参数）= 权重内存 2×，但约 9B 参数 → ~9GB（84GB 完全够用）
- 激活量化：每行每 128 个 K 元素 → scale = max(abs(x_row_block)) / 448.0

**为什么选 blockwise（128×128）而非 per-tensor 或 per-token**：
- FP8 E4M3 动态范围有限（约 ±448）
- Per-tensor：单一 scale 导致大方差被截断 → 精度严重损失
- Per-token A × per-column B：需要 (M) + (N) 个 scale → 但 SM120 UMMA 需要对齐的块 scale
- 128×128 blockwise：与 UMMA tile 大小对齐，精度/速度权衡最优

**预期性能**：
| 指标 | 当前（Marlin W4） | 选项B后 | 提升 |
|------|-------------------|---------|------|
| TFLOPS 利用率 | ~100-140 BF16 有效 | ~350-450 FP8 | 2.5-3.5× |
| Prefill GEMM 时间 | 基准 | 减少 25-35% | ↑ |
| 端到端 S1 | 121.71s | 估计 ~85-100s | ↑ |
| 端到端 S8 | 44.09s | 估计 ~30-35s | ↑ |
| 权重内存 | ~4.5GB（W4 9B 模型） | ~9GB（FP8 9B 模型） | 2× |

注：decode 提升受内存带宽限制（1398 GB/s 上限）。compute-bound 的 prefill 受益最多。

**时间**: 3-5 天  
**风险**: 低-中（内核已经过验证；权重转换和精度需要测试）

---

### 选项C：自定义 SM120 W4A8 融合反量化 GEMM 内核

**核心思路**：
编写新的 CUDA 内核：
1. 直接从内存加载 W4 GPTQ 权重（不反量化 — 保持 ~4.5GB 内存占用）
2. 在寄存器中：使用 GPTQ 组 scale 将 W4 反量化为 FP8
3. 使用 SM120 UMMA `tcgen05.mma.ws.sync` 指令处理 FP8 操作数
4. 以 FP32 累加，BF16 输出

这样在保持 W4 内存占用的同时利用 SM120 FP8 计算吞吐量。

**为什么这很难**：
- 必须编写自定义 CuTe mainloop，加载 W4 并为 UMMA 生成 FP8
- GPTQ 权重打包：8 × INT4 值打包在每个 INT32 中，带有每组反量化
- SM120 UMMA 需要特定的内存对齐和寄存器布局（每 warp lane 4 个寄存器组）
- 必须实现 TMA-based 预取来隐藏 W4 数据的延迟
- 必须处理 GPTQ group_size=128 的 scale 查找，避免 bank conflict
- 无法使用现有 CUTLASS CollectiveBuilder 做融合反量化（不是支持的路径）
- 需要用 CuTe 汇编级原语完整实现自定义 mainloop

**变更内容**：
| 组件 | 变更 |
|------|------|
| `sgl-kernel/csrc/gemm/w4a8_sm120/` | 新建：约 2000-3000 行 CUDA 内核 |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_w4a8_gemm_kernel.cu` | 主内核 |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_w4a8_tma_prefetch.cuh` | TMA 辅助工具 |
| `sgl-kernel/csrc/gemm/w4a8_sm120/gptq_dequant_fp8.cuh` | 反量化寄存器工具 |
| `sgl-kernel/python/sgl_kernel/gemm.py` | 添加 `gptq_w4a8_fp8_mm()` 绑定 |
| `python/sglang/srt/layers/linear.py` | 分发到新内核 |
| 需要完整重建 sgl-kernel | 首次约 4 小时，增量约 3 分钟 |

**内核设计挑战**：

1. **W4 TMA 内存布局**：GPTQ 将权重存储为 `(K/8, N)` 打包的 INT32（每 INT32 8 个 INT4）。TMA 需要对齐的、非步进的张量。需要为打包的 W4 自定义 TMA 描述符。

2. **寄存器内反量化流水线**：
   ```
   // 对每个 128×128 tile：
   // 1. TMA 加载 128×16 打包 INT32 tile（等价于 128×128 INT4 权重）
   // 2. 查找组 scale：scales[k_group, n]，tile 内所有 n
   // 3. 解包 INT4 对，减去零点，乘以 scale → FP8
   // 4. 在共享内存中暂存 FP8，供 UMMA 使用
   // 5. 运行 tcgen05.mma.ws.sync FP8×FP8 → FP32
   ```

3. **UMMA 寄存器布局**：SM120 UMMA（warp 级 MMA）需要特定寄存器布局。CuTe 抽象有所帮助，但自定义 mainloop 意味着手动管理布局。

4. **流水线深度**：需要对 TMA 加载（W4）+ 反量化 + UMMA 进行软件流水线，以充分利用 SM120 L1 带宽。流水线差 = 占用率差 = 性能差。

5. **精度**：W4 的 FP8 反量化路径引入额外量化噪声。必须验证归一化精度保持 > 99% 以获得 C=1.0。

**预期性能**：
| 指标 | 当前（Marlin W4） | 选项C后 | 提升 |
|------|-------------------|---------|------|
| TFLOPS 利用率 | ~100-140 BF16 有效 | ~400-500 FP8 | 3-4× |
| Prefill GEMM 时间 | 基准 | 减少 30-45% | ↑ |
| 权重内存 | ~4.5GB W4 | ~4.5GB W4（不变！） | 相同 |
| 端到端 S1 | 121.71s | 估计 ~75-95s | ↑ |
| 端到端 S8 | 44.09s | 估计 ~28-33s | ↑ |

**时间**: 4-8 周  
**风险**: 高

---

## 正面对比表

| 维度 | 选项B：FP8 Blockwise | 选项C：W4A8 自定义内核 |
|------|---------------------|----------------------|
| **内核编写量** | 无（使用现有内核） | 约 2000-3000 行 CUDA |
| **时间** | 3-5 天 | 4-8 周 |
| **权重内存** | 2×（W4 → FP8） | 不变（保持 W4） |
| **Prefill TFLOPS** | 350-450 FP8 | 400-500 FP8 |
| **Prefill 加速** | GEMM 约 2.5-3.5× | GEMM 约 3-4× |
| **Decode 加速** | <10%（受带宽限制） | <10%（受带宽限制） |
| **精度风险** | 低（FP8 经过验证） | 中等（新反量化路径） |
| **构建风险** | 无 | 高（UMMA, TMA, 流水线） |
| **调试风险** | 低 | 高 |
| **正确性验证** | 简单（现有测试） | 需要数值测试 |
| **依赖风险** | 无 | 无（都在 sgl-kernel 中） |
| **提交包大小** | wheel 略大 | 相同或略大 |
| **SOAR 合规** | ✅ | ✅ |

---

## 按推理阶段的性能分析

### S1（单请求，max-concurrent=1）— 权重 40%

**Prefill 阶段**（长提示词时主导）：
- 大批量/序列长度时受计算限制
- 选项B：GEMM 组件（占 85% 时间）约 2.5-3× 加速 → 整体 prefill 减少 50-60%
- 选项C：GEMM 约 3-4× 加速 → 整体 prefill 减少 60-70%
- Δ（C vs B）：prefill 选项C好 10-15%

**Decode 阶段**（短提示词/流式输出时主导）：
- 受内存带宽限制：1398 GB/s 上限
- FP8 权重 = 1 字节 → 需要加载的数据比 W4（0.5 字节）多 2×
- 选项B：**decode 略慢**于 W4（权重占用更多带宽）
- 选项C：**decode 与当前相同**（W4 不变，无额外带宽开销）
- Δ（C vs B）：decode 选项C 明显更好

### S8（8 个并发请求）— 权重 30%

**Prefill 阶段**：有效批量更大 → 更受计算限制
- 选项B：prefill 减少 55-65%
- 选项C：prefill 减少 65-75%

**Decode 阶段**：8 个请求 × decode 步骤 → 带宽 × 8
- 选项B：每个请求 decode 加载 FP8 权重 → 仍受带宽限制，但每步 2× 更多
- 选项C：保持 W4 → 与基准相同带宽

### S∞（无限并发）— 权重 30%

**Prefill 主导**：大批量 → 强受计算限制
- 选项B：峰值收益，prefill 减少 60-70%
- 选项C：prefill 减少 70-80%
- 两个选项在 S∞ 都表现优秀

---

## 建议

### 立即执行：选项B

**理由**：
1. SM120 FP8 内核已存在 → 零 CUDA 风险
2. 3-5 天实现时间适合 SOAR 竞赛节奏
3. 预期在 S1（权重40%）和 S∞（权重30%）上获得明显更高分
4. 对精度影响风险低（FP8 blockwise 已有充分研究）
5. 为进一步优化留有时间

### 未来：选项C（如有必要）

**何时启动C**：
- 选项B部署并获得评分后
- 如果距离前5名的差距需要额外 >20% 的提升
- 如果B后性能分析显示 decode 成为新瓶颈（W4 用于 decode 避免了 FP8 的 2× 带宽开销）
- 如果竞赛时间允许 4+ 周的内核工作

**如何高效推进C**：
1. **从 CuTe mini-kernel 开始**（1 SM，小 tile，无流水线）验证正确性
2. **添加 TMA 预取**用于 W4 加载（第2阶段）
3. **添加多级流水线**（第3阶段）
4. **针对 MiniCPM-SALA 形状调优 tile 大小**（第4阶段）

选项C的关键参考：`sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`（现有 W4 反量化逻辑），`sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`（SM120 UMMA 模板）。

---

## 选项B后续步骤（实现计划）

详见 `PROPOSAL_option_b_fp8_blockwise_gemm.zh.md` 完整实现细节。

**高层步骤**：
1. 在 `preprocess_model.py` 中添加 `gptq_to_fp8_blockwise()` — 将 GPTQ 层转换为 FP8 + block scales
2. 在 `linear.py` 中添加 `FP8BlockwiseLinear` 模块 — 每行每块激活量化 + `fp8_blockwise_scaled_mm`
3. 在 `minicpm.py` 中根据权重 dtype 条件使用 FP8 linear
4. 在本地评估上验证精度（`eval_model_001.py`）
5. 在 fcloud 上进行速度基准测试（S1, S8, Smax）

**测试需要回答的关键问题**：
- FP8 blockwise 权重的归一化精度是否保持 > 99%？（目标：C=1.0）
- SM120 上 MiniCPM-SALA 形状实际 TFLOPS 利用率是多少？
- 2× 权重带宽导致的 FP8 decode 开销是否损害 S1（可能是 decode 主导）？

---

## 选项C参考实现说明

供未来实现选项C的工程师参考，关键代码：

**现有 W4 反量化逻辑**（适配为 FP8 输出）：
- `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu` — GPTQ 反量化逻辑
- `sgl-kernel/csrc/gemm/gptq/qdq_4.cuh` — INT4 打包/解包工具

**SM120 UMMA 设置**（参考模板）：
- `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` 第 206-353 行 — 完整 SM120 内核模板
- `cutlass::arch::Sm120`, `cute::UMMA::Major::MN`

**W4 权重的 TMA**（非标准 — 打包 INT32）：
- 需要为 `(K/8, N)` 打包布局自定义 `cute::Tensor` 描述符
- 参考：CuTe 非标准布局教程

**目标内核结构**：
```cuda
// 选项C 伪代码
__global__ void gptq_w4a8_sm120_gemm_kernel(
    int8_t* A_fp8,          // (M, K) FP8 激活（每 token 预量化）
    int32_t* B_w4_packed,   // (K/8, N) GPTQ 打包 W4
    float* B_scales,        // (K/128, N) GPTQ 组 scale
    int8_t* B_zeros,        // (K/128, N) GPTQ 零点
    float* A_scales,        // (M, K/128) 激活块 scale
    bfloat16_t* C,          // (M, N) 输出
    int M, int N, int K
) {
    // 第1阶段：TMA 预取 128×128 W4 tile（= 128×16 int32）
    // 第2阶段：使用 GPTQ scale 在 smem 中解包+反量化为 FP8
    // 第3阶段：tcgen05.mma.ws.sync fp8×fp8 → fp32 累加器
    // 第4阶段：通过 TMA 存储 fp32 acc + scale → bf16 输出
}
```

**MiniCPM-SALA 形状的预期 tile 配置**：
- 标准线性层：K≈4096-8192，N≈4096-16384
- 最佳 tile：M=128, N=128, K=128（一个 GPTQ 组恰好适合 K 维度）
- SM 占用率目标：最少 2 个 wave（96 SM 需要 192 个 tile）

---

## 结论

选项B是当前正确的选择：
- **零 CUDA 内核风险** — 使用现有 SM120 FP8 内核
- **3-5 天实现和测试**
- **预计 S1/S∞ 加速 30-50%**（compute-bound prefill 主导）
- 选项C已在此文档化，并提供了清晰的实现指南供未来参考

两个选项均完全符合 SOAR 规则（无禁止技巧，无外部依赖，可复现）。
