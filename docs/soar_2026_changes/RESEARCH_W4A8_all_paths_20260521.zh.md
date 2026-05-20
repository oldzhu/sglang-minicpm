# 研究: W4A8 融合 GEMM — 全路径对比（排除两步法）

**日期**: 2026-05-21
**范围**: 对 SM120 (Blackwell) 上实现 >148 TFLOPS 的融合 INT4→FP8 反量化 + GEMM 内核的所有可行方向的详尽调查。两步法已排除（此前验证 +118% 性能倒退）。

---

## 总结

| # | 路径 | TFLOPS | 工作量 | 风险 | 可行？ |
|---|------|--------|--------|------|---------|
| A | **创建 SM120 混合输入 cutlass 构建器** | ~280 | 2-3 周 | 中 | ✅ 最佳 |
| B | **修复原始 PTX 内核 (w4a8_fp8_qmma.cu)** | ~280 | 1-2 周 | 高 | ⚠️ |
| C | **移植 SM100 UMMA 混合输入 → SM120 tcgen05** | ~280 | 1-2 周 | 中 | ✅ |
| D | **CuTeDSL Python 代码生成 → C++ 内核** | ~280 | 1 周 | 高 | ⚠️ |
| E | **优化 wmma（无 tcgen05）** | ~190 上限 | 3-5 天 | 低 | ❌ |
| F | **INT8 反量化 + INT8 IMMA** | ~136 | 3-5 天 | 低 | ❌ |

---

## A: 创建 SM120 混合输入 Cutlass 构建器（最佳路径）

### 已有的
- NVIDIA cutlass: `sm120_blockscaled_mma_builder.inl` (305 行) — 带块缩放的 FP8×FP8
- NVIDIA cutlass: `sm120_mma_builder.inl` — 标准 FP16/BF16/TF32
- sgl-kernel: `sm90_gmma_builder_mixed_input.inl` (280 行) — SM90 混合输入参考
- sgl-kernel: `CollectiveBuilderMixedInput` 桩代码 (48 行)

### 需要做什么
创建新文件 `sm120_mixed_input_mma_builder.inl` (~350-400 行)：
1. 接受 `ElementPairB = cute::tuple<QuantType, ScaleType, ZeroType>`（INT4 + 缩放因子 + 零点）
2. 检测窄位宽操作数（4-bit < 8-bit），通过反量化变换处理
3. 使用 `rr_blockscaled_op_selector_sm120()` 选择 tcgen05 QMMA 指令
4. 为混合输入调整流水线阶段数以适应 SMEM 开销
5. 直接复用现有的 SM120 epilogue（无需修改）

### 关键优势
- 利用所有现有的 cutlass 基础设施：TMA、TMEM、tcgen05.mma、warp 专业化、软件流水线化
- SM120 blockscaled 构建器已正确处理 FP8×FP8 QMMA
- 只需将 INT4→FP8 反量化注入权重加载路径
- 无需原始 PTX — cutlass 自动生成正确的 PTX

### SMEM 预算 (SM120: 101KB)
| 缓冲区 | 大小 | 备注 |
|--------|------|------|
| 权重 (INT4 压缩) | 8 KB | 128×128 × 0.5 B/元素 |
| 权重缩放 + 零点 | ~8 KB | 每组缩放因子 |
| 反量化 FP8 权重 | 16 KB | 如直接送入 MMA 可消除 |
| 激活 FP8 | 16 KB | TMA 加载 |
| TMA 描述符 | ~512 B | 4 个描述符 |
| 流水线缓冲区 | ~32 KB | 双缓冲 |
| Epilogue | ~16 KB | 输出暂存 |
| **总计** | **~96 KB** | 适合 101KB |

---

## B: 修复原始 PTX 内核 (`w4a8_fp8_qmma.cu`)

### 当前状态
- 已编写（440 行，已提交）
- ptxas 编译失败：`tcgen05.alloc` 语法错误，`elect_one.sync` 语法错误
- 无 SMEM swizzling，无双缓冲，无 TMA 全局→SMEM 传输

### 需要修复的内容
1. **TMEM 分配**: 必须使用 `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_ptr], num_cols` — 写入 SMEM 而非寄存器
2. **elect_one.sync**: 语法为 `elect_one.sync p, 0xFFFFFFFF;`（逗号而非管道符）
3. **TMA 描述符**: `fill_tma_desc_2d()` 格式未经验证 — 硬件不匹配风险高
4. **Warp 专业化**: cutlass 使用 6-7 个专门化的 warp；我们的单个 warp-group 可能不满足硬件要求
5. **无 swizzling**: tcgen05 读取 SMEM 时产生 bank 冲突

### 结论
**可行但风险高。** 每个 PTX 语法错误都需要在 fcloud 上进行编译-测试循环（每次约 5 分钟）。仅 TMA 描述符格式就可能需要几天才能正确。如果其中任何一项错误，内核将产生无诊断信息的垃圾输出。

---

## C: 移植 SM100 UMMA 混合输入 → SM120 tcgen05

### 已有的
NVIDIA cutlass 有 `sm100_mixed_input_umma_builder.inl` (~350 行) — 完整的 SM100 混合输入构建器。处理：
- INT4/FP8/FP16 混合位宽操作数
- 主循环中的缩放+零点反量化
- 基于 TMA 的数据加载
- Warp 专业化调度

### 关键区别: SM100 UMMA vs SM120 tcgen05
- SM100 UMMA: 使用 `SM100_MMA_F8F6F4_SS` 等
- SM120 tcgen05: 通过 `rr_op_selector_sm120()` 选择不同指令
- 两者都是 Blackwell！UMMA 构建器可能通过少量修改即可在 SM120 上运行
- SM120 blockscaled 构建器使用 `OpClassBlockScaledTensorOp`，而混合输入使用 `OpClassTensorOp`

### 结论
**非常有希望。** SM100 混合输入构建器已处理所有 INT4→FP8 反量化逻辑。移植到 SM120 主要需要将 MMA 指令选择器从 UMMA 改为 tcgen05，并调整流水线阶段数以适应 SM120 较小的 SMEM。

---

## D: CuTeDSL Python → C++ 代码生成

### 已有的
NVIDIA cutlass 有 SM120 混合输入 GEMM 的 Python CuTeDSL 示例：
`examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/`

### 工作原理
这些 Python 脚本使用 CuTe DSL 在高级别描述内核。框架生成与 cutlass 一起编译的 C++ CUDA 代码。自动处理 TMA、TMEM、tcgen05.mma、warp 专业化。

### 可行性
- **优点**: 自动生成正确 PTX — 无语法错误
- **缺点**: 需要 CuTeDSL Python 环境（cutlass 构建依赖、MLIR 等）
- **缺点**: 生成的代码与 cutlass Python 框架紧密耦合
- **缺点**: 难以集成到 sglang 的 cmake 构建系统

### 结论
**对我们的构建流水线来说过于复杂。**

---

## E: 优化 wmma 内核（无 tcgen05）

### 可能的优化（不使用 tcgen05）
| 优化 | 收益 | 工作量 |
|---|---|---|
| SMEM 双缓冲（K-tile 重叠） | +10-20% | 中 |
| `cp.async` 全局→SMEM | +5-10% | 中 |
| 更大 tile（减少启动开销） | +5% | 低 |
| 更好的占用率 | +0-5% | 低 |
| **组合上限** | **~190 TFLOPS** | — |

### 为什么不够
- 基线 Marlin: 148 TFLOPS 理论值
- 优化 wmma: 最高约 190 TFLOPS
- tcgen05 QMMA: ~280 TFLOPS（实测）
- **与 tcgen05 的差距: ~90 TFLOPS（高 47% 吞吐量）**

---

## F: INT8 反量化 + INT8 IMMA

### 实测 SM120 上的 INT8 吞吐量
来自 `PHASE0_INT8_vs_FP8_SM120_20260427_1630.en.md`：
- INT8 IMMA: ~136 TFLOPS（与 BF16 相同）
- FP8 QMMA: ~275 TFLOPS（INT8 的 2 倍）

### 为什么不用
SM120 上的 INT8 tcgen05 上限与 BF16 相同。FP8 QMMA 是此硬件上突破 200 TFLOPS 的唯一路径。

---

## 实施计划: 路径 A（推荐）

### 步骤 1: 参考资料研究 (1 天)
- 通读 NVIDIA cutlass `sm100_mixed_input_umma_builder.inl`
- 阅读 sgl-kernel `sm90_gmma_builder_mixed_input.inl`
- 理解 INT4 反量化如何注入到主循环中

### 步骤 2: 创建 SM120 混合输入构建器 (3-5 天)
- 新文件: `sm120_mixed_input_mma_builder.inl`
- 按照 SM90 混合输入模式进行操作数检测和反量化设置
- 按照 SM120 blockscaled 构建器进行 tcgen05 指令选择
- 调整 SMEM 预算（SM120 101KB vs SM90 232KB）

### 步骤 3: 集成 (1 天)
- 添加到 `collective_builder_mixed_input.hpp`: `#include` 新的 SM120 构建器
- 添加新内核文件（类似 `fp8_blockwise_gemm_kernel.cu` 但用于 W4A8）
- 在 `common_extension.cc` 注册 torch op

### 步骤 4: 构建和测试 (2-3 天)
- 在 fcloud 上完整构建 sgl-kernel wheel（需要 cutlass FetchContent）
- 正确性测试: 单 tile → 多 tile → 模型维度
- 速度测试: S1/S8/Smax 基准测试

### 预计总计: 7-10 天

---

## 备选: 路径 C（移植 SM100 UMMA）

如果路径 A 的 blockscaled 操作类带来复杂性（OpClassBlockScaledTensorOp vs OpClassTensorOp），回退到移植 SM100 UMMA 混合输入构建器。使用更接近 SM90 混合输入模式的 `OpClassTensorOp`。

## 参考资料

- `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu` — 原始 PTX 内核（编译受阻）
- `sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu` — 可工作的 wmma 内核
- `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` — 可工作的 SM120 FP8 GEMM
- `sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl` — SM90 参考
- `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` — SM90 W4A8 分组
- NVIDIA cutlass: `sm100_mixed_input_umma_builder.inl` — SM100 混合输入
- NVIDIA cutlass: `sm120_blockscaled_mma_builder.inl` — SM120 blockscaled
