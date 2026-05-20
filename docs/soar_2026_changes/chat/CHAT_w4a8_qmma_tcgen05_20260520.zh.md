# 聊天记录：W4A8 QMMA tcgen05 内核实现

**日期**: 2026-05-20
**主题**: Path B — 融合 INT4→FP8 反量化 + SM120 tcgen05 QMMA 内核（方案3）
**提交**: `3d619d81d`（已推送到 `minicpm-src`）

## 用户问题/操作

1. "让我审查并讨论下一步。审查期间请暂停 fcloud 实例。"
2. "我记得我们之前测试过阶段一……如果在加载时将所有权重 INT4→FP8 反量化，基准测试结果大约比 marlin 基线差 2 倍"
3. 用户确认阶段 B1（两步法）是死胡同——跳过。
4. "方案2和3，哪个更好？各自的优缺点"
5. "选方案3，要求详细的文档和源代码注释，以便编码后我能完整审查和修改。"

## Agent 操作

1. 暂停了 fcloud 实例（控制台 504，回退到 shutdown）
2. 调查了先前的 W4A8 测试结果：
   - 确认 W4A8#1 两步法：S1 +118% 倒退（权重带宽翻倍）
   - v25 声称 −9% 但无法复现；是每次前向的两步法，不是融合内核
3. 分析了方案2 vs 方案3：
   - 方案2（cutlass SM120 扩展）：1500+ 行模板代码，2-3 周，无 NVIDIA 先例
   - 方案3（原始 PTX tcgen05）：~300 行，3-5 天，复用现有的反量化框架
   - 推荐方案3
4. 从 NVIDIA/cutlass `sm100_umma.hpp` 研究了 tcgen05.mma PTX 格式：
   - 找到 SS 变体：`tcgen05.mma.cta_group::1.kind::f8f6f4 [tmem_c], desc_a, desc_b, ...`
   - 找到 TS 变体、tcgen05.ld、tcgen05.st、tcgen05.commit 模式
   - 理解了 TMEM 要求（Blackwell 张量内存）
   - 理解了 TMA 描述符要求（用于 SMEM 访问）
5. 编写了新内核：`sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu`（~340 行）
   - SMEM 中 INT4→FP8 反量化（列优先，与旧内核相同）
   - FP8 激活加载（行优先）
   - 手动 TMA 描述符构建（fill_tma_desc_2d）
   - tcgen05.mma SS — warp-group QMMA，296 TFLOPS
   - Epilogue: tcgen05.commit → tcgen05.st → 每线程 BF16 存储
   - 大量的内联注释解释架构、风险、线程映射
6. 创建了独立构建的 CMakeLists.txt
7. 将新内核添加到 sgl-kernel/CMakeLists.txt
8. 编写了双语文档：CHANGE_W4A8_QMMA_TCGEN05.{en,zh}.md
9. 更新了 TEST_RESULTS_TRACKING.md，添加了 v26 占位符
10. 提交并推送到 minicpm-src

## 关键设计决策

- **手动创建 TMA 描述符** — 无 cutlass 依赖。风险：描述符格式可能不正确。
- **无 SMEM swizzling** — 简单的行/列优先布局。可能存在 bank 冲突。
- **无双缓冲** — 顺序 K-tile 处理，带屏障。
- **TMEM 未释放** — 对于每 SM 单个块的启动可接受。
- **相同的 Python 接口** — `torch.ops.w4a8_fused.w4a8_fp8_fused_gemm` 不变。

## 已知风险

| 风险 | 严重程度 | 状态 |
|------|----------|--------|
| TMA 描述符格式错误 | 高 | 未在硬件上验证 |
| SMEM bank 冲突 | 中 | 无 swizzling |
| tcgen05.st 子 tile 映射不正确 | 中 | 假设 8×16 线程网格 |
| TMEM 泄漏 | 低 | 每块启动可接受 |

## 后续步骤（待用户执行）

1. 启动 fcloud 实例并同步代码
2. 使用 `-arch=sm_120a` 构建独立 .so
3. 测试正确性（先单 tile 128×128，再多 tile）
4. 如果编译失败 → 调整 TMA 描述符或添加 cutlass 头文件
5. 如果输出错误 → TMA 描述符格式是主要怀疑对象
6. 如果正确但慢 → 添加 swizzling、双缓冲
7. 服务器集成 + 速度基准测试

## 交叉引用

- `docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.en.md`
- `docs/soar_2026_changes/CHANGE_W4A8_QMMA_TCGEN05.zh.md`
- `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu`
- `sgl-kernel/csrc/gemm/CMakeLists_standalone.txt`
- `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`（v26 条目）
