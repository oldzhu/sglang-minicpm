# 提案 — 1 天 CUTLASS W4-FP8 spike 验证 SM120 上 FP8 上限

**日期**：2026-04-27 17:30
**状态**：仅提案。**尚无代码改动。** 等待用户批准。
**前置**：`ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.{en,zh}.md`（完整 kernel = 3–4 周）。

## 1. 目标与预期收益

**降风险**：花 1 天测一下基于 CUTLASS 的 W4-FP8 参考实现在 SM120 上能否接近稠密 FP8 上限（Phase 0 实测 281 TF）。若可，3–4 周投入有真实上限。若不可（例如反量化开销把它压在 <200 TF），优先级即定，不浪费 3–4 周。

**这不是生产 kernel。** 是测量 spike——合成数据基准，找 W4-权重 × FP8-激活形状的 FP8 上限。

### 1.1 与 Phase 0（已完成）的区别

Phase 0（`PHASE0_INT8_vs_FP8_SM120_20260427_1630`）通过 `torch._scaled_mm` 测的是**稠密 FP8×FP8 → BF16 GEMM**。MMA 发射时两个输入已经在寄存器里是 FP8——K 循环中没有发生反量化。那个数字是**硬件上限 = 281 TF**。

Proposal B 测的是不同的东西：**W4-权重 × FP8-激活 GEMM**，INT4 权重从 HBM 以打包形式加载，在 K 循环中**反量化**（mask + shift int4 → 乘 fp16 group scale → 编码为带饱和的 FP8 e4m3 比特模式）后才馈送进同一条 MMA 指令。反量化链路在跟 MMA 同一批 SM 上跑，可能串行化或阻塞 warp。Proposal B 的数字告诉我们 **281 TF 上限中多少层能在反量化税后存活**——这正是决定 3–4 周生产 kernel 投资是否值得的关键未知量。

| 测试 | MMA 输入 | 衡量 | 数字 |
|---|---|---|---|
| Phase 0（已） | FP8 × FP8（无反量化） | 硬件 FP8 上限 | 281 TF |
| Proposal B（本） | INT4 打包 → 内核反量化 → FP8 × FP8 | W4→FP8 反量化后的 FP8 上限 | TBD |

## 2. 规则合规检查

不适用——这是一天的**测量**任务。无模型改动、无提交包改动、无精度风险。纯粹合成数据基准。

## 3. 风险

- 零精度/稳定性风险（无生产改动）。
- 风险：1 fcloud 小时的 spike 可能"无定论"。
- 缓解：预先定义清晰的通过/失败阈值（见 § 6）。

## 4. 实施计划

### 4.1 方法
以 CUTLASS 3.x 例子为起点。CUTLASS 有：
- `examples/65_distributed_gemm` 和 `examples/55_hopper_int4_fp8_gemm/`（Hopper SM90 W4-FP8 参考——最接近的现有实现）。
- 对 SM120：基于 SM90 例子改 `arch::Sm90` → `arch::Sm120` 并使用 SM120 兼容 MMA atom（Blackwell warp 级 MMA，**非** warpgroup）。

### 4.2 步骤（~1 fcloud 工作日）

| # | 步骤 | 时间 |
|---|---|---|
| 1 | 在 CUTLASS 子模块（或上游 clone）中找 `examples/55_hopper_int4_fp8_gemm/` | 0.5 h |
| 2 | 写自包含 `bench_w4fp8_sm120.cu` 调用 CUTLASS 模板，形状 `M=16384, N=14336, K=4096`（匹配 Phase 0 矩阵） | 2 h |
| 3 | fcloud 上 `nvcc -arch=sm_120 -O3 -std=c++17 -lcudart` 编译 | 1 h |
| 4 | 若 SM90 atom 在 SM120 上编译失败：回退手写最小 W4→FP8 反量化 kernel，直接调 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`，输入合成 INT4 权重 + FP8 激活 | 2 h |
| 5 | 跑 100 次迭代，计算中位 TFLOPS，记日志 | 0.5 h |
| 6 | 记录结论到 `RESULT_W4_FP8_CUTLASS_SPIKE_<date>.{en,zh}.md` | 1 h |
| | **总计** | **~7 h，单个 fcloud session** |

### 4.3 文件（全部临时，不入 sglang）
- `/root/bench_w4fp8_sm120.cu`（仅 fcloud）
- `/root/run_w4fp8_spike.sh`（仅 fcloud）
- 本地文档：`docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_<date>.{en,zh}.md`（提交到仓库）

## 5. 验证命令（fcloud）

```bash
# （用户启动 fcloud）
# Agent 通过 fcloud_exec.py 上传 bench_w4fp8_sm120.cu
ssh fcloud "cd /root && nvcc -arch=sm_120 -O3 -std=c++17 bench_w4fp8_sm120.cu -lcudart -o bench_w4fp8 && ./bench_w4fp8"
```

## 6. 通过/失败标准

| M=16384,N=14336,K=4096 实测 TFLOPS | 决策 |
|---|---|
| **≥ 250 TF**（稠密 FP8 281 TF 的 ≥ 89%） | **绿灯**。W4-FP8 上限为真。后续作为迭代立项。 |
| **180–249 TF** | **黄灯**。FP8 杠杆存在但 ~30% 峰值被反量化吞掉。vs NVFP4 KV 边际收益——延后。 |
| **< 180 TF** | **红灯**。反量化开销主导。**永久终止** W4-FP8 稠密。 |

## 7. 回滚

不适用——无生产改动。

## 8. 下一步建议

- 绿灯：把 W4-FP8 完整 kernel 排在 NVFP4 KV 之后的迭代。
- 黄灯/红灯：从优化路线图移除 W4-FP8；专注 NVFP4 KV + 目录其他项。

## 9. 交叉引用

- [ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md](ANALYSIS_w4a8_fp8_kernel_feasibility_20260427_1730.zh.md)
- [PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md](PHASE0_INT8_vs_FP8_SM120_20260427_1630.zh.md)
- [PROPOSAL_NVFP4_KV_CACHE_20260427_1730.zh.md](PROPOSAL_NVFP4_KV_CACHE_20260427_1730.zh.md)
- CUTLASS 上游：https://github.com/NVIDIA/cutlass `examples/55_hopper_int4_fp8_gemm/`
