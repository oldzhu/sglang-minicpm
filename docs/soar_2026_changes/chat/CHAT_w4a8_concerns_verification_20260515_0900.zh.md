# CHAT: W4A8 顾虑解答与验证 — 2026-05-15

## 1. 用户提问

1. 要求全面讨论最佳配置后再决定下一步："what is our best config so far and what is our best config for nvfp4 fos so far and let us dicuss more on this."
2. 讨论后提问："so our next improving plan better be based on Best GPTQ (Marline W4A16) Baseline, do u think real W4A8 could be direction to make our rank back to top 30?"
3. 对 W4A8 提出两个具体顾虑：
   - 顾虑 1：FP8 精度风险（尾数位少于 BF16）
   - 顾虑 2：旧 W4A8 路径倒退（S1 +118%）可能预示真 W4A8 也会慢
4. 要求将顾虑分析文档化，并在 fcloud 上执行 W4A8 验证测试。

## 2. Agent 回答/操作

### 最佳配置总结
- **最佳 GPTQ 基线**：v24（当前 HEAD，提交 `8ce03caaf`）—— 本地 S1=110.54s、S8=40.32s、Smax=33.48s。官方：pre-v18 得分 30.04（#22），v23 得分 25.55（#31）。本地准确率在 75-79% 区间波动，±3pt 方差，mcq runaway 为主导因素。
- **最佳 NVFP4-FOS**：FOS-7 准确率 70.98%，norm≈88.73% → 始终 C=0。速度 S1=173.83s（比 GPTQ 慢 57%）、Smax=31.07s（快 13%）。已搁置 —— 不具备提交资格。
- **策略**：W4A8 是最高 ROI 事项。预估有效速度提升 15-20% → 得分 ~29-30（回到 #25-30）前提是准确率稳定。

### 顾虑分析（记录于 PROPOSAL_W4A8_REAL_002）
- **顾虑 1（FP8 精度）**：FP8 尾数位 3 对比 BF16 的 7，但逐 token 缩放可补偿精度。旧 W8A8 测试（两端均为 FP8）准确率中性（79.20% vs 79.29%）。真 W4A8 仅改变激活端 → 风险更低。缓解措施：Phase 0 MSE 微基准 + 完整准确率门槛（C=1.0 否则放弃）。
- **顾虑 2（旧倒退预示速度差）**：旧路径慢是因为 INT4 权重在加载时上转为 FP8（HBM 权重流量翻倍）。真 W4A8 保留 INT4 权重存储 → 零带宽惩罚。真 W4A8 在每个维度上均等于或优于基线。

### fcloud W4A8 验证测试
- 启动 fcloud 实例，同步代码，覆盖 `SOAR_W4A8_FP8_GEMM=1`
- 服务启动并进行 FP8 转换（观察到全部 32 层的 [SOAR W4A8] enabled 日志）
- CUDA graph 捕获耗时 ~5 分钟（bs 1-24，7 buckets）
- 速度基准测试完成：

| 级别 | W4A8 FP8 GEMM=1 | v24 基线 | Δ |
|---|---:|---:|---:|
| S1 | **267.19s** | 110.54s | **+142%** |
| S8 | **68.85s** | 40.32s | **+71%** |
| Smax | **45.86s** | 33.48s | **+37%** |

- **旧路径倒退已确认。** 当前 Tier1/long-context 配置无关 —— 权重带宽惩罚仍占主导。
- 已还原环境覆盖并暂停实例。

## 3. 结果

- **决策**：进入真 W4A8 内核开发（INT4 存储 + FP8 MMA），按照 `PROPOSAL_W4A8_REAL_001` Option A。
- **验证完成**：旧路径倒退以同等幅度确认，无意外。
- **文档创建**：`PROPOSAL_W4A8_REAL_002_concerns_and_verification.{en,zh}.md`
- **TEST_RESULTS_TRACKING.md 更新**：新增 W4A8#1-reverify 行。

## 4. 交叉引用

- `PROPOSAL_W4A8_REAL_002_concerns_and_verification.en.md` — 顾虑分析 + 验证计划
- `PROPOSAL_W4A8_REAL_002_concerns_and_verification.zh.md` — 中文版
- `TEST_RESULTS_TRACKING.md` — 新增 W4A8#1-reverify 行
- `PROPOSAL_W4A8_REAL_001.en.md` — 真 W4A8 内核设计（Option A：Machete 风格 FP8 QMMA）
- 提交 `8ce03caaf` — v24 基线（mcq cap 已关闭）
- 提交 `7ce21c3f5` — 旧 W8A8 FP8 路径（已废弃）

## 5. 待解决问题 / 后续步骤

- 进入真 W4A8 内核开发？（3-4 天工作量，需用户批准）
- 并行轨道：修复 mcq 准确率稳定性（更大 cap 16384+ 或重复检测器）？
