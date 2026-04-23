# 提案：修订版 Iteration A — 运行时 / 服务参数层面的加速调优

**日期**：2026-04-23
**替代**：[STRATEGIC_ROADMAP_TOP5.zh.md](STRATEGIC_ROADMAP_TOP5.zh.md) 中原有的 Iteration A（Marlin SM120 decode tile 专门化）
**状态**：提案 — 在获得用户批准前不做任何代码改动

---

## 1. 为什么必须放弃原始 Iteration A（Marlin tile 专门化）

[CHANGE_0125_sm120_marlin_tiles_001.zh.md](CHANGE_0125_sm120_marlin_tiles_001.zh.md) 的深度分析已经证明：继续为 Marlin 增加 tile 实例化，对 MiniCPM-SALA 在 SM120 上毫无帮助：

1. **评分函数的物理规律**：SM120 Marlin 评分中 `fill_ratio × 1000` 项正确地偏好窄 tile（`thread_n=64`）。MiniCPM-SALA 的 `N ∈ [1024, 28672]` 下窄 tile 会产生 72–448 个 tile，可以填满全部 96 个 SM；而宽 tile（`thread_n=256`）只能产生 18–112 个 tile，会让 78+ 个 SM 闲置。评分是数学正确的，而不是调优 bug。
2. **指令集不匹配**：Marlin 使用 SM80 时代的 `mma.sync.aligned.m16n8k16`。调整 tile 根本无法调用 SM120 的原生 warp 级 MMA、TMA 或 QMMA，因此硬件 2× 吞吐优势不可达。
3. **实测验证**：Test 27（CHANGE_0125）扩充了 tile 表后，三档速度均为 0 变化。

**结论**：想从 SM120 拿到真实增益，必须**替换 Marlin**，而不是重排 Marlin。那是以周为单位的工程（CUTLASS SM120 集成或 SM120 原生 MMA 重写），超出本轮 iteration 范围。

## 2. 修订版 Iteration A — 范围与目标

**目标**：在**不改核函数**的前提下，用服务参数 / 运行时调优榨出现有 kernel 栈剩余的免费吞吐。

**预期增益**：S1 2–5%、S8 2–5%、Smax 0–3%（幅度不大，但免费、确定、可回滚）。

**C 影响**：中性（纯配置变化，不改变数学）。

**对最终得分的杠杆**：如果速度提升 4%，performance score 按乘法约提升 4%，当前 team-beta 分数 39.62 → 约 41.2，大约 +1.6 分。幅度有限但是真实收益，而且能**打扫出一个干净的 runtime baseline**，为 Iteration C（投机解码）提供可解释的测试环境。

## 3. 建议变更（打包）

全部变更仅涉及 `benchmark/soar/demo_sala/prepare_env.sh`（GPTQ 分支的 `SGLANG_SERVER_ARGS`），**不改源码**。

### Change A-1：去掉 `--enable-torch-compile --torch-compile-max-bs 8`

**证据**：Test 33（commit `e625363a8`）用相同配置但关闭 torch.compile，实测：
- 服务启动：36s vs 开启 compile 时 219s —— **省 183s**
- 全量评测总耗时：3016s vs 3005s —— **差异 0.4%，在噪声内**
- 各档速度：torch.compile 在该工作负载下几乎 0 加速

torch.compile 在动态 shape 的 prefill 上会发生 graph break；而 sglang 自带的 CUDA graph 捕获已经覆盖 decode。该编译在 MiniCPM-SALA / SM120 下纯粹是额外开销。

**风险**：未观察到；Test 33 已验证。

### Change A-2：将 `--cuda-graph-max-bs` 显式设为 24

**理由**：当前隐式默认值可能没有覆盖到 `--max-running-requests 24` 的全部 decode 批量。如果任何 decode 批量命中不了 graph，就会落到 eager 路径（单步大约慢 15–20%）。显式让 `cuda-graph-max-bs = max-running-requests` 可以保证 decode 全覆盖。

**风险**：每个额外捕获的批量会多占几百 MB 显存。`--mem-fraction-static 0.84` 已经留出余量；如 warmup 阶段出现 OOM，将回退到 16。

### Change A-3：加上 `--stream-interval 2`

**理由**：默认 stream-interval=1，每 decode 一个 token 就触发一次 detokenize / send。设为 2 能把这部分开销减半，代价只是 streaming 稍有延迟。本次评测看的是完成时延，不是逐 token 时延，因此没有可见代价。

**风险**：可忽略，纯服务端批处理旋钮。

### Change A-4（可选，在 A-1..3 通过后再做）：恢复 `prefill-max-req=4, sched-cons=0.8`

**证据**：Test 25A-spd 用此组合把 S1 从 120.48s 降到 110.58s（**−8.2%**），当时没有精度回退。后来在精度 bisect（Test 29+）中回退，但 bisect 也证明精度噪声与调度无关。

**顺序**：A-1..3 干净落地后再单独作为一次测试，单独隔离其影响。

**风险**：较小，可回滚。

## 4. 实施计划（获批后才执行）

1. 修改 `benchmark/soar/demo_sala/prepare_env.sh` GPTQ 分支：
   - 去掉 `--enable-torch-compile --torch-compile-max-bs 8`
   - 加入 `--cuda-graph-max-bs 24 --stream-interval 2`
2. 提交到 `mixed_minicpm_cudagraph`，push 到 `minicpm-src`。
3. 请用户启动 fcloud，然后跑完整流水线：
   - `sync` → `restart-server --quant-mode gptq` → `wait-server`
   - `accuracy`（确认 C ≥ 0.92）
   - `speed --variant all`
   - `shutdown`
4. 结果记入 [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md)，标记为 **Test 35**。
5. 决策门：
   - S1 ≥ 3% 加速 **且** C ≥ 0.96：晋升为 v20 提交候选，继续做 Change A-4。
   - 速度中性：仅保留 A-1（省启动时间），回退 A-2/A-3。
   - 精度跌到 C=0：整包回滚（纯配置还原，成本极低）。
6. 若后续执行 A-4（记为 Test 35b）：对比 Test 35，如果 S1 再降 ≥5% 且 C 保持，就固化。

## 5. 验证命令

```bash
# 本地
git diff benchmark/soar/demo_sala/prepare_env.sh

# fcloud（sync 之后）
grep 'SGLANG_SERVER_ARGS=' /root/submission_sim/prepare_env.sh | head -2
```

速度参考（当前 fcloud 的 Test 34a 数据）：
| 档位 | Test 34a | 预期（本轮后） | 目标 |
|---|---|---|---|
| S1 | 108.91s | ≤ 106s | ≤ 105s |
| S8 | 39.99s | ≤ 39s | ≤ 38.5s |
| Smax | 33.44s | ≤ 33.5s | ≤ 33s |

精度门槛：C ≥ 0.92（normalized ≥ 97.01%）。更高是加分项，不是硬要求。

## 6. 回滚

```bash
git checkout benchmark/soar/demo_sala/prepare_env.sh
git push minicpm-src mixed_minicpm_cudagraph --force-with-lease
```
（纯配置回滚，不需要重建 wheel）

## 7. 本轮不包含

- **不改核函数**。CUTLASS SM120 / SM120 原生 MMA / QMMA-fp8 GEMM 推迟到 Iteration C（投机解码）确定范围后再单独提案。
- **不引入新量化路径**。FP8 blockwise（`OPTION_B-final`）已经在精度上失败；W4A8 Marlin 变体也需要核函数改动。
- **不改 draft model**。投机解码（C1 n-gram、C2 EAGLE 训练）作为下一个主要 iteration。

## 8. 为什么先做本轮，再做 Iteration C（投机解码）

- A-1..3 近乎免费（一次 prepare_env 编辑、一次 commit、一轮测试）。
- 去掉 torch.compile 的混淆后，后续投机解码的基准比较会更可解释。
- 启动时间省 3 分钟，在官方 5 小时封顶的 quant+eval 预算里是真收益。
- 完全可逆，如果后面 Iteration C 需要不同的 runtime 形状，随时可改。

---

## 待用户确认的问题

1. 是否批准 A-1..3 打包一起做？
2. A-4 是和 A-1..3 合并到一次测试，还是单独再跑一次？
3. 是否同意关掉 torch.compile？（Test 33 证据足够，但如果用户有关于官方评测环境的背景，请告知。）
