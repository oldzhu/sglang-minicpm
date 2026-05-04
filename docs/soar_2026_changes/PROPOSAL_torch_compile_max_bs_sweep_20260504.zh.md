# 提案 #2-A — 在 v21 之上做 `--torch-compile-max-bs` 扫描

**日期**：2026-05-04
**作者**：Agent（待用户批准）
**前置版本**：v21（`SOAR_TIER1_LONG_CONTEXT=1` 默认开启，提交 `edf97175e`）
**目录引用**：`OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` Tier-2
"torch.compile 在更高 batch size 下的图覆盖"
**风险**：低（环境变量门控，默认关闭）

## 1. 目标

v20/v21 出厂值 `--torch-compile-max-bs 8`，意味着编译后的 CUDA graph 仅覆盖
batch ≤ 8 的解码。在 Smax 档（max-running-requests=24，无并发上限）时，
runtime 对 `bs ∈ [9, 24]` 退回 eager 模式。把这些更高 batch 桶也编译进去
预计能在 Smax 档（这是官方长上下文速度分主要来源）带来双位数百分比的
解码吞吐提升。

假设：在 v21 完全相同的服务参数足迹上，把 `--torch-compile-max-bs 8 → 16`
（可能再到 24）将图覆盖扩展到 bs=9-16 区段，正好是 Smax 解码真正花时间
的位置。

## 2. 合规性

- **允许**：`--torch-compile-max-bs N` 是 sglang 上游标准开关；不涉及自定义
  内核改动；不私下重启 prefix cache；不修改提交模板。
- **评测接口**：未变。`/root/data/eval_model_001.py` 不动。
- **并发档位**：未变。继续按官方 harness 的 `--max-concurrent {1,8,inf}` 跑。
- **提交约束**：tarball ≤ 2GB；不引入额外权重。

## 3. 准确率 / 稳定性风险

| 风险 | 缓解 |
|------|------|
| 服务启动编译时间从 ~214s 增长到 ~400-600s（每个桶都要编译） | 可接受。官方启动器会等 `/health`。我们会测量。 |
| 静态显存增加（更多捕获的图）→ Smax OOM | `--mem-fraction-static 0.84` 应仍能容纳；若 OOM 则回退到 bs=12 或撤销。用户提到此前 bs=16/24 曾 OOM，但本机 fcloud 的硅与 runtime 已不同。 |
| 某些 bs 值的 torch.compile 桶 bug 静默破坏输出 | 速度测试**之前**先跑准确率；要求 ≥ 78%（不低于 Test 29 底线）才上线。 |
| 任何 sglang / torch 升级会让编译缓存失效 | 提交线上不存在此问题——wheel 已固定版本。 |

## 4. 改动文件（极小）

仅 `benchmark/soar/demo_sala/prepare_env.sh` —— 把硬编码的
`--torch-compile-max-bs 8` 替换为环境变量门控值。

```bash
# 在 gptq 分支（~187 行）：
SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs ${SOAR_TORCH_COMPILE_MAX_BS}"
```

- 默认 = 8（v21 字节级等价）。
- `SOAR_TORCH_COMPILE_MAX_BS=16` → fcloud 测试通过则作为 v22 候选发版。
- `=24` → 可选的天花板探测。

（无其他代码改动；无模型 / 内核 patch。）

## 5. fcloud 测试详细计划

待用户批准并完成一行 `prepare_env.sh` 修改与推送之后：

| # | 步骤 | 预期 |
|---|------|------|
| 1 | `start-instance` | 任务启动 |
| 2 | `sync` | 拉取 patch |
| 3 | `restart-server --env SOAR_TORCH_COMPILE_MAX_BS=16` | server 启动；启动时间预期 300-500s；通过 cmdline 验证 `--torch-compile-max-bs 16` |
| 4 | `accuracy` | acc ≥ 78%（不低于 Tier1-B 的 78.73%）|
| 5 | `speed --variant all` | 记录 S1/S8/Smax |
| 6 | 若 OK：再次 `restart-server --env SOAR_TORCH_COMPILE_MAX_BS=24` + accuracy + speed | 可选天花板 |
| 7 | `pause-instance` | 完成 |

fcloud 时间预算：bs=16 单独约 80 分钟，若再测 bs=24 总约 150 分钟。

## 6. 成功 / 失败判据

| 结果（相对 Tier1-B 基线 78.73 / 111.36 / 40.49 / 33.62）| 决策 |
|---|---|
| acc ≥ 78% **且 Smax ≤ 32s**（≥4% 提升）| 发版 v22；#2-A 完成。 |
| acc ≥ 78% 且 Smax 在 [32, 33.5] | 仍发版（小幅但免费）。 |
| acc ≥ 78% 且 Smax ≥ 33.5（无提升）| 不发版；记录；前往 #3。 |
| acc < 78% 或启动 > 800s 或 OOM | 撤销默认值；在 TEST_RESULTS_TRACKING 中记录失败模式。 |

注：本地 Smax 输入很短（≤1K tokens）；Smax 解码时 bs 通常 ~8-12，因此本地
sweep **可以**度量本项（与 Tier 1 chunked-prefill 不同）。

## 7. 回滚

`export SOAR_TORCH_COMPILE_MAX_BS=8`（或不设）→ v21 字节级等价。一个 env
旋钮；非常简单。

## 8. 后续建议

- 如果 #2-A 胜出：开 #2-B = `bs=16` + 把 `--max-running-requests` 从 24 提到 32
  （Smax 时更多并发请求）。
- 如果 #2-A 中性：转向 #3 = prefill GEMM 内核方向（Marlin M=2048 tile path；
  根据 `R13e-prof-128k` profile 数据，BF16 GEMM 占 128K context 内核时间的 76.9%）。
- 如果 #2-A 退化：把 torch-compile 整体作为 Tier-2 死路停掉，#3 提前。

## 9. 交叉引用

- v21 基底：提交 `edf97175e`、[PROPOSAL_tier1_long_context_retest_20260430.zh.md](PROPOSAL_tier1_long_context_retest_20260430.zh.md)、[TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) 的 `Tier1-A-baseline` / `Tier1-B-candidate` 行。
- 优化目录：[OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)。
- 128K 性能剖面证据：`profile_data/round13e_analyze.txt`（BF16 GEMM 76.9%）。

---

## 待批准

回复 **approve** 即应用一行 `prepare_env.sh` 修改、推送，并执行 §5 的
fcloud 流程。回复 **adjust** 给出不同的 bs 上限（或跳过 bs=24 探测）。
