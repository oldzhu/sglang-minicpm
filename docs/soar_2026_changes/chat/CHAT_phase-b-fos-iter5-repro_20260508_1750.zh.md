# CHAT — Phase B FOS iter-5 复现性 + speed bench

开始：2026-05-08 17:50 本地时间。结束：2026-05-08 19:05 本地时间。

## 第 1 轮 — 用户："approve"

用户批准 iter-5 run-2 + S1/S8/Smax speed bench（按 iter-5 配方再量化、跑
精度 + 速度）。详见
[CHANGE_0151_phase_b_four_over_six_005.zh.md](../CHANGE_0151_phase_b_four_over_six_005.zh.md)
下一步 #1 + #3。

## 代理动作

1. `start-instance`（HTTP 200），等 JupyterLab 就绪（~80 s）。
2. `sync` —— 拉到 `cce6b5a7e`，无源码变更。
3. 在 fcloud 按 iter-5 配方再量化：
   `SAMPLES=32 sequential MAX_CALIB_SEQ_LEN=4096 SOAR_QUANT_PROFILE=nvfp4_fos
   SOAR_NVFP4_FOUR_OVER_SIX=1 SEED=20260320`。日志：`pct_m4=43.14%`、
   `tokenizer.save_pretrained complete`。CHANGE_0151_005 的 tokenizer 修复
   一次成功，无需任何手动 `cp`。
4. `restart-server --quant-mode gptq --model-path .../MiniCPM-SALA-NVFP4-FOS
   --env SOAR_QUANT_PROFILE=nvfp4_fos --env SOAR_NVFP4_FOUR_OVER_SIX=1
   --env SOAR_TIER1_LONG_CONTEXT=1 --env SOAR_TORCH_COMPILE_MAX_BS=24`，
   `wait-server` 第二次扫描就绪（torch.compile 预热 ~80 s）。
5. `accuracy`：**70.98%**（vs iter-5 run-1 71.24%，Δ −0.26pt）。耗时
   2513.84 s。
6. `speed --variant all`：**S1=173.83 s，S8=46.05 s，Smax=31.07 s**。
7. `pause-instance`（首次 504，重试成功）。

## 结果

### 精度复现性

iter-5 run-1 / run-2 = 71.24% / 70.98%。**整体可复现（±0.26pt）**，但单任务
方差 ±10pt：

| 任务 | r1 | r2 | Δ |
|------|---:|---:|---:|
| cwe  | 70.67 | 76.00 | +5.33 |
| fwe  | 92.22 | 85.56 | −6.66 |
| mcq  | 53.33 | 46.67 | −6.66 |
| niah | 90.00 | 100.00 | +10.00 |
| qa   | 50.00 | 46.67 | −3.33 |

### 速度对比（NVFP4-FOS vs Test 12 GPTQ 基线）

| 档 | 权重 | NVFP4-FOS | GPTQ（Test 12） | Δ |
|----|------|----------:|----------------:|---:|
| S1   | 40% | 173.83 s | 121.71 s | +52.12 s（NVFP4 慢） |
| S8   | 30% |  46.05 s |  44.09 s |  +1.96 s |
| Smax | 30% |  31.07 s |  35.86 s |  −4.79 s（NVFP4 快） |

正面对比得分：GPTQ 96.0 vs NVFP4 86.7（S1 的 40% 权重决定）。NVFP4-FOS
精度 ~71%（归一化 89% < 97%）→ **C = 0 → 最终得分 0**。
**NVFP4-FOS 不具备提交资格。**

## 决策

1. NVFP4-FOS 留在分支继续研究，不进入提交包。
2. GPTQ 基线（Test 12）继续作为提交基线。
3. 暂停 FOS 标定调优，回到 GPTQ 优化目录。

## 交叉引用

- [CHANGE_0151_phase_b_four_over_six_006.en.md](../CHANGE_0151_phase_b_four_over_six_006.en.md)
- [CHANGE_0151_phase_b_four_over_six_006.zh.md](../CHANGE_0151_phase_b_four_over_six_006.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) 行 `NVFP4-FOS-7`
- 在真实的 FOS=1 量化上验证了 `CHANGE_0151_005` 的 tokenizer 修复

## 未决项 / 后续

1. server 端 `generation_config` 缓解 mcq 失控生成 —— 可恢复 5–10pt（失败
   质量在任务间重新分配）。GPTQ 基线也适用（不修改评测脚本）。
2. NVFP4 S1 延迟剖析 —— 单请求工作负载下相对 GPTQ 多 52 s 开销，可能来自
   kernel launch / dequant / KV lookup。
3. 回到 GPTQ 优化目录优先级（OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md）。
