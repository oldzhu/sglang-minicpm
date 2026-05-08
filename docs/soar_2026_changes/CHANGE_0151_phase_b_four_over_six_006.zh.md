# CHANGE 0151 — Phase B FourOverSix，续篇 006

承接 [CHANGE_0151_phase_b_four_over_six_005.zh.md](CHANGE_0151_phase_b_four_over_six_005.zh.md)。

第 7 轮 = **iter-5 复现性验证 + speed bench**：以完全相同的 iter-5
NVFP4-FOS 配方（FOS=1，SAMPLES=32 sequential，calib_seq_len=4096，默认
qa,mcq,cwe 过滤）重新量化，跑 run-2 与 iter-5 跳过的 S1 / S8 / Smax 三档
速度。

本轮也是 CHANGE_0151_005 引入的 tokenizer 修复（提交 `39c0045c5` +
`83921b207`）首次在 FOS=1 的成功量化上验证（iter-6 用的是 FOS=0）。

## 背景 / 动机

iter-5 只有一个 71.24% 的精度数据点，没有 run-2，也没有 speed。iter-6
（FOS=0 消融）确认了 FOS 在短题任务上具有保护作用，且 iter-5 与 iter-1 之间
~1.89pt 差距并非 FOS flag 本身造成。决定 NVFP4-FOS 是否进入提交包前的两个
开放问题：

1. iter-5 的 71.24% 是否可复现？FOS-1 → FOS-1b 的 −5.71pt 大幅波动表明
   单点不能信任。
2. iter-5 配置下的 S1 / S8 / Smax 是多少？提交决策需要这组数据与当前 GPTQ
   基线对比。

## 实施

无源码改动。复用 iter-5 配方再量化，跑 run-2 与 speed。

```bash
# (1) 重新量化（确定性 seed）
SOAR_QUANT_PROFILE=nvfp4_fos \
SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
SOAR_GPTQ_CALIBRATION_SEED=20260320 \
python3 -u preprocess_model.py \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/MiniCPM-SALA-NVFP4-FOS \
  --mode nvfp4

# (2) 重启 server（Tier1 long-ctx，modelopt_fp4）
SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_TIER1_LONG_CONTEXT=1 SOAR_TORCH_COMPILE_MAX_BS=24 \
python3 -m sglang.launch_server --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  ${SGLANG_SERVER_ARGS[@]}

# (3) accuracy + speed
fcloud_workflow.py accuracy --quant-mode gptq --model-path .../MiniCPM-SALA-NVFP4-FOS
fcloud_workflow.py speed --variant all
```

量化日志确认 FOS 已激活、pct_m4 一致：
```
[preprocess] NVFP4 FourOverSix activating for export
[preprocess] NVFP4 FourOverSix summary: layers=224 blocks=521142272
  blocks_picked_m4=224801387 pct_m4=43.14%
[preprocess] NVFP4 tokenizer.save_pretrained complete
```

tokenizer-save 修复一次成功，无需任何手动 `cp`。

## 结果

### 精度（run-2 复现性）

| 任务 | iter-5（run-1） | iter-5（run-2） | Δ |
|------|---------------:|---------------:|---:|
| cwe  | 70.67% | 76.00% | +5.33 |
| fwe  | 92.22% | 85.56% | −6.66 |
| mcq  | 53.33% | 46.67% | −6.66 |
| niah | 90.00% | 100.00% | +10.00 |
| qa   | 50.00% | 46.67% | −3.33 |
| **平均 ori_accuracy** | **71.24%** | **70.98%** | **−0.26** |
| 评测耗时 | 2485.20 s | 2513.84 s | +28.64 s |

run-2 长度桶：len_0_4k 46.67%，len_4k_32k 62.25%，len_32k_128k 84.46%。

**复现性结论**：整体精度稳定（±0.26pt）—— iter-5 可复现。**但单任务方差很
大**（单任务两次相差 ±10pt），这与历史上调度 / 思考-回答失控的方差模式一致。
两次整体精度差不到 0.3pt 但单任务波动 ±10pt，说明失败分布大致守恒（失控
生成的概率质量在任务间重新分配，但总错题数稳定）。

### Speed bench（S1 / S8 / Smax）

| 并发 | iter-5 NVFP4-FOS（run-2 ckpt） | NVFP4-FOS-1b（方差探针） | Test 12 GPTQ 基线 |
|------|------------------------------:|------------------------:|------------------:|
| S1   | 173.83 s | 175.08 s | 121.71 s |
| S8   |  46.05 s |  47.37 s |  44.09 s |
| Smax |  31.07 s |  31.01 s |  35.86 s |

NVFP4-FOS 相对 Test 12 GPTQ 基线的速度特征：
- **S1 慢 +52.12 s**（+42.8%）
- **S8 慢 +1.96 s**（+4.4%）
- **Smax 快 −4.79 s**（−13.4%）

这是 SM120 上 NVFP4 的预期画像：FP4 权重加载 + dequant 开销在 S1 占主导
（短 prefill + 长尾 latency），但在高并发长上下文 prefill 计算密集时，FP4
GEMM 带宽优势体现在 Smax。

## 与 GPTQ 提交基线（Test 12）对比

按 `Final = S1×40% + S8×30% + Smax×30%`、各档以
`Duration_best / Duration_player × 100` 评分：

| 档 | 权重 | Test 12 GPTQ | NVFP4-FOS（iter-5 r2） | NVFP4 比率 |
|----|------|-------------:|----------------------:|-----------:|
| S1   | 40% | 121.71 s | 173.83 s | 70.0%（−12.0pt 加权得分） |
| S8   | 30% |  44.09 s |  46.05 s | 95.7%（−1.3pt） |
| Smax | 30% |  35.86 s |  31.07 s | 115.4%（NVFP4 占优） |

若以 NVFP4-FOS 提交：GPTQ 得分 = 100×0.4 + 100×0.3 + 35.86/35.86×100×0.3 =
**100**；NVFP4 得分 = 70.0×0.4 + 95.7×0.3 + 100×0.3 = **86.7**；GPTQ 在 Smax
档对 NVFP4 最佳的得分 = 100×0.4 + 100×0.3 + (31.07/35.86)×100×0.3 = **96.0**。

正面对比：
- GPTQ 总分占优（96.0 vs 86.7），因为 S1 的 40% 权重起决定作用。
- 仅当 **Smax** 权重超过 ~60% 时 NVFP4 才会反超 —— 即只在「高并发长上下文
  prefill 主导」的负载上有优势。

加上 NVFP4-FOS 当前精度天花板（~71%，C=0.92 上限 —— 归一化精度 89% < 97%
**不达标，最终得分 0**），**NVFP4-FOS 当前不具备提交资格**。GPTQ +
sparse_qkv_w8 + FP8 KV 仍是提交基线。

## 决策

1. **NVFP4-FOS 不进入提交包**（被精度天花板限制）。保留在分支上继续研究
   （见下一步）。
2. **GPTQ 基线（Test 12）继续作为提交包**，沿其方向迭代速度 / 精度。
3. iter-5 现已确认可复现：两次跑 70.98% / 71.24%，整体方差 ±0.26pt。在没有
   突破 ~71% 上限的路径前，暂停 FOS 标定调优。

## 回滚

本轮无代码变更。CHANGE_0151_005 的 tokenizer-save 修复已通过此轮新量化
验证，无需回滚。

## 下一步

| # | 想法 | 预期 | 工作量 | 风险 |
|---|------|------|--------|------|
| 1 | 排查 NVFP4 在短题 mcq 的失控生成：server 端 `generation_config`（降低 temperature/top-p、stop tokens）—— 当前 mcq 平均输出 7000+ token | 若能控制失控可恢复 mcq 5–10pt | 中（不能影响长上下文任务） | 中 |
| 2 | profile NVFP4 S1 的 52s 开销来源（kernel launch、dequant、KV lookup）；可能修复：pin 权重、缩减 torch_compile graph 集合 | NVFP4 路径可行性数据 | 1–2h | 低 |
| 3 | 回到 GPTQ 基线优化（按 OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md）。当前精度差距下杠杆更高 | 提交包累计速度收益 | 视项目 | 低–中 |
| 4 | 考虑在 Phase C（BF16 标定 + NVFP4 导出）之前彻底搁置 NVFP4 路径 | 资源分配清晰化 | – | – |

建议：**(3) 最具杠杆**，因为 NVFP4-FOS 当前被精度封锁。搁置 NVFP4-FOS，
回到 GPTQ 目录优先级。
