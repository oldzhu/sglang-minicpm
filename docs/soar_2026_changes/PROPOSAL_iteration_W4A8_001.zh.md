# 提案 — Iteration W4A8 (#1)：通过 TRT-LLM 启用 FP8 in-kernel GEMM（基于 v18 基线）

**日期**: 2026-04-27 09:05
**状态**: ⏳ 等待用户批准 — 暂未做任何代码改动
**优先级**: 后 v18 优化清单 #1
**基线 (v18-revert，等价 Test 25)**: S1=110.51s, S8=40.46s, Smax=33.61s, 归一化精度 77.44 % → C=0.92
**前置文档**: `PROPOSAL_fp8_w4_dequant_gemm.md` (Phase 1 调研, 2026-04-21)、`PROPOSAL_option_b_fp8_blockwise_gemm.en.md` (sgl-kernel 内置替代方案)。
**本提案在 v18 基线锁定后取代以上两份**，作为可执行计划。

---

## 1. 背景与动机

依据 `CLARIFICATION_quant_layers_runtime_verify_20260427_0811`：

- 当前 GPTQ + Marlin 路径权重以 INT4 存储，但 **kernel 内部反量化为 BF16**，再以 **BF16 × BF16** 跑 BF16 tensor core (SM120 上 148 TFLOPS)。
- SM120 的 **FP8 QMMA tensor core 峰值 296 TFLOPS** — 正好是 2× — 当前完全闲置。
- 依据最新 `RESEARCH_mixed_arch_speed_optimization_20260426_1526`，在长上下文官方 speed 数据集上 GEMM 主导 prefill 时间。
- 因此把 GEMM 计算迁到 FP8（权重保持 W4 存储），直接拿到设备上最大的一块未利用算力。

这就是 "**W4A8**"：W = 4-bit 存储，A = 8-bit 激活，FP8 乘累加。

## 2. 规则合规检查

| 规则 | 状态 | 说明 |
|---|---|---|
| Submission ≤ 2 GB | ✅ | TRT-LLM Python wheel ≈ 100-200 MB，在限额内 |
| 现场量化 | ✅ | 权重格式转换在 `preprocess_model.py` 里跑（不提交预量化权重） |
| ≤ 5 小时准备时间 | ✅ | 转换是矩阵 reshape + per-row scale fold；分钟级 |
| Apache 2.0 + 可复现 | ✅ | TRT-LLM 是 Apache-2.0；autotuned tactic 一次性记录 |
| 精度 C ≥ 0.96 (目标 C=1.0) | ⚠️ | FP8 e4m3 有 3 mantissa + 4 exp；类似模型文献中激活量化误差 ≤ 0.5 %。缓解：保留 BF16 fallback；用环境变量 gate |
| 并发档位 flag `--max-concurrent {1,8,∞}` 不变 | ✅ | 不改调度 |

## 3. 精度 / 稳定性风险

**已识别风险**：
1. **激活异常值**：FP8 e4m3 在 ±448 处饱和。MiniCPM-SALA 的 SiLU、softmax 输出可能尖刺。缓解：per-token 动态激活 scale（TRT-LLM kernel 默认）。
2. **Lightning 层交互**：24 个 lightning 层在 decode 步间乘性叠加误差。缓解：**v1 只把 FP8 GEMM 用在 8 个标准注意力层 + MLP**；lightning 的 Q/K/V/O 仍走 Marlin BF16，验证安全后再单独立项升级。
3. **fcloud 环境的 TRT-LLM wheel 兼容性**：第 1 步先验证可 pip 安装。
4. **Scale 转换正确性**：GPTQ group_size=128 的 BF16 scale 必须折成 per-row FP8 scale。缓解：写单元测试，比较转换前后反量化权重的 Frobenius 差 < 1e-3。

**警戒点**：启用后立刻在 `perf_public_set.jsonl` 上重测精度。如归一化精度跌破 79 %（v18 是 77.44 %，留 ~1.5 pt 安全垫），通过环境变量关闭 FP8 path，按失败处理。

## 4. 改动文件

### 4.1 新代码（小面积）

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/layers/quantization/gptq_marlin.py` | forward 里加分支：若 `os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"` 且 TRT-LLM 可用且层在白名单（std-attn QKV/O + MLP gate/up/down）→ 调 TRT-LLM kernel；否则走原 Marlin |
| `python/sglang/srt/models/minicpm.py` | 构造时给可用层打标签（linear forward 里读这个属性）— v1 排除 lightning Q/K/V/O |
| `benchmark/soar/demo_sala/preprocess_model.py` | GPTQ 量化后，把 per-group BF16 scale 折成 per-row FP8 scale 并写入 safetensors（新增 `weight_fp8_scale` tensor）。环境变量关闭则跳过 |
| `benchmark/soar/demo_sala/prepare_env.sh` | 追加 `export SOAR_W4A8_FP8_GEMM=1` 与 `pip install tensorrt-llm-blackwell-min==<version>`（具体 wheel 名第 1 步确定后填入） |

### 4.2 不需改动

- 所有 attention backend (FlashInfer / FA / SimpleGLA) — 它们不直接调 GEMM。
- Lightning recurrent state 路径 — 不变（与本提案正交，见提案 #3）。
- 评测脚本 `eval_model_001.py` — 不变。
- Server args 结构 — 不变。

## 5. 详细实施计划（改动前 — 供 review）

```
Step 1: 验证 fcloud 上 TRT-LLM wheel 可用
   └─ ssh fcloud → pip install tensorrt-llm... (仅测试，不提交)
   └─ python -c "import torch; torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell"
   └─ 报告 wheel 大小 + 启动 import 开销
   └─ 不可用则放弃此 iteration，回退到 Option B (sgl-kernel 内置)

Step 2: 写 scale 转换工具 (preprocess_model.py)
   └─ 对每个有 .qweight + .scales (group=128, BF16) 的 linear:
        per_row_max = scales.float().amax(dim=group_dim)
        per_row_fp8_scale = per_row_max / 448.0
        weight_fp8_scale = per_row_fp8_scale (per output row)
   └─ 单元测试：用原 BF16 scale 与新 FP8 scale 各反量化一次 INT4，Frobenius 差 < 1e-3。

Step 3: Linear forward 分支 (gptq_marlin.py)
   └─ if SOAR_W4A8_FP8_GEMM and self.allow_fp8 and trtllm_available:
          input_fp8, input_scale = quantize_per_token_fp8(input)
          out = torch.ops.trtllm.cute_dsl_fp8_gemm_blackwell(
              input_fp8, self.weight_fp8, input_scale, self.weight_fp8_scale,
              output_dtype=torch.bfloat16, use_tvm_ffi=True)
      else:
          out = existing_marlin_path(...)

Step 4: 在 MiniCPM 构造里给层打白名单
   └─ Std-attn QKV (sparse + non-sparse): 可用
   └─ Std-attn O proj: 可用
   └─ MLP gate/up/down: 可用
   └─ Lightning Q/K/V/O: v1 不可用 — 仍走 BF16 Marlin
   └─ lm_head: 仅在归一化精度通过后启用，否则继续 BF16
```

## 6. 验证命令

### 正确性（合并前必跑）
```bash
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
# 期望：归一化精度 ≥ 79 %（v18 是 77.44，保留 ≥1.5pt 安全垫）
```

### 速度（正确性通过后）
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### 运行时验证（CLARIFICATION 文档里的方法）
- **V4 (必跑)**：`ncu --section ComputeWorkloadAnalysis ...` 现在应能在 GEMM kernel 上看到**非零的 `sm__inst_executed_pipe_tensor_op_qmma`**。这是最关键的运行时信号——若依然为 0，就说明并未真正用上 FP8 硬件。
- **V1 (便宜)**：确认白名单层加载了 `weight_fp8` 与 `weight_fp8_scale` tensor。

## 7. 预期结果（baseline vs 新）

| 指标 | v18 baseline | W4A8 后保守预期 | W4A8 后乐观预期 |
|---|---|---|---|
| S1 | 110.51 s | 102-105 s (-5 %) | 95-100 s (-10 %) |
| S8 | 40.46 s | 36-38 s (-7 %) | 33-35 s (-13 %) |
| Smax | 33.61 s | 30-32 s (-8 %) | 27-29 s (-15 %) |
| 归一化精度 | 77.44 % | 76-79 % (噪声内) | 76-79 % |
| C 系数 | 0.92 | 0.92 (目标) | 0.92 |
| 最终分数（线性比例） | 参考 | +5-8 % | +10-15 % |

保守值用于规划；乐观值是 TRT-LLM 达到 >80 % FP8 峰值时的上限。

## 8. 回滚

```bash
# 源码回滚
git revert <merge-commit>
git push minicpm-src mixed_minicpm_cudagraph

# 运行时回滚（无需 revert）
unset SOAR_W4A8_FP8_GEMM   # 在 prepare_env.sh 里
# 下次 server 启动时自动走 Marlin BF16 path
```

环境变量 gate 让我们可以**不重新量化模型**就关掉 FP8 GEMM。`weight_fp8` tensor 留在 safetensors 里但不被读取。

## 9. Iteration 范围控制

按 copilot 规则（"一次只做一个改进特性"），本 iteration 仅包括：
- 标准注意力 + MLP 层的 W4A8 FP8 GEMM
- preprocess 里的 scale 转换
- 环境变量 gate

**不包括**：
- Lightning state FP8（提案 #3，下一轮）
- Lightning fused kernel（提案 #4，下一轮）
- KV cache FP4（M2.0，独立）
- 投机解码（#6，下一轮）

测试通过后，以上各自单独立项。

## 10. 批准合约 — 改代码前请确认

请确认或修订：

1. **范围**：是否同意 v1 仅限标准注意力 + MLP（lightning Q/K/V/O 仍 BF16）？（推荐）
2. **路径选择**：TRT-LLM wheel (Option A，低工作量) 还是 sgl-kernel 内置 `fp8_blockwise_scaled_mm` (Option B，无额外 wheel)？（推荐 A）
3. **环境变量 gate**：`SOAR_W4A8_FP8_GEMM=1` 启用，验证前默认关闭？（推荐）
4. **精度保险**：本地 public set 归一化精度跌破 79 % 即终止？（推荐）
5. **先做 Step 1**：在改本仓库代码之前，先在 fcloud 上验证 TRT-LLM wheel 可装？（推荐）

确认（或修订）后，计划是**仅执行 Step 1**，把 wheel 可用性结果汇报回来，再继续。

## 11. 交叉引用

- `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md` — 论证今日是 BF16-on-BF16-tensor-core（动机来源）
- `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md` — 把本项列为 #1
- `PROPOSAL_fp8_w4_dequant_gemm.md` — 原 Phase-1 TRT-LLM 调研 (kernel API、scale 转换草图)
- `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` — Fallback (无额外 wheel) 用 sgl-kernel
- `ANALYSIS_nvfp4_offline_quant_20260427_0857.{en,zh}.md` — 解释为什么本步**不**走 FP4 权重
