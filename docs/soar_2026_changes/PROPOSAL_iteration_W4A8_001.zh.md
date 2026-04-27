# 提案 — Iteration W4A8 (#1)：通过 TRT-LLM 启用 FP8 in-kernel GEMM（基于 v18 基线）

**日期**: 2026-04-27 09:05  
**状态**: ✅ 2026-04-27 已批准（含修订：选 Option B，不引入 TRT-LLM wheel，软性精度守门）— 可开始实施
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

**警戒点**：启用后立刻在 `perf_public_set.jsonl` 上重测精度。如归一化精度跌破 79 %（v18 是 77.44 %，留 ~1.5 pt 安全垫），**不**自动放弃：要把精度损失与实测的 S1/S8/Smax 提速放在一起评估，由用户共同决定保留 / 调参 / 回退。硬底线仍是 97 % 规则（C=0），永远不主动跨过。

## 4. 改动文件

### 4.1 新代码（小面积）

| 文件 | 改动 |
|---|---|
| `python/sglang/srt/layers/quantization/gptq_marlin.py` | forward 里加分支：若 `os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"` 且 `sgl_kernel.fp8_blockwise_scaled_mm` 可导入且层在白名单（std-attn QKV/O + MLP gate/up/down）→ 调 sgl-kernel FP8 blockwise GEMM；否则走原 Marlin |
| `python/sglang/srt/models/minicpm.py` | 构造时给可用层打标签（linear forward 里读这个属性）— v1 排除 lightning Q/K/V/O |
| `benchmark/soar/demo_sala/preprocess_model.py` | GPTQ 量化后，把 INT4 反量化到 BF16，再做 128×128 block FP8 e4m3 量化，存为 `weight_fp8` + `weight_fp8_scale`。环境变量关闭则跳过 |
| `benchmark/soar/demo_sala/prepare_env.sh` | 追加 `export SOAR_W4A8_FP8_GEMM=1`。**不新增 pip install** — 直接用我们已经打包的 sgl-kernel wheel |

### 4.2 不需改动

- 所有 attention backend (FlashInfer / FA / SimpleGLA) — 它们不直接调 GEMM。
- Lightning recurrent state 路径 — 不变（与本提案正交，见提案 #3）。
- 评测脚本 `eval_model_001.py` — 不变。
- Server args 结构 — 不变。

## 5. 详细实施计划（改动前 — 供 review）

```
Step 1: 找到 sgl-kernel FP8 blockwise op + 确认 SM120 dispatch
   └─ 在 sgl-kernel 中 grep `fp8_blockwise_scaled_mm` 与 `sm120_fp8_blockwise_dispatch_shape`
   └─ 确认 op 签名：weight (K,N) FP8 e4m3 列主序 + 每 128x128 block scale
   └─ 确认激活契约：per-token 或 per-128 动态 FP8 量化
   └─ 不需要 fcloud — sgl-kernel 已在我们构建管线里

Step 2: 写 FP8-blockwise 权重转换 (preprocess_model.py)
   └─ 对每个白名单 linear 层：
        w_bf16 = gptq_dequantize(qweight, scales, qzeros, group_size=128)
        每 128x128 block:
            block_amax = block.abs().amax()
            scale_b   = block_amax / 448.0
            w_fp8_blk = (block / scale_b).clamp(-448, 448).to(torch.float8_e4m3fn)
        保存 weight_fp8 (K,N 列主序) + weight_fp8_scale ((K/128, N/128) fp32)
   └─ 单元测试：w_bf16_recovered = w_fp8.float() * scale_b_broadcast；与原 w_bf16 的 Frobenius 差 < 1e-3。

Step 3: Linear forward 分支 (gptq_marlin.py)
   └─ if SOAR_W4A8_FP8_GEMM and self.allow_fp8 and sgl_kernel_fp8_available:
          input_fp8, input_scale = per_token_fp8_quantize(input)   # 也来自 sgl-kernel
          out = sgl_kernel.fp8_blockwise_scaled_mm(
              input_fp8, self.weight_fp8,
              input_scale, self.weight_fp8_scale,
              out_dtype=torch.bfloat16)
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
- **V4 (必跑)**：`ncu --section ComputeWorkloadAnalysis ...` 现在应能在新的 `fp8_blockwise_scaled_mm` kernel 上看到**非零的 `sm__inst_executed_pipe_tensor_op_qmma`**。这是最关键的运行时信号——若依然为 0，就说明并未真正用上 FP8 硬件。
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

## 10. 批准记录（2026-04-27 已落实）

| # | 项目 | 用户决议 |
|---|---|---|
| 1 | 范围：v1 仅限标准注意力 + MLP | ✅ 同意 |
| 2 | 路径选择（Option A TRT-LLM wheel vs Option B sgl-kernel 内置） | ✅ **选 Option B** — 担忧：额外 wheel 可能破坏既有依赖且增加 2 GB 提交包体积 |
| 3 | 环境变量 gate `SOAR_W4A8_FP8_GEMM=1`，默认关闭 | ✅ 同意 |
| 4 | 精度保险：跌破 79 % 即硬中止 | ⚠️ 修订为软保险。要结合实测速度提升一起判断。硬底线仍是 97 % 规则 (C=0) |
| 5 | Step 1 = 验证 TRT-LLM wheel | ❌ 跳过 — Option B 用 sgl-kernel 内置 op，无需 wheel |

**实施顺序更新**：跳过 wheel 验证。从 Step 1（确认 sgl-kernel op 签名）开始，依次走完 Step 4。

## 11. 交叉引用

- `CLARIFICATION_quant_layers_runtime_verify_20260427_0811.{en,zh}.md` — 论证今日是 BF16-on-BF16-tensor-core（动机来源）
- `RESEARCH_mixed_arch_speed_optimization_20260426_1526.{en,zh}.md` — 把本项列为 #1
- `PROPOSAL_fp8_w4_dequant_gemm.md` — 原 Phase-1 TRT-LLM 调研 (kernel API、scale 转换草图)
- `PROPOSAL_option_b_fp8_blockwise_gemm.en.md` — Fallback (无额外 wheel) 用 sgl-kernel
- `ANALYSIS_nvfp4_offline_quant_20260427_0857.{en,zh}.md` — 解释为什么本步**不**走 FP4 权重

---

## 12. Step 1 调研结果 (2026-04-27)

按计划调研 sgl-kernel + sglang 的 FP8 栈，发现**upstream sglang 里集成已经大部分完成**：

| 发现 | 位置 | 含义 |
|---|---|---|
| SM120 上 `fp8_blockwise_scaled_mm` op 已存在 | `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu:368, 453`（经 `sm120_fp8_blockwise_dispatch_shape` 分发） | 不需写 kernel；我们的构建里已打进 wheel |
| op 签名 | `(a: (M,K) e4m3 行主序, b: (N,K) e4m3 调用 .t() 后为列主序, scales_a: (M, K/128) fp32, scales_b: (N/128, K/128) fp32, out_dtype) -> bf16/fp16` | 激活 = per-token 且沿 K 分 128 一组 |
| 已有高层包装 | `python/sglang/srt/layers/quantization/fp8_utils.py:342` `cutlass_w8a8_block_fp8_linear_with_fallback` | 已经做了 `per_token_group_quant_fp8(input, 128) → fp8_blockwise_scaled_mm(q_input, weight.T, x_scale, weight_scale.T)`。**可直接调用**。 |
| 激活量化 op | `sglang_per_token_group_quant_fp8`（来自 `sglang.srt.layers.quantization.fp8_kernel` 的 Triton kernel） | FP8 path 已在用，不需写 |
| 分支 gate 点 | `python/sglang/srt/layers/quantization/gptq.py:787` `GPTQMarlinLinearMethod.apply()` | 单一函数，一个 if/else 即可 |

**含义**：本迭代的运行时部分 ≈ 30 行分支代码 + 在权重 post-processing 里一次性设一个 `weight_fp8` 属性。大头在 `preprocess_model.py`（离线转换）。

**运行时分支草图**（取代 §5 Step 3）：

```python
# GPTQMarlinLinearMethod.apply() 内
if (
    os.environ.get("SOAR_W4A8_FP8_GEMM") == "1"
    and getattr(layer, "_soar_w4a8_eligible", False)
    and hasattr(layer, "weight_fp8")
):
    return cutlass_w8a8_block_fp8_linear_with_fallback(
        input=x,
        weight=layer.weight_fp8,            # (N, K) e4m3
        block_size=[128, 128],
        weight_scale=layer.weight_fp8_scale,  # (N/128, K/128) fp32
        bias=bias,
    )
# 否则：原 Marlin path（不变）
```

**`_soar_w4a8_eligible`** 是在 `minicpm.py` 构造时设的白名单标签 — 仅限 std-attn QKV/O + MLP gate/up/down。

**下一步（等用户发令）**：实施 Step 2 — `preprocess_model.py` 中从 GPTQ INT4 权重产生 `weight_fp8` + `weight_fp8_scale`。之后 Step 3 (gptq.py 分支) 与 Step 4 (minicpm.py 白名单标签) 都是轻量改动。
