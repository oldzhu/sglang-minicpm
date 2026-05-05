# CHANGE 0150 — Phase A：NVFP4 基线（统一 NVFP4）首次端到端尝试

日期：2026-05-04
分支：`mixed_minicpm_cudagraph`
提交：`aa1304292`（Phase A 初版）、`0a56da668`（modelopt 0.43 / torch 2.9 兼容修复）

## 背景

冠军博客复现路线图 Phase A（详见 `PROPOSAL_phase_a_nvfp4_baseline_design_20260504.zh.md`）：
通过 `nvidia-modelopt` 把 MiniCPM-SALA 的全部线性权重量化为 **统一 NVFP4**
（block_size=16，FP8 E4M3 每块缩放），用 sglang 现有 `modelopt_fp4` 加载器加载，
确认管线端到端可用，再进入 Phase B 加 FourOverSix 自适应。

## 代码改动

1. `benchmark/soar/demo_sala/prepare_env.sh`
   - 新增 `SOAR_QUANT_PROFILE` 开关（`gptq` | `nvfp4` | `nvfp4_fos`）。
   - 条件式安装 `nvidia-modelopt`（仅当 profile 命中时）。
   - server-arg 分支 `--quantization gptq_marlin` ↔ `--quantization modelopt_fp4` 切换。
2. `benchmark/soar/demo_sala/preprocess_model.py`
   - 新增 `run_nvfp4_quantization(...)`：调 `mtq.quantize` + `NVFP4_DEFAULT_CFG`，
     排除 `lm_head`/`o_gate`/`z_proj`/`norm`/`embed_tokens`，
     通过 `modelopt.torch.export.export_hf_checkpoint` 导出。
   - 新增 `mode='nvfp4'` argparse；mode 解析优先采用 `SOAR_QUANT_PROFILE`。

## 依赖发现：modelopt vs torch 2.9

- 最初计划的 modelopt **0.31.0** 会 `import torch.onnx._type_utils`，但该符号在 torch 2.8+ **已被删除**。
  → 在我们固定的 `torch==2.9.1+cu128` 上直接 ImportError。
- modelopt **0.43.0** 是首个兼容 torch 2.9 的版本。同步安装时
  `nvidia-modelopt-core`（Cython 内核）目前最高只到 **0.33.1** —— 该组合是 fcloud 上验证通过的版本（2026-05-04）。
- `import modelopt.torch.quantization` 实际触发的传递依赖：
  `cppimport`、`pulp`、`onnx`、`pydantic`、`rich`、`torchprofile`
  （全部用 `--no-deps` 安装，确保 `torch`/`transformers`/`gptqmodel`/`flash-attn`/`huggingface-hub` 不被覆盖）。
- **重要：**`uv pip install nvidia-modelopt[torch]` 不带 `--no-deps` 时会悄悄把 torch 升到 2.10。
  我们后续强制 `torch==2.9.1` 回写。

`prepare_env.sh` 在 commit `0a56da668` 中按上述结论更新。

## 端到端验证结果

| 步骤 | 结果 |
| --- | --- |
| `SOAR_QUANT_PROFILE=nvfp4` 下 `prepare_env.sh` | ✅ 顺利完成，输出含 `--quantization modelopt_fp4` |
| `python3 -c 'import modelopt.torch.quantization as mtq'` | ✅ 通过（modelopt 0.43.0） |
| sglang 加载已有的 `/root/models/MiniCPM-SALA-NVFP4` | ✅ 显存占用 6.65 GB，CUDA Graph 抓取正常 |
| 短提示词 ("The capital of France is") | ⚠️ 先输出 "Paris" 然后陷入 "unedoc.com/abc/u.com/..." 复读式垃圾输出 |
| 150 样本精度评测（concurrency=32） | ⏰ 客户端 3600 s 超时，约 62/150（41 %）；长上下文样本单条耗时超过 5 分钟 |
| 短提示词解码吞吐 | 约 57 tok/s（64 tokens / 1.118 s）—— **短上下文足够快**，长上下文才是瓶颈 |

**关键发现 1（质量）：** 统一 NVFP4（block_size=16，仅 FP8 缩放，不做 M=6/M=4 自适应）
在长生成上崩溃。这与冠军博客的明确警告吻合：必须先引入 **FourOverSix 块级自适应**
模型才可用。我们当前的实现就是无 FOS 的基线，本就预期会过度有损。

**关键发现 2（速度）：** 模型用 `modelopt_fp4` 加载后，长上下文延迟比 GPTQ 基线明显更差。
最初的猜测（退回 BF16 反量化）**是错的** —— 走查
`python/sglang/srt/layers/quantization/modelopt_quant.py` 后确认 `ModelOptFp4LinearMethod.apply`
实际执行：

1. `x_fp4, x_scale = fp4_quantize(x, layer.input_scale_inv)` —— 每步把 BF16 激活动态 cast
   到 NVFP4（SM120 上走 flashinfer `fp4_quantize`，其它走 sgl-kernel `scaled_fp4_quant`，
   由 `is_sm120_supported()` 在 import 时分发）。
2. `out = fp4_gemm(x_fp4, w_fp4, x_scale, w_scale, alpha, out_dtype, w_n)` —— 调用
   flashinfer `mm_fp4`（cutlass 后端），落到 SM120 的 FP4 cutlass GEMM。assert 确认
   `weight.dtype == uint8`（FP4 打包）、`weight_scale.dtype == float8_e4m3fn`，
   **权重全程不会被反量化回 BF16。**
3. 内核把 BF16/FP16 输出回写给下一层。

所以 FP4 Tensor Core **确实被用上了**。长上下文慢的真正原因，按优先级：

1. **质量崩塌 → 生成失控。** 统一 NVFP4 输出乱码，mcq/qa 不会自然停在答案上，会一路生成
   到 `max_tokens=65536`。"5 分钟/条" 大概率是单条吐了 ~65k token，而不是 token 慢。
   与冒烟测试 "Paris → unedoc.com/abc/u.com..." 是同一种失效模式。
2. **逐步激活再量化开销。** 每个线性、每一步都做 BF16→FP4。短 prompt 上很便宜，但
   chunk=65536 × prefill_max_requests=4 时可能把 FP4 GEMM 的提速吃掉。
3. **注意力还是 BF16。** Q/K/V 投影是 FP4 GEMM，但 flashinfer 注意力内核吃 BF16，
   所以 attention 前后还要 FP4↔BF16 来回转。

第 1 项大概率主导，必须先用 FourOverSix 救回质量才能独立衡量第 2/3 项。后续把
（a）nsys 内核分发验证、（b）`fp4_quantize` 与 `mm_fp4` 时间比、（c）`max_tokens=512`
下的短上下文 mcq 探针放进调研任务清单。

## 新增/修改的文件

- `benchmark/soar/demo_sala/prepare_env.sh` —— modelopt 安装开关 + arg 切换
- `benchmark/soar/demo_sala/preprocess_model.py` —— `run_nvfp4_quantization`、`mode='nvfp4'`
- `docs/soar_2026_changes/RESEARCH_week7_champion_review_20260504.{en,zh}.md`
- `docs/soar_2026_changes/PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.{en,zh}.md`
- `docs/soar_2026_changes/PROPOSAL_phase_a_nvfp4_baseline_design_20260504.{en,zh}.md`
- `docs/soar_2026_changes/chat/CHAT_week7-champion-review_20260504_1700.{en,zh}.md`

## 下一阶段结论

1. 统一 NVFP4 **不能** 直接用作提交基线（质量崩溃）—— 与冠军博客的判断完全一致。
2. Phase A 的价值在 **打通管线**：profile 开关、modelopt 安装路径、server-arg 切换、sglang
   `modelopt_fp4` 加载器整合在 fcloud 上已经全部跑通。
3. 现有的 `/root/models/MiniCPM-SALA-NVFP4` 是更早一次尝试的产物，本轮还没真正跑过我们自己的
   `run_nvfp4_quantization`（评测在重新量化之前就超时了）。**TODO：** 下一次会话需要执行
   `SOAR_QUANT_PROFILE=nvfp4 bash prepare_model.sh --input ... --output ...` 验证我们的标定路径。
4. 两个可行的下一步方向：
   - **Phase B（FourOverSix）** —— 在自定义的 modelopt `forward_loop` 内实现块级 M=6/M=4
     自适应，目标达到冠军那条 ≥99% 的精度线。
   - **排查内核路径** —— 代码走查已经证实 FP4 Tensor Core **确实被用**（不会走 Marlin
     也不会反量化回 BF16）。剩下的未知项：在 nsys 时间线里实际看到 `cutlass_scaled_fp4_mm`
     的内核占比、`fp4_quantize` 相对 `mm_fp4` 的开销比、以及只跑短上下文 mcq 看精度，
     用来把质量崩塌和内核速度两个因素拆开。

## 探针结果 —— 内核分发 + 微基准（2026-05-05）

在 fcloud SM120 实例上以 v22 wheel 集运行 `scripts/fcloud/probe_nvfp4_kernel.py`。
确认 `is_sm120_supported() == True`，且 `flashinfer.fp4_quantize` / `flashinfer.mm_fp4`
（cutlass 后端）能正常执行。对典型投影尺寸做微基准（K=4096，ffn N=10880，qkv N=4096）：

| 场景 | M | BF16 ms | FP4 e2e ms | 只 mm_fp4 | 只 quant | quant 占比 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-1   × qkv | 1 | 0.019 | 0.038 | 0.043 | 0.007 | 17 % | **0.49×** |
| decode-1   × ffn | 1 | 0.035 | 0.038 | 0.034 | 0.007 | 18 % | **0.92×** |
| prefill-512× qkv | 512 | 0.160 | 0.036 | 1.661¹ | 0.007 | 18 % | 4.40× |
| prefill-512× ffn | 512 | 0.476 | 0.075 | 0.091 | 0.007 | 9 %  | 6.38× |
| chunk-4096 × qkv | 4096 | 1.118 | 0.267 | 0.274 | 0.012 | 5 %  | 4.19× |
| chunk-4096 × ffn | 4096 | 2.706 | 0.683 | 0.651 | 0.018 | 3 %  | 3.96× |
| chunk-65536× qkv | 65536 | 15.45 | 3.95 | 3.89 | 0.55 | 14 % | 3.91× |
| chunk-65536× ffn | 65536 | 41.25 | 10.28 | 10.21 | 0.55 | 5 %  | 4.01× |

¹ 首次调用的 JIT/启发式离群值；端到端路径预热路径不同，忽略。

结论：
1. **FP4 内核健康，在 prefill/chunk 尺寸上给出预期的 ~4× 加速。** 这个比例正好对应
   SM120 上 BF16 (148 TF) → FP4 (593 TF) 的硬件比值。
2. **`fp4_quantize` 激活 cast 不是瓶颈**（在 prefill/chunk 尺寸下 ≤10 %；chunk=65536
   也只有 14 %）。否定了"逐 token 再量化"是长上下文主要开销的假设。
3. **decode-1 反而比 BF16 慢（0.49–0.92×）。** 此时是访存瓶颈，再加上多发一次内核
   的启动开销，把 compute 节省吃掉了。NVFP4 单独看不会让单步 decode 更快 —— 要在
   decode 上获益必须把 `fp4_quantize` 融入前一个算子，或者批量 decode。
4. **因此 Phase A 测试中长上下文慢的根因不是内核。** 既然 GEMM 已经有 ~4× 提速，
   prefill 本应更快而不是更慢。主导原因只能是**质量崩塌导致的失控生成** ——
   `max_tokens=65536` 下，单个坏样本可以一直吐乱码几分钟。这进一步确认下一步必须是
   Phase B（FourOverSix）先把质量救回来，再谈速度。

## 验证命令

```bash
# 在 fcloud 上重建环境（自动注入 --quantization modelopt_fp4）
cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 bash prepare_env.sh

# 用现成的 NVFP4 权重启动 sglang
python3 -m sglang.launch_server \
  --model-path /root/models/MiniCPM-SALA-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  $SGLANG_SERVER_ARGS

# 短提示词冒烟测试
curl -s -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":64,"temperature":0.0}}'
```

## 回滚

把 `SOAR_QUANT_PROFILE` 设回 `gptq`（默认值）即可。所有 NVFP4 路径都是特性开关式的，
环境/模型布局都不留状态。
