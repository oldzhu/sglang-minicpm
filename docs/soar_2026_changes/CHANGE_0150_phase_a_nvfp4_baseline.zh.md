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

**关键发现 2（速度）：** 模型用 `modelopt_fp4` 加载后，长上下文的延迟比 GPTQ 基线
明显更差。可能原因：sglang 自带的 `modelopt_fp4` 加载器并 **没有** 走 SM120 的 FP4 Tensor
Core 内核，每次线性计算都把权重反量化回 BF16（加载时打印的 "experimental and subject to
change" 也佐证）。后续需要量化这个内核走向。

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
   - **排查内核路径** —— 确认 SM120 上 sglang 的 `modelopt_fp4` 加载器是否真的用了 FP4
     Tensor Core；如果它退回到 BF16 反量化，那预期的 FP4 速度收益就是空头，需要加自定义内核
     （类似 CHANGE_0090 的 W4A8 FP8 GEMM）。

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
