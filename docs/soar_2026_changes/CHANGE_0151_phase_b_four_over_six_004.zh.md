# CHANGE 0151 — Phase B FourOverSix 续篇 004

承接 [CHANGE_0151_phase_b_four_over_six_003.zh.md](CHANGE_0151_phase_b_four_over_six_003.zh.md)。

iter-5 在 iter-4 的结论（FOS 暂时搁置；校准**内容**才是回退主因）基础上，
执行 **Option A**：保持 FOS=1、calib_seq_len=4096、Tier1 调度，将校准
样本数与抽样方式回退到 iter-1 的设定（`SAMPLES=32 sequential`），重新
量化并测精度。

## 背景

iter-4 已通过将 calib_seq_len 从 16384 改为 4096 排除其作为回退源的可能
（66.00% vs iter-3 的 68.20%，二者均为 stratified-90 qa,mcq,cwe）。剩余
假设是**校准样本数 / 抽样方式**才是主导：iter-1 用 `SAMPLES=32 sequential`
（默认混合任务分布），iter-{2,3,4} 用 `SAMPLES=90 stratified` 配合
`TASK_INCLUDE=qa,mcq,cwe`。

| Iter | Samples | Sampling | TASK_INCLUDE | calib_seq_len | 调度 | ori_accuracy |
|------|---------|----------|--------------|---------------|------|--------------|
| 1    | 32      | sequential | （prepare_env 默认 qa,mcq,cwe） | 4096 | Tier1 | ~73.13% |
| 2    | 90      | stratified | qa,mcq,cwe | 16384 | 保守 | ~62.02% |
| 3    | 90      | stratified | qa,mcq,cwe | 16384 | Tier1 | 68.20% |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | Tier1 | 66.00%（ABORT）|
| 5（本次） | **32** | **sequential** | qa,mcq,cwe（默认） | 4096 | Tier1 | **71.24%** |

注：`SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` 在 prepare_env.sh 第 210 行的默认
值就是 `qa,mcq,cwe`，因此即便不显式设置该环境变量，task 过滤器仍然生效，
与 iter-{2,3,4} 一致。本次仅 `SAMPLES` 与 `SAMPLING` 与 iter-4 不同。

## 实施

未改动源码。仅在调用时设置环境变量：

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && \
   cd /root/submission_sim && source ./prepare_env.sh && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
   SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'
```

校准日志确认：

```
calibration_sampling={"available": 90, "mode": "sequential",
  "records_after_task_filter": 90, "records_before_task_filter": 150,
  "seed": 20260320, "selected": 32, "selected_buckets": {"all": 32},
  "task_balance": true, "task_filter_applied": true,
  "task_include": ["qa","mcq","cwe"], "use_prompt_tokens": true}
```

`pct_m4 = 43.14%`（与 iter-4 完全相同——在同一个 90 条 qa,mcq,cwe 池中，
无论选哪 32 / 90 条样本，FOS 的 scale 选择统计几乎不变）。

### tokenizer 文件缺失（一次性补救）

`preprocess_model.py` 的流式 NVFP4 导出路径并未调用
`tokenizer.save_pretrained(dst)`，因此首次新建 `dst` 后缺失
`tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` /
`special_tokens_map.json`。第一次启动 sglang 时报错：

```
ValueError: Unrecognized configuration class
  ...MiniCPMSALAConfig... to build an AutoTokenizer.
```

本次手动补救（**未持久化为代码**）：

```bash
cp /root/models/openbmb/MiniCPM-SALA/tokenizer.json \
   /root/models/openbmb/MiniCPM-SALA/tokenizer.model \
   /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json \
   /root/models/openbmb/MiniCPM-SALA/special_tokens_map.json \
   /root/models/MiniCPM-SALA-NVFP4-FOS/

# 重跑 mcq chat-template 补丁（change-0140）；首次预处理时 tokenizer_config
# 还不存在，被跳过了。
python3 -c "
import sys; sys.path.insert(0, '/root/submission_sim')
from pathlib import Path
import preprocess_model as pm
pm._patch_chat_template_for_mcq(Path('/root/models/MiniCPM-SALA-NVFP4-FOS'))
"
```

下一轮建议在 `run_nvfp4_quantization` 内补上
`tokenizer.save_pretrained(dst)`，使 NVFP4 量化产物自包含。

### 启动 sglang

`--quant-mode gptq` 才能进入 prepare_env.sh 中根据 `SOAR_QUANT_PROFILE=nvfp4_fos`
将 `--quantization gptq_marlin` 替换为 `--quantization modelopt_fp4` 的分支。
本轮初次误用 `--quant-mode noquant`，导致 sglang 加载时回退到默认
`ModelOptFp8Config` 并报：

```
ModelOptFp8Config only supports static FP8 quantization in SGLang.
For FP4 quantization, use ModelOptFp4Config.
```

后续修正为：

```bash
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

## 结果 — iter-5 run-1

`outputs/20260508_071351/predictions.jsonl`

| 指标 | 数值 |
|------|------|
| ori_accuracy（Average Score）| **71.24%** |
| Total Duration | 2485.20 s |
| Total Tokens | In=8,644,406  Out=938,189 |
| FOS pct_m4 | 43.14% |

各任务对比：

| Task | Iter-3 | Iter-4 | **Iter-5** |
|------|--------|--------|-----------|
| cwe  | 77.67  | 66.67  | **70.67** |
| fwe  | 76.67  | 80.00  | **92.22** |
| mcq  | 56.67  | 46.67  | **53.33** |
| niah | 80.00  | 93.33  | **90.00** |
| qa   | 50.00  | 43.33  | **50.00** |

iter-5 通过 70% 的 abort gate（71.24%）。本轮按用户范围只跑了一次精度，
未执行 run-2 与 speed bench。

## 与历次对比

| Iter | Samples | Sampling | TASK_INCLUDE | seqlen | ori_acc | 与 iter-1 差距 |
|------|---------|----------|--------------|--------|---------|----------------|
| 1    | 32      | sequential | qa,mcq,cwe | 4096 | ~73.13% | — |
| 2    | 90      | stratified | qa,mcq,cwe | 16384 | ~62.02% | −11pt |
| 3    | 90      | stratified | qa,mcq,cwe | 16384 | 68.20% | −5pt |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | 66.00% | −7.13pt |
| **5** | **32** | **sequential** | qa,mcq,cwe | **4096** | **71.24%** | **−1.89pt** |

在同一个 qa,mcq,cwe 池里，把 90-stratified 改成 32-sequential，单步
**+5.24pt**（在 FOS=1 仍开启的前提下）。这证实**在 qa,mcq,cwe 任务范围
内，更小的 32-sequential 选样比 90-stratified 更适合 NVFP4 权重的校准
内容**，原因可能是：

- `samples=32 sequential` = 公共集前 32 条记录，三个任务桶的混合比例与
  评测分布更接近；
- `samples=90 stratified` 过度倾向于长输出 / 高 FOS 分数的样本，导致下游
  层的 per-channel scale 统计偏置。

剩余 ~1.9pt 与 iter-1 的差距，最可能来自 FOS 本身（iter-1 大概率
`FOS=0`）或 fcloud 实例间的方差。

## 下一步选项

| 选项 | 描述 | 成本 | 目的 |
|------|------|------|------|
| A1 | iter-5 + run-2 + S1/S8/Smax 速度评测 | 1 个 fcloud 会话 ~70 分钟 | 验证 71.24% 可复现；为 FOS-32 checkpoint 建立速度基线 |
| A2 | 重新量化：`SAMPLES=32 sequential FOS=0`（纯 NVFP4） | 1 个 fcloud 迭代 | 测试残留 ~1.9pt 是否由 FOS 引入。若 A2 ≥ 73%，则 FOS 永久搁置 |
| B  | 去掉 `TASK_INCLUDE` 过滤（5 任务全集），保持 `SAMPLES=32 sequential FOS=1` | 1 个 fcloud 迭代 | 测试 qa,mcq,cwe 过滤本身的精度成本 |
| C  | 在 `preprocess_model.py` 中固化 tokenizer.save_pretrained | 小改 | 让 NVFP4 量化产物自包含，免去手动 cp |

推荐：先 **C**（成本极低），再 **A2**（最直接验证 FOS 是否值得保留）。
若 A2 能补齐与 iter-1 的差距，则 FOS 永久搁置。

## 验证命令

```bash
# 检查 _init_rope 补丁
grep -c "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# 预期：2

# 检查 tokenizer 文件是否齐全
ls /root/models/MiniCPM-SALA-NVFP4-FOS/tokenizer*

# 重新跑精度
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## 回滚

无源码改动。要恢复 iter-4 checkpoint，按 003 文档的 iter-4 命令重新量化
（`SAMPLES=90 SAMPLING=stratified`）即可。

## 副产品：setup 脚本现在做真正的 `git clone`

本轮顺手修复了 `fcloud_workflow.py sync` 长期回退到 force-copy 的问题：
将 `step_setup` 中的"上传 tarball 再解压"替换为
`git clone --depth 1 --branch mixed_minicpm_cudagraph https://github.com/oldzhu/sglang-minicpm.git`。
clone 完成后，`sync` 走 `git pull` 路径，仅在 diff 计算失败时回退到
force-copy。新增"无新提交"分支：报告 `(no new commits)` 并跳过 copy /
sgl-kernel-build 子流程。

针对老实例（`/root/sglang-minicpm` 缺 `.git/`）的人工补救：

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/sglang-minicpm && \
   git clone --depth 1 --branch mixed_minicpm_cudagraph \
     https://github.com/oldzhu/sglang-minicpm.git /root/sglang-minicpm'
```

`copilot-instructions.md` 已同步该说明。

## 交叉引用

- 续篇 003: [CHANGE_0151_phase_b_four_over_six_003.zh.md](CHANGE_0151_phase_b_four_over_six_003.zh.md)
- `_init_rope` 补丁: [CHANGE_0152_init_rope_transformers5_compat.zh.md](CHANGE_0152_init_rope_transformers5_compat.zh.md)
- 测试记录: TEST_RESULTS_TRACKING.md → NVFP4-FOS-5
- 会话日志: chat/CHAT_phase-b-fos-iter5_20260508_1430.zh.md
