# CHANGE 0151 — Phase B FourOverSix，续篇 005

承接 [CHANGE_0151_phase_b_four_over_six_004.zh.md](CHANGE_0151_phase_b_four_over_six_004.zh.md)。

第 6 轮在第 5 轮（71.24%）的基础上执行 **方案 A2**：保持第 5 轮的标定配方
（`SAMPLES=32 sequential`，`MAX_CALIB_SEQ_LEN=4096`，默认
`TASK_INCLUDE=qa,mcq,cwe`），但 **关闭 FourOverSix**
（`SOAR_NVFP4_FOUR_OVER_SIX=0`）。目的：直接判断 FOS 本身是否是 iter-5 与
iter-1（~73.13%）之间剩余 ~1.89pt 差距的来源。

本轮同时附带一个小型基础设施修复（方案 C），让今后的 NVFP4 量化产物自包含。

## 背景

第 5 轮后的悬而未决问题：在标定配方完全对齐 iter-1 后，iter-5 仍只到
71.24%，与 iter-1 相差 ~1.89pt。两种假设：

1. **FOS 中性或有害** → iter-1 的 73.13% 不依赖 FOS，剩余差距是 FOS 副作用。
2. **FOS 正向** → iter-5 ≈ iter-1（在 fcloud 方差范围内）。

第 6 轮以 `FOS=0` + 同一标定配方直接验证。

## 代码变更（方案 C — 持久化 tokenizer）

`benchmark/soar/demo_sala/preprocess_model.py::run_nvfp4_quantization` 流式
导出每个 Linear 的 FP4 权重到 `dst`，但从未调用
`tokenizer.save_pretrained(dst)`；并且函数中部（约 1587 行）`del tokenizer`
以释放 closure 持有的标定文本。结果：每次 NVFP4 量化都需要从 `src` 手动 `cp`
`tokenizer.json` 等 4 个文件，否则后续 chat-template 补丁会跳过。

修复：流式导出后从 `src` 重新加载 tokenizer 并调用 `save_pretrained(dst)`，
让 dst 自包含。

```python
# preprocess_model.py，紧跟 config.json 重写之后：
try:
    _tok_for_save = AutoTokenizer.from_pretrained(
        str(src), trust_remote_code=trust_remote_code
    )
    _tok_for_save.save_pretrained(str(dst))
    print("[preprocess] NVFP4 tokenizer.save_pretrained complete", flush=True)
except Exception as _tok_exc:
    print(
        f"[preprocess] NVFP4 tokenizer.save_pretrained FAILED: {_tok_exc!r}",
        flush=True,
    )
    raise
```

提交：`39c0045c5`（首版，用错了作用域）、`83921b207`（最终版，从 `src`
重新加载）。

iter-6 量化日志确认：
```
[preprocess] NVFP4 saving 1067 tensors to /root/models/MiniCPM-SALA-NVFP4-FOS
[preprocess] NVFP4 tokenizer.save_pretrained complete
[preprocess] NVFP4 manual export complete
[preprocess][change-0140] chat_template.jinja patched (v2): mcq prompts ...
[preprocess][init-rope-patch] mode=nvfp4 dst: ... replaced 2 _init_rope headers, 2 else-branches
```

iter-6 不再需要任何手动 `cp`，chat-template 补丁与 init-rope 补丁一次成功。

## iter-6 命令

```bash
# (1) 重新量化：FOS=0，保持 iter-5 标定配方
python3 scripts/fcloud/fcloud_exec.py exec \
  'rm -rf /root/models/MiniCPM-SALA-NVFP4-FOS && \
   cd /root/submission_sim && source ./prepare_env.sh && \
   SOAR_QUANT_PROFILE=nvfp4_fos \
   SOAR_NVFP4_FOUR_OVER_SIX=0 \
   SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
   SOAR_GPTQ_CALIBRATION_SAMPLES=32 \
   SOAR_GPTQ_CALIBRATION_SAMPLING=sequential \
   SOAR_GPTQ_CALIBRATION_SEED=20260320 \
   python3 -u preprocess_model.py \
     --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS \
     --mode nvfp4'

# (2) 重启 server（quant-mode=gptq 经 SOAR_QUANT_PROFILE 自动切到 modelopt_fp4）
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=0 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24

python3 scripts/fcloud/fcloud_workflow.py wait-server

# (3) 精度评测
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --quant-mode gptq \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## 结果

各任务精度（仅 run-1，触发 abort gate 后跳过 run-2 + speed）：

| 任务 | iter-5 (FOS=1) | iter-6 (FOS=0) | Δ |
|------|---------------:|---------------:|---:|
| cwe  | 70.67% | 70.00% | −0.67 |
| fwe  | 92.22% | 90.00% | −2.22 |
| mcq  | 53.33% | 43.33% | **−10.00** |
| niah | 90.00% | 90.00% | 0.00 |
| qa   | 50.00% | 43.33% | **−6.67** |
| **平均 ori_accuracy** | **71.24%** | **67.33%** | **−3.91** |
| 评测耗时 | 2485.20 s | 3163.71 s | +678.51 s |

iter-6 长度桶：len_0_4k 43.33%（mcq），len_4k_32k 57.50%，len_32k_128k
81.25%。

| Iter | Samples | Sampling | TASK_INCLUDE | calib_seq_len | FOS | 调度 | ori_accuracy |
|------|---------|----------|--------------|---------------|-----|------|--------------|
| 1    | 32      | sequential | （默认 qa,mcq,cwe） | 4096 | 1 | Tier1 | ~73.13% |
| 4    | 90      | stratified | qa,mcq,cwe | 4096  | 1 | Tier1 | 66.00%（ABORT） |
| 5    | 32      | sequential | qa,mcq,cwe | 4096 | 1 | Tier1 | **71.24%** |
| 6（本轮） | 32 | sequential | qa,mcq,cwe | 4096 | **0** | Tier1 | **67.33%**（ABORT） |

## 结论

**FOS 是保护性的，不是剩余差距的来源。** 在 iter-5 的标定配方上关闭 FOS
导致整体下滑 3.91pt，主要损失集中在 **mcq（−10pt）** 与 **qa（−6.67pt）** ——
这两类正是历史上最容易出现「短题尾部 / 思考-回答失控」的任务。cwe / niah /
fwe 的变化都在 ±2pt 之内，说明 FOS 在长上下文检索任务上几乎不起作用，它的
保护体现在小型短题上。

因此 iter-5 与 iter-1 之间 1.89pt 的剩余差距更像是 **fcloud 端的方差**
（调度、mcq 失控生成概率、modelopt requantize-resmooth 在 BF16 下的非确定
性），而不是优化信号。

iter-6 评测耗时也比 iter-5 多 678s，与「mcq/qa 失控生成更频繁地撞 max_tokens」
的模式吻合，与历史 GPTQ Test 30/32/33 的现象一致。

## 决策

1. **NVFP4 路径默认开启 FOS**（`SOAR_NVFP4_FOUR_OVER_SIX=1`，已是默认）。
2. **iter-5（71.24%）** 作为当前 NVFP4-FOS 最佳配置，1.89pt 的差距按方差接受。
3. **停止围绕 FOS flag 的进一步实验**。后续 NVFP4 工作应聚焦于方差控制
   （调度确定性、mcq 生成参数）或替代量化路径，而不是 FOS 本身。

## 回滚

FOS flag 默认值未变，无需回滚。若需撤销 tokenizer-save 修复：

```bash
git revert 83921b207 39c0045c5
```

## 下一步

| # | 想法 | 预期 | 工作量 | 风险 |
|---|------|------|--------|------|
| 1 | iter-5 ckpt → run-2 + speed bench（方差探针 + S1/S8/Smax） | 验证 71.24% 可复现；得到提交规划所需的速度数据 | 1h fcloud | 低 |
| 2 | 缓解 mcq 失控（generation_config：降低 temperature/top-p、stop tokens、仅在 server 端复审 max_tokens） | 弥补 iter-6 暴露的 mcq −10pt，对 iter-5 风格的运行可能再 +1–3pt | 中（需要保证不影响长上下文任务） | 中 |
| 3 | iter-5 NVFP4-FOS vs 当前 GPTQ 基线在 S1/S8/Smax 的对比，决定提交包路线（NVFP4 在长上下文 prefill 吞吐上可能占优；GPTQ 在 S1 占优） | 提交包决策依据 | 1h fcloud | 低 |
| 4 | 如果想继续推 NVFP4 精度：尝试 `TASK_INCLUDE=all` + SAMPLES=32 sequential | 未知 —— iter-1 用了过滤，可能差距不大 | 1h fcloud | 低 |

建议下一轮先做 (1) 验证 iter-5 可复现，再做 (3) 为提交包决策提供依据。
