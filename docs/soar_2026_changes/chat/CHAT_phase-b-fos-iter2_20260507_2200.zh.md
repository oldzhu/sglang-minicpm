# CHAT — Phase B FOS 迭代 2(2026-05-07 22:00)

英文兄弟版:[CHAT_phase-b-fos-iter2_20260507_2200.en.md](CHAT_phase-b-fos-iter2_20260507_2200.en.md)

## 本轮摘要

用户在手动硬重启 fcloud 实例(用于清理 pause/resume 无法清除的 84 GiB GPU 泄漏)
后,要求 agent 继续 iter-2 重试。

## 操作序列

1. **预检** — GPU 已用 0 MiB(干净)。`/root/sglang-minicpm`、`/root/submission_sim`、
   `/root/data` 全部丢失(硬重启清空非持久存储)。`/root/models` 保留。

2. **Setup** — `python3 scripts/fcloud/fcloud_workflow.py setup` — 上传
   `submission_sim.tar`(731 MB)和 `data.tar.gz`(8 MB),克隆 `sglang-minicpm`,
   复制 python/ + demo_sala/ 脚本。约 10 分钟。

3. **量化第 1 次(env 出 bug)** — 通过 `fcloud_exec.py` 同步执行,使用
   `source ./prepare_env.sh | tail -20`。**Bug**:管道把 `source` fork 到子 shell,
   导出永远到不了父进程。量化以**默认值**运行:`calibration_samples=32`(顺序,
   无任务过滤),`max_calib_seq_len=4096`。FOS pct_m4=43.14% 与迭代 1 相同。

4. **量化第 2 次(env 正确)** — `source ./prepare_env.sh >/tmp/prep.log 2>&1;
   echo SAMPLES=$SOAR_GPTQ_CALIBRATION_SAMPLES …; python3 -u preprocess_model.py …`。
   验证 env:`SAMPLES=90 SEQLEN=16384 TASKS=qa,mcq,cwe FOS=1`。量化 OK。
   **FOS pct_m4 仍为 43.14%** — 证实 FOS 尺度选择仅由权重决定,校准数据不影响
   M=4 vs M=6 选择。

5. **复制 tokenizer** — manual export 不包含 tokenizer 文件。从源 MiniCPM-SALA
   目录复制。

6. **重启 server** — 启动日志中确认了保守调度参数:`--chunked-prefill-size 32768
   --prefill-max-requests 1 --schedule-conservativeness 1.0 --quantization
   modelopt_fp4 --kv-cache-dtype fp8_e5m2 --enable-torch-compile
   --torch-compile-max-bs 8`。约 5 分钟 cudagraph capture 后 server 起来。

7. **冒烟** — 返回 `<think>` 块,OK。

8. **Accuracy 第 1 次** — `fcloud_workflow.py accuracy`(第一次默认走 GPTQ
   tokenizer 路径不存在,改用 `--model-path` 标志后成功)。`fcloud_exec` 在 3600s
   超时,但 server 端 eval 继续执行。通过 `wc -l predictions.jsonl` 确认完成。
   结果:**ori 60.73%** / norm 75.92%,duration 3614.90s。niah(93→73)、
   fwe(92→70)、cwe(74→60)大幅下降。

9. **Accuracy 第 2 次** — `ori 63.31%` / norm 79.14%,duration 3631.80s。失败
   模式转移:本次 qa 崩到 36.67%,fwe 出现 runaway-think(avg_out=15792)。

10. **Speed 测试** — S1=173.69, S8=45.95, Smax=34.39(对比迭代 1:
    175.08/47.37/31.01;Smax 因 torch_compile_max_bs=8 而慢 11%)。

11. **暂停实例** — 第一次 HTTP 504,第二次重试 HTTP 200。

## 结果

- **迭代 2 平均 ori = 62.02%**,对比迭代 1 平均 ≈ 73.13% — **明确回退 11pt**。
- 运行间方差仍约 3pt;失败模式在任务间轮转(mcq → niah/cwe → fwe → qa)。
- 迭代 2 两个假设**均被否定**:
  - 保守调度未消除 runaway-think 方差。
  - 更长 + qa/mcq/cwe 分层校准在 niah/fwe 上**变得更糟**(可能由于激活 amax 改变)。
- FOS pct_m4 是**仅由权重决定**(本会话 3 次量化均为 43.14%)。校准变化只影响
  激活 amax,不影响 FOS 尺度选择。

## 结论与建议

**暂停 Phase B FOS。** 两次迭代均未通过 C ≠ 0 阈值。NVFP4 权重量化(无论是否
带 FOS)似乎在长 context 路径(niah/fwe)上通过激活量化丢失精度,16 元素块的 FP4
input quantizer 无法表示。修复需要混合精度或不同的 `input_quantizer` 配置 —
两者都是大量工作。

推荐下一方向:**回归 GPTQ + FP8 KV + dense 基线**(T12 系列,ori 79.29%, norm
99.11%, C=1.0),并继续追求
[OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
中尚未穷尽的优化向量。

## 交叉引用

- 新续篇文档:
  - [CHANGE_0151_phase_b_four_over_six_001.en.md](../CHANGE_0151_phase_b_four_over_six_001.en.md)
  - [CHANGE_0151_phase_b_four_over_six_001.zh.md](../CHANGE_0151_phase_b_four_over_six_001.zh.md)
- 新 TEST_RESULTS 行:NVFP4-FOS-1b、NVFP4-FOS-2、NVFP4-FOS-2b,见
  [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md)。
- 分支 HEAD:`a6b34a41a`(本轮无新代码 commit;iter-2 prepare_env 变更已在
  上一会话推送)。

## 给用户的开放问题

1. 同意"暂停 Phase B FOS,回归 GPTQ 基线"的建议吗?
2. 若想在暂停前再试一个 NVFP4 角度:你倾向于 plain NVFP4 + 迭代 1 默认校准
   (廉价 A/B 测试"迭代 1 只是校准走运"假设)还是混合精度
   QKV-int8/MLP-NVFP4 配方(大量工作,多日)?
