# CHAT — Phase B FourOverSix 端到端（2026-05-06）

## 第 1 轮 — 用户："go"

用户要求恢复执行 Phase B FourOverSix：fcloud 上量化 NVFP4-FOS → smoke → 精度 + S1/S8/Smax → 文档 → 暂停。

Agent 操作：
- 清理 `/root/models/MiniCPM-SALA-NVFP4-FOS`，重新启动
  `preprocess_model.py --mode nvfp4`，环境 `SOAR_QUANT_PROFILE=nvfp4_fos`。
- 量化成功：`EXIT=0`，GPU 峰值 34.20 GiB，FOS pct_m4=43.14%，
  但 ckpt 高达 23 GiB（1163 个张量）。

诊断膨胀原因：16 GiB 的 fp32 rotary `cos_cached`/`sin_cached` 被写入 ckpt，
因为 `named_buffers()` 默认会返回非持久 buffer。

修复（commit `829128503`）：手动导出的 buffer 遍历环节，先按各模块的
`_non_persistent_buffers_set` 构造完整名集合，再过滤掉所有非持久 buffer。

重跑量化：ckpt 降到 **6.5 GiB**，1067 个张量，EXIT=0。

## 第 2 轮 — 起服

- restart-server 携带 `--model-path /root/models/MiniCPM-SALA-NVFP4-FOS`
  以及 env `SOAR_QUANT_PROFILE=nvfp4_fos` + `SOAR_NVFP4_FOUR_OVER_SIX=1`。
- `wait-server` 5 分钟超时，原因是该配置 cudagraph 捕获耗时 366 s（7 个 batch ×
  torch.compile）。
- 捕获结束后 `/health` 返回 200；服务端日志 `Detected nvfp4 checkpoint`，
  `mem_usage=7.31 GB`（fp4 + bf16 norms/embed 的预期）。
- Smoke (`What is 2+2?`) 输出连贯 `<think>` 文本。

## 第 3 轮 — 精度

- 第一次失败：`--api_base http://127.0.0.1:30000/v1` 让脚本拼出
  `/v1/v1/models` 导致 404；改用 `--api_base http://127.0.0.1:30000`
  与 `fcloud_workflow.py` 的 `API_BASE` 对齐后正常。
- 评测 150 样本耗时 2697.48 s。

**结果：ori_accuracy = 75.98%**（低于 77% 门槛，C = 0）。

按任务：cwe 74.33、fwe 92.22、mcq 63.33、niah 93.33、qa 56.67。
按长度：0_4k 63.33、4k_32k 70.25、32k_128k 83.58。

## 决策：跳过速度跑、写文档、暂停实例

C=0 时跑速度无意义，跳过。
暂停 fcloud 实例（前两次 504/500，第三次 `HTTP 200 任务已暂停`）。

## 产出

- `mixed_minicpm_cudagraph` 推送 3 个提交到 `minicpm-src`：
  - `a2cbedd65` 手动流式导出（替代漏内存的 modelopt export）
  - `f14c3f3e8` 流式循环跳过非 Linear 模块（修 NotImplementedError）
  - `829128503` 跳过非持久 buffer（修 16 GiB rotary 缓存膨胀）
- 手动导出设计已验证：6.5 GiB ckpt、GPU 峰值 34.20 GiB、服务端可加载并完成
  KV 分配 + cudagraph + smoke + 150/150 精度评测，全部 EXIT=0。
- **精度是当前阻塞。** 当前 FOS 实现（在 M=4 / M=6 间按局部 MSE 二选一）
  对该模型保留精度不足。

## 下一步建议（已写入 CHANGE_0151）

A. 按层跳过 FOS（Q/K/V 或前后若干层退回普通 NVFP4）
B. 混精度：MLP fp4 + QKV w8（工作量更大）
C. 跑纯 NVFP4（关 FOS）作 baseline 隔离 FOS 影响
D. 校准前向也激活 FOS 感知尺度
E. 暂搁 Phase B，回到 GPTQ + FP8 baseline 推进速度优化

## 交叉引用

- [CHANGE_0151_phase_b_four_over_six.en.md](../CHANGE_0151_phase_b_four_over_six.en.md)
- [CHANGE_0151_phase_b_four_over_six.zh.md](../CHANGE_0151_phase_b_four_over_six.zh.md)
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) — 新增 "Phase B" 子表，行 NVFP4-FOS-1
- 提交：a2cbedd65、f14c3f3e8、829128503（`mixed_minicpm_cudagraph` 分支）
