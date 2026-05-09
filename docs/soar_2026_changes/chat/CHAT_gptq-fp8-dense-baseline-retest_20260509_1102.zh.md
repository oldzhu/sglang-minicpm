# 对话记录 — GPTQ_FP8_DENSE 基线在新 fcloud 上的回测 (2026-05-09 11:02)

主题：暂停 NVFP4-FOS 路线；在新 fcloud 实例上对 **GPTQ_FP8_DENSE 目录最佳配置** 做一次端到端的精度 + 速度回测，作为 GPTQ 目录下一轮优化的基准锚点。

交叉引用：
- [TEST_RESULTS_TRACKING 中的 `GPTQ-FP8-DENSE-retest-newinst` 行](../TEST_RESULTS_TRACKING.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- [CHANGE_0151_phase_b_four_over_six_007.zh.md](../CHANGE_0151_phase_b_four_over_six_007.zh.md)（触发本次回测的深度分析）
- 上一份对话记录：[CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.zh.md](CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.zh.md)

## 1. 用户原话

> while I am reviewing, pls temp park NVFP4+FOS path and do 1 time testing for GPTQ_FP8_DENSE catalog best config for accuracy and bechmark in the new fcloud instance to see what result we will get.

## 2. Agent 执行步骤

1. 通过 `start-instance` 唤醒 fcloud 实例（HTTP 200；console JWT 有效期至 2026-05-14）。
2. fcloud 上 `git pull` 同步：`cce6b5a..ac91b1a` 快进于 `mixed_minicpm_cudagraph` 分支。
3. **新实例上引导 GPTQ**（无现成量化检查点）：
   - `apt-get install -y libpcre3-dev`
   - `CC=gcc CXX=g++ pip install --no-binary :all: python-pcre`
   - 用 stdlib `re` 透传 shim 替换 `site-packages/pcre.py`（gptqmodel writer 需要 `re.fullmatch`，`python-pcre` 不提供）。
   - 原 `pcre.py` 备份为 `pcre.py.bak`。
   - 全程 `prepare_model.sh`：31 层、约 74 分钟，产出 `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/`（6.4 GB，sparse_qkv_w8 预设）。
4. 用 `prepare_env.sh` 默认（v22 上线配置）重启 server。启动约 7 分钟（torch.compile + bs=[1,2,4,8,12,16,24] 的 cudagraph 录制；仅 bs=1 就约 5 分钟）。健康检查需要 >300s — 越过 wait-server 超时手工轮询。
5. **跑精度**（concurrency=32）：**ori_acc=77.47%、normalized=96.83%、duration=2922.53s、total_tokens=1,294,965**。
   - 按任务：mcq=50.00%（avg_out=11086，runaway-think 持续）/ cwe=84.00% / fwe=100.00% / niah=100.00% / qa=53.33%。
   - 按桶：0_4k=50.00（仅 mcq）/ 4k_32k=87.25 / 32k_128k=82.87。
6. **跑速度全部档位**：**S1=110.68s / S8=40.33s / Smax=32.53s**（workflow 总结解析器显示 0/0 是已知 bug，日志中每行 `[S*] Benchmark duration:` 才是真实值）。
7. 通过 `pause-instance` 暂停实例。第一次 504 网关超时，重试后 200 — 实例已暂停，停止计费。

## 3. 结论

### 数字对比

| 指标 | 本次（`GPTQ-FP8-DENSE-retest-newinst`，ac91b1afe） | Test 12（v18-A 旧 fcloud 基线） | Tier1-B / 2A-bs24（前一台新 fcloud 上的 v22 参考） |
|---|---|---|---|
| ori_acc | **77.47%** | 79.29% | 78.73% / 79.11% |
| normalized | 96.83% | 99.11% | 98.42% / 98.89% |
| C | **0** | 1.0 | 0.96 |
| S1 | **110.68s**（≈ 比 Test 12 快 9%） | 121.71s | 110.56–111.36s |
| S8 | **40.33s**（≈ 快 9%） | 44.09s | 40.47–40.49s |
| Smax | **32.53s**（≈ 快 9%） | 35.86s | 33.36–33.62s |

### 解读

- **速度结果优秀且可复现**：S1/S8/Smax 与前一台新 fcloud 上 Tier1-B/2A-bs24 的扫描相差 ±0.5% 以内，表明 v22 server 配置完全等价、Mar-29 构建产物没有引入波动。
- **精度 77.47% / norm 96.83% 落在 C=0 区间（norm < 97%）**。
  - 与 Tests 29 (78.73%)、30、33 (78.73%)、34a (77.51%)、v18-revert (77.44%) 以及 Round 13f-4 四联跑（74.87–76.20%，均值约 75.6%）所观察到的"新 fcloud 地板"基本一致。
  - 单次 150 样 concurrency=32 的本地评测有 ±2-3pt 的噪声，这次落点在该带内。
  - **mcq runaway 是主导**：avg_out=11086、acc=50% — 与之前所有新 fcloud 精度波动同源。
- `copilot-instructions.md` 中的 official-vs-local 备忘要求本地至少 ≥ 80% 才能在 97%/98%/99% 三个门槛后留有私集漂移的安全冗余。**此单次跑没有提供这一冗余。**

### 决策

1. **不应将此结果当作 vs Test 12 的回归** — 旧 vs 新 fcloud 硅片差异加单跑噪声足以解释整个 gap。Tests 29/33/v18-revert/2A-bs24 在前一台新 fcloud 上同配置都给出 78–79%。
2. **v22 默认 server 配置（`SOAR_BACKEND_VARIANT=flashinfer` + 去掉 force-dense + 去掉 DAS）在速度上完全复现预期。**
3. **下一轮 GPTQ_FP8_DENSE 目录优化必须配套至少一次方差探针 2nd run**，否则无法把真正的优化与 ±2.5pt 地板区分开。
4. **NVFP4-FOS 按用户要求继续暂停**（CHANGE_0151_007 也得出速度分综合 96.0 vs 86.7 偏向 GPTQ 的结论）。
5. fcloud 实例已暂停；成本守则满足。

### 待确认 / 后续

- 下一轮是否值得开一次 `SOAR_BACKEND_VARIANT=minicpm_flashinfer` 来检验在本台 fcloud 硅片上是否真正等价于 Test 12？（Round 13f-4 四联跑显示两种 backend 在前一台新 fcloud 上落点相同，但本台尚未单独验证。）
- `pcre.py` shim 现已留在本 fcloud 的 venv 中持久存在；下一次 setup 时需在文档中提醒，避免再走一遍 4 步调试。
- mcq runaway-think（avg_out=11086）依然是单一精度杠杆中最大的，Iteration A-0 的生成停止工作仍是目录中 EV 最高的优化项。

## 4. 涉及文件 / 提交

- `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` — 追加 `GPTQ-FP8-DENSE-retest-newinst` 行。
- `docs/soar_2026_changes/chat/CHAT_gptq-fp8-dense-baseline-retest_20260509_1102.zh.md`（本文件）与对应 `.en.md`。
- 无源码改动。
- 提交并推送至 `minicpm-src/mixed_minicpm_cudagraph`，将文档与 chat log 一并入库。
