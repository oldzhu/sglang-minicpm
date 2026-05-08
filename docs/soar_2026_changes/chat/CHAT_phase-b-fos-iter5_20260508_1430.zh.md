# 会话 — Phase B FOS iter-5 + fcloud setup git-clone 修复

时间：2026-05-08 14:30（开始）
主题：NVFP4-FOS Option A（重新量化 SAMPLES=32 sequential），并把 fcloud
setup 的 tarball 上传替换为真正的 git clone。

## 用户提问（第 1 轮，原文）

> go and besides that, pls also git clone from https://github.com/oldzhu/sglang-minicpm.git
> at /root/sglang-minicpm so that we could git pull and synce from the
> remote repo which our local change pushed default. Pls also add a note
> into the fcloud instance environment re-setup so that furure re-setup
> would create the clone at /root/sglang-minicpm too.

拆解为 3 个并行目标：
1. **go** = 继续 iter-4 后续 Option A（重新量化 `SAMPLES=32 sequential`，
   还原 iter-1 的校准设定）。
2. 把 `/root/sglang-minicpm`（tarball 解压、缺 `.git`）替换为真正的 git
   clone，使 `fcloud_workflow.py sync` 走 `git pull`。
3. 更新 setup 脚本与文档，使后续重建实例时一直创建真正的 clone。

## Agent 行动

| 步骤 | 行动 | 结果 |
|------|------|------|
| 1 | 修改 `scripts/fcloud/fcloud_workflow.py::step_setup` Step 2，把 tarball 上传 + 解压改为 `git clone --depth 1 --branch mixed_minicpm_cudagraph https://github.com/oldzhu/sglang-minicpm.git /root/sglang-minicpm`（超时 600 秒） | 本地修改 ✓ |
| 2 | 修改 `.github/copilot-instructions.md`：更新 setup Step 2 + 增加缺 `.git/` 实例的人工补救步骤 | 本地修改 ✓ |
| 3 | `start-instance`（控制台 API） | HTTP 200，~3 分钟后 JupyterLab READY |
| 4 | 人工补救：`rm -rf /root/sglang-minicpm && git clone --depth 1 …` | 第一次 exit 0 但 `.git/` 半成品；以后台任务方式重试，最终 `git rev-parse HEAD = 6ae5ef33f59a7507e9a5380f6b49e08159ae2815`（与最近一次 push 相同）✓ |
| 5 | 改进 `step_sync`：处理 pre==post sha 的情形（无新提交时返回 `(no new commits)`，不再回退到 force-copy） | 本地修改 ✓ |
| 6 | 验证 `fcloud_workflow.py sync` | `[git pull] Already up to date.` + `(no new commits)` ✓ |
| 7 | commit + push 上述两个补丁 | `e41ecb518` 已推送到 `minicpm-src/mixed_minicpm_cudagraph` ✓ |
| 8 | push 后再 `sync` 一次 | 拉取 `e41ecb518`，正确报告 `[changed files]`，无 force-copy ✓ |
| 9 | iter-5 量化：后台执行 `preprocess_model.py --mode nvfp4`，环境变量 `SAMPLES=32 SAMPLING=sequential MAX_CALIB_SEQ_LEN=4096 FOS=1` | ~1 分钟完成；pct_m4=43.14%；**但 tokenizer 文件缺失**于 dst ✓⚠ |
| 10 | 第 1 次启动 sglang（`--quant-mode noquant`） | 失败：`ValueError: Unrecognized configuration class …MiniCPMSALAConfig`（tokenizer 缺失）|
| 11 | 人工修复：从 `/root/models/openbmb/MiniCPM-SALA` 拷贝 `tokenizer*` + `special_tokens_map.json` 至 dst；重跑 `_patch_chat_template_for_mcq` | ✓ |
| 12 | 第 2 次启动 sglang（`--quant-mode noquant`） | 失败：`ModelOptFp8Config only supports static FP8 quantization`（`noquant` 分支不带 `--quantization`） |
| 13 | 第 3 次启动 sglang（`--quant-mode gptq`，prepare_env 会根据 `SOAR_QUANT_PROFILE=nvfp4_fos` 把 flag 切换为 `--quantization modelopt_fp4`） | 服务 READY ✓ |
| 14 | accuracy run-1（仅一次） | **71.24%** ori_accuracy ✓ |
| 15 | 暂停实例 | 第 1 次 504，重试 200 ✓ |
| 16 | 在 `CHANGE_0151_phase_b_four_over_six_004.{en,zh}.md` + 本会话日志 + TEST_RESULTS_TRACKING（行 `NVFP4-FOS-5`）中记录 | 本次 commit |

## 结论

- **iter-5 结果**：71.24% ori_accuracy，通过 70% abort gate。比 iter-4
  （66.00%）+5.24pt，距 iter-1（~73.13%）仍差 ~1.9pt。
- **结论**：在 qa,mcq,cwe 这一受限校准池里，`SAMPLES=32 sequential` 明显
  优于 `SAMPLES=90 stratified`。校准**内容/选样**才是主因；calib_seq_len
  （iter-4）属于伪线索。
- **基础设施收获**：`fcloud_workflow.py sync` 在真正 clone 之后已能端到端
  使用 `git pull`，force-copy 仅作为兜底；后续重建实例会自动创建真正的
  clone。
- **遗留问题**（带入下一轮）：`preprocess_model.py` 的 NVFP4 流式导出路径
  没有保存 tokenizer，需要人工 cp。建议 iter-6 中作为 Option C 修掉。

## 本轮新增/修改的文件

| 路径 | 类型 | 简述 |
|------|------|------|
| `scripts/fcloud/fcloud_workflow.py` | 代码（已 commit `e41ecb518`） | step_setup 改用 git clone；step_sync 妥善处理无新提交 |
| `.github/copilot-instructions.md` | 文档（已 commit `e41ecb518`） | setup Step 2 + 人工补救步骤 |
| `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_004.en.md` | 新增（本次 commit） | iter-5 结果、对比、下一步选项 |
| `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_004.zh.md` | 新增（本次 commit） | 中文同步 |
| `docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5_20260508_1430.{en,zh}.md` | 新增（本次 commit） | 本会话日志 |
| `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` | 追加 `NVFP4-FOS-5` 行（本次 commit） | 71.24% |

## 待办

1. （代码）`run_nvfp4_quantization` 中补 `tokenizer.save_pretrained(dst)`，
   使 NVFP4 量化产物自包含。
2. （测试）iter-5 run-2 + S1/S8/Smax 速度评测，确认精度可复现并测量 FOS-32
   checkpoint 的速度足迹。
3. （测试）iter-6 = 重新量化 `SAMPLES=32 sequential FOS=0`；若达到 ≥73%，
   则 FOS 本身就是残差回退源，应永久搁置。

## 交叉引用

- iter-4 文档: [CHANGE_0151_phase_b_four_over_six_003.zh.md](../CHANGE_0151_phase_b_four_over_six_003.zh.md)
- iter-5 文档（本轮）: [CHANGE_0151_phase_b_four_over_six_004.zh.md](../CHANGE_0151_phase_b_four_over_six_004.zh.md)
- setup 脚本提交：`e41ecb518`（`minicpm-src/mixed_minicpm_cudagraph`）
