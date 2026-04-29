# CHAT — Round 13e Option-B 稀疏线 profile (2026-04-29 ~12:12 本地)

`CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.{en,zh}.md` 的配套对话记录。

## 用户诉求(本轮)
1. "approve" —— 批准 CHANGE_0135(从 `prepare_env.sh` 中移除 `--dense-as-sparse`,并在稀疏路由下做 32k / 64k / 128k 单请求 torch profile)。
2. "go fcloud started" —— fcloud 实例已启动,继续跑 profile。

## Agent 行动
1. `git push minicpm-src mixed_minicpm_cudagraph` —— 推送 CHANGE_0135(commit `f4097eef6`:文档对 + `prepare_env.sh` 清理),让 fcloud 可以拉取。
2. `python3 scripts/fcloud/fcloud_workflow.py sync` —— 同步。
3. 清理旧的 `/root/profile_round13e/`;杀掉残留 server。
4. 写入 `/root/launch_profile_server.sh`(source `prepare_env.sh`,设置 `SOAR_QUANT_MODE=noquant` + `SGLANG_TORCH_PROFILER_DIR=/root/profile_round13e/`,exec sglang launcher)。
5. 第一次启动失败:`command 'clang' failed: No such file or directory`,出在 `uv pip install --force-reinstall pypcre`。原因:Python 3.10.19 sysconfig 里 `CC=clang -pthread`,而 fcloud 镜像里只有 `gcc`。临时变通:`ln -sf /usr/bin/gcc /usr/local/bin/clang && ln -sf /usr/bin/g++ /usr/local/bin/clang++`。之后 server 36 s 内 ready 健康。
6. 本地写好 `profile_driver.py`,base64 上传成 `/root/profile_driver.py`。Driver 从 `/root/data/perf_public_set.jsonl` 中取 `prompt_tokens` 最接近目标的样本,用 `/start_profile` + `/stop_profile` 包住 `/generate` POST,`max_new_tokens=64`。
7. 跑了三个单请求 profile:32k(样本 129,9.7 s)、64k(样本 49,24.7 s)、128k(样本 149,49.3 s)。trace 重命名为 `round13e_<size>.trace.json.gz`(56 / 112 / 211 MB)。
8. 上传 `analyze_profile.py` 后台运行,把每个 trace 的 Top-25 + 类别 breakdown 收集到 `/root/analyze.out`,复制到 `docs/soar_2026_changes/profile_data/round13e_analyze.txt`。
9. 按"省钱规则"运行 `python3 scripts/fcloud/fcloud_workflow.py shutdown`。
10. 写出 `CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.{en,zh}.md`(本续档对),记录完整 breakdown、决策树落实、建议。

## 结果 / 关键数据

| 长度 | wall  | GPU kernel 总时间 | BF16 256×128 GEMM | 稀疏 FA | 稀疏元数据+topk | GPU 活跃 |
|------|-------|--------------------|--------------------|---------|------------------|----------|
| 32k  |  9.7s | 5 658 ms          | 66.7%              | 3.6%    | ~2%              | 58%      |
| 64k  | 24.7s |10 344 ms          | 73.2%              | 4.0%    | ~2%              | 42%      |
| 128k | 49.3s |19 775 ms          | 76.9%              | 4.3%    | ~2%              | 40%      |

- 瓶颈 = BF16 cuTLASS GEMM(全精度的线性 / MLP 投影权重),**不是**稀疏注意力。
- 稀疏注意力本身在三种长度下都 ≤4.3% GPU 时间,线性扩展,健康。
- compress_k1/k2 fill(CHANGE_0133 bounded 修复后)三种长度下都 1.0–1.1%,修复足够。
- CHANGE_0135 决策树落到 **分支 (d):BF16 权重才是成本** → SOAR 提交关闭 BF16+sparse 线,保留 dense + GPTQ + FP8 KV baseline (Test 12) 作为提交路径。
- 可选后续(中期研究,不影响下一次提交):
  - GPTQ + sparse + FP8 KV,重新调 calibration(规避历史上 sparse_qkv_w8 50% 精度崩溃)。
  - 稀疏路由下全部层用 SM120 mxfp8 / W8A8 权重。
- 尾部内核清理(8% "其它")暂缓 —— 与 77% GEMM 主瓶颈相比上限有限。

## 待办
- 把 clang→gcc symlink 变通并入 `prepare_env.sh`,或在 `submission_sim.tar` 里预打 pypcre wheel,免得每次干净 fcloud 镜像跑 noquant 都要手动修。
- Trace 文件(共约 379 MB)留在 fcloud `/root/profile_round13e/`,未下载。后续若要 TensorBoard 深入查看,可用 `profile_driver.py` 复现。

## 交叉引用
- 提案:[CHANGE_0135_sparse_path_cleanup_and_profile_plan.en.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan.en.md)、[.zh.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan.zh.md)
- 本轮结果:[CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md)、[.zh.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.zh.md)
- 原始 kernel 数据:[profile_data/round13e_analyze.txt](../profile_data/round13e_analyze.txt)
- 相关历史:CHANGE_0133(compress_k bounded fill)、CHANGE_0134(排除 tokenizer mismatch 是 Round 13e 超时原因)。
- 优化清单:[OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- Submission baseline 参考:`TEST_RESULTS_TRACKING.md` Test 12(S₁=121.71 s, S₈=44.09 s, S∞=35.86 s, ori_acc=79.29%, C=1.0)。
