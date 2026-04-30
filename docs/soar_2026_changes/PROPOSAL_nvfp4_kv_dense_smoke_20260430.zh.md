# 提案 — NVFP4 KV 缓存稠密模式冒烟测试

**日期**：2026-04-30
**状态**：提案 — 等待用户批准
**前置文档**：
- `SURVEY_NVFP4_KV_P1_20260428_1130.en.md`（P1 plumbing 调研，已完成）
- `CHANGE_0131_nvfp4_kv_p2_plumbing.{en,zh}.md`（env 开关已落地）
- `CHANGE_0132_nvfp4_kv_force_dense_compat.{en,zh}.md`（稠密兼容补丁已草拟）
- `PROPOSAL_iteration_M20_kv_fp4_ablation.{en,zh}.md`（逐层敏感性消融 — 推迟到后续迭代）
**姊妹迭代**：`PROPOSAL_tier1_long_context_retest_20260430.{en,zh}.md`（在本次之前执行）

## 1. 背景

NVFP4（MXFP4，`fp4_e2m1`）是 SM120（Blackwell，RTX 6000D）原生支持的格式。P1 调研发现：

- 走线 80 % 已在主干：`prepare_env.sh:155` 已有 `SOAR_FP4_KV_CACHE=1` 开关（CHANGE_0131）；`MHATokenToKVPoolFP4` 在上游 sglang。
- MiniCPM 自定义 attention backend 中有 4 处 gate 当前判 `kv_cache_dtype_str.startswith("fp8")`，需要扩展支持 `fp4_e2m1`，调研已逐处定位。
- `--force-dense-minicpm` 直接绕过其中 2 处稀疏 gate。稠密冒烟是最干净的第一道信号。

KV 缓存内存节省：**FP8 → FP4 ≈ 每 token KV 字节 −44 %**。在长上下文速度集（68 % 输入 32K–512K）上直接打中我们落后最多的 Smax 档（允许更大并发批）。

## 2. 目标

在 fcloud 上端到端跑一次最简单的 NVFP4-KV 配置：

- `SOAR_FP4_KV_CACHE=1`（`prepare_env.sh:155` 已联通）
- `--force-dense-minicpm`（v20 默认开启）
- 其他 Tier 1 调度 flag 继承自 #1 选定的 baseline（v20 或 v21）。

本次迭代要回答三个问题：

1. **能否启动**：FP4 KV 下服务器是否干净启动？
2. **精度**：端到端精度是否守住安全阈值？
3. **速度**：Smax 是否有可测量的提升？（S1 / S8 持平亦可）

## 3. 规则合规

- 不改模型权重，不改现场量化流程（GPTQ pipeline 不变）。
- 提交包格式不变；通过 env 开关 opt-in。
- KV 在运行时计算 — 完全现场、可复现、Apache-2.0。

## 4. 风险

| 风险 | 缓解 |
|---|---|
| `MiniCPMAttentionBackend` 4 处 gate 没扩展导致启动失败 | 调研已点出 4 处；CHANGE_0132 稠密兼容补丁已草拟 — 冒烟前先打上。`grep -n 'fp4_e2m1' python/sglang/srt/layers/attention/minicpm_backend.py` 验证。|
| FP4（1 mantissa + 2 exp + sign）vs FP8 KV，精度回归 > 5pt | 硬阈值：**acc_ori ≥ 75 %** 且 normalized ≥ 97 %（C 档下沿）。低于则立即终止。当前精度有 ~5pt 余量（80 %）。|
| `KVFP4QuantizeUtil` 用 `@torch.compile`，可能与 cudagraph 抓取冲突 | 调研已标记；若 cudagraph 抓取失败，本次去掉 `--enable-torch-compile` 再测（更慢但能隔离问题）。|
| Smax 因更大批次 OOM | `mem-fraction-static 0.84` 不变；FP4 下 KV 更小，OOM 风险其实更低。|
| `set_kv_buffer` 通过 `layer.k_scale` 二次缩放（调研 Gap D）| 验证 `kv_cache_dtype == "fp4_e2m1"` 时 `layer.k_scale` 为 None；否则在冒烟测试里强制置 None。|

## 5. 修改文件

| 文件 | 修改 |
|---|---|
| `python/sglang/srt/layers/attention/minicpm_backend.py` | 把调研定位的 4 处 gate（A、C、D）扩展也对 `fp4_e2m1` 触发。具体 diff 参考 CHANGE_0132。|
| `benchmark/soar/demo_sala/prepare_env.sh` | 无代码改动 — `SOAR_FP4_KV_CACHE` 已联通；执行器导出。|
| （无模型预处理改动） | KV 每步算；权重不需要重量化。|

## 6. 验证命令

```bash
# 预检：确认 FP4 gate 补丁已落
ssh fcloud "grep -nE 'fp4_e2m1|kv_cache_dtype_str' /root/submission_sim/sglang/python/sglang/srt/layers/attention/minicpm_backend.py | head -20"

# 同步代码
python3 scripts/fcloud/fcloud_workflow.py sync

# baseline 保险（FP8 KV — 同一天 #1 step A 跑过即可复用）

# FP4 KV 候选
ssh fcloud "cd /root/submission_sim && export SOAR_FP4_KV_CACHE=1 && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 1) 启动检查 — server log 干净？（无异常、无 NaN）
python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 200

# 2) 精度
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 3) 速度
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

总 fcloud 时长估计 ~30 min（1 × acc + 1 × 速度）。

## 7. 成功 / 失败判据

| 结果 | Acc | S1 | S8 | Smax | 决策 |
|---|---|---|---|---|---|
| **WIN** | ≥ 78 %（回退 ≤ 2pt） | 持平 ± 3 % | 持平 ± 3 % | ≤ 0.95 × baseline | 上线下一版；开启逐层 M2.0 ablation 提案。|
| **精度勉强守住** | 76–78 % | 持平 ± 3 % | 持平 ± 3 % | ≤ 0.95 × baseline | 暂缓；先做 M2.0 ablation 恢复 1–2pt 再决定上线。|
| **精度回归** | < 76 % 或 normalized < 97 % | — | — | — | 暂搁置 NVFP4 KV，等 ablation。文档化。|
| **速度回归** | ≥ 78 % | — | — | ≥ 1.05 × baseline | 意外；检查 prefill kernel FP4 反量化开销；非平凡则放弃。|
| **启动 / cudagraph 失败** | — | — | — | — | 记录是哪个 gate；若是 cudagraph，去掉 `--enable-torch-compile` 重测。|

（baseline = 同一天 v20 或 v21 选定 baseline 的 FP8 KV 运行结果。）

## 8. 回滚

`unset SOAR_FP4_KV_CACHE` 并重新 source `prepare_env.sh`。MiniCPM backend gate 补丁在 FP8 路径上是 no-op（只是扩展字符串匹配），即使回滚也安全保留。

## 9. 后续步骤

- **WIN** → 开 `PROPOSAL_iteration_M20_kv_fp4_ablation`（已草拟）做逐层敏感性消融，回收剩余精度同时保留内存红利。
- **WIN + #1 也赢** → 合并提交 v22 = Tier 1 + NVFP4 KV，更新 leaderboard memory。
- **回归** → 回目录挑 #3（**不**走 FP8 W8A16 — 该路径已关闭）。

## 10. 待用户确认的点

主干已存在 `PROPOSAL_iteration_M20_kv_fp4_ablation.{en,zh}.md`，那是直接做逐层敏感性消融的提案。**本次提案是更简单的稠密冒烟，应当先跑。** 若稠密冒烟暴露灾难性精度损失，逐层 ablation 是自然的后续。请确认这个顺序。
