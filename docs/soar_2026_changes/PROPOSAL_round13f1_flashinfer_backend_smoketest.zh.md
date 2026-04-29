# PROPOSAL — Round 13f-1:Test 12 baseline 上 `--attention-backend flashinfer` 兼容性快测

## 状态: 提案(待批准)

## 1. 目标

快速兼容性 + 速度 sanity 检查:把 GPTQ + FP8 KV + dense 提交 baseline (Test 12) 改用官方默认的 `--attention-backend flashinfer`,会不会比当前 `--attention-backend minicpm_flashinfer --force-dense-minicpm` 有任何**可测量**的差异?用户回忆早期跑非量化原模型时用 `flashinfer` "好像没问题",这次想确认它是一条可用的更简单配置,还是会悄悄把 lightning-attn / sparse_attention 层路由错。

这**不是** feature 提交,是一次 exploration 测试。结果要么变成 CHANGE,要么记录成 dead-end。

## 2. 假设

- **速度:** 大概率没有可见提升。`--force-dense-minicpm` 下每层"是不是 sparse 层"的判断只是一次布尔读,dispatch 本身基本零成本,没有可消除的"router 开销"。
- **正确性:** **不确定。** MiniCPM-SALA 有 24 个 lightning-attn(线性 / Mamba 风格)层 + 8 个 sparse_attention 层。`minicpm_flashinfer` 后端同时承载 **lightning-attn / FLA / SimpleGLA** 路径 (`HybridLinearAttnBackend`)。纯 `flashinfer` 后端没有这些 hybrid hook。两种可能失败方式:
  1. server 直接起不来(模型注册的某层 flashinfer 不能 dispatch)。
  2. server 起来了但精度崩(lightning-attn 层悄悄退化成 full KV 上的普通 MHA — 数学不对)。
- **最好的合理结果:** 速度和 Test 12 完全一致、精度也一致 → 确认 `flashinfer` 只是 dense 路径的别名,可以用官方默认配置跑。价值边际,但验证成本低。

## 3. 规则合规

- 不改模型文件。不改 eval 脚本。只在 `prepare_env.sh` 切换 server arg。
- 与提交打包兼容(只是 server arg)。
- 不引入新的量化 / kernel / 调度改动。

## 4. 涉及文件

- `benchmark/soar/demo_sala/prepare_env.sh` — 加一段临时 env 门控分支:当 `SOAR_BACKEND_VARIANT=flashinfer` 时输出 `--attention-backend flashinfer`,否则保持 `--attention-backend minicpm_flashinfer --force-dense-minicpm`(Test 12 配置不变)。
- 不动 `python/sglang/srt/`。

## 5. 验证计划

1. 同步到 fcloud(不重建 wheel)。
2. 设 `SOAR_BACKEND_VARIANT=flashinfer` 启动 server,其他配置完全保持 Test 12(GPTQ + FP8_e5m2 KV + dense + torch.compile bs=8 + Test 20 server args)。
3. **Tier 1(启动 smoke):** server 能否在 60s 内健康?如果不能 → 记录错误类型,中止,标 dead-end。
4. **Tier 2(精度 smoke):** server 起来,跑 `--max-concurrent 8` accuracy(便宜变体)。如果 `ori_accuracy < 75%` → 中止,dead-end。
5. **Tier 3(完整):** Tier 2 通过则跑完整 accuracy + S₁/S₈/S∞。
6. 与 Test 12 对照(S₁=121.71s, S₈=44.09s, S∞=35.86s, ori_acc=79.29%)。

## 6. 通过 / 失败标准

- **通过(作为备选):** ori_accuracy 与 Test 12 ±1pt 以内,且各档速度与 Test 12 ±2% 以内。只在打包/官方默认有偏好时才采用。
- **中性(记录后丢弃):** 完全一致 — `flashinfer` 只是 dense 路径的别名,无收益,baseline 不变。
- **失败(dead-end):** server 起不来,或精度跌 > 1pt,或任一档速度回退 > 2%。停止,在 TEST_RESULTS_TRACKING 写一段症状说明,不再追。

## 7. 风险

- **对当前 baseline 风险:零。** `prepare_env.sh` 默认分支不变,只新增一条 env 门控的非默认分支。
- **成本:** 约 1 轮 fcloud(1 次 boot + 1 次 accuracy + 3 次 speed)。

## 8. 回滚

如果觉得这条 env 分支太脏,直接从 `prepare_env.sh` 删掉。无源码改动可回滚。

## 9. 后续建议

本提案与 `CHANGE_0136_minicpm_sparse_dense_len_flag.{en,zh}.md`(Option 3 — 真正有意思的方向)并列。Round 13f-1 通过与否都不影响 Round 13f-2 / CHANGE_0136;如果失败,继续保持 `minicpm_flashinfer --force-dense-minicpm`。
