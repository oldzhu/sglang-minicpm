# 提案 — 复刻第 7 周冠军：NVFP4 FourOverSix + Medusa GLA verify

**日期**：2026-05-04
**作者**：Agent（待用户批准）
**前置版本**：v22（`SOAR_TORCH_COMPILE_MAX_BS=24` 默认开启，提交 `234f3fed8`）
**调研依据**：[RESEARCH_week7_champion_review_20260504.zh.md](RESEARCH_week7_champion_review_20260504.zh.md)
**风险**：高（跨层改动 —— 新量化流水线 + 新模型 head + 新注意力路径）
**预期收益**：目标 Smax ~16-22s（约 30-50% 加速），官方分 30 → 50-65 进入前 5/6 名

## 0. 为什么这是下一步

- 本地服务参数扫描已饱和（#2-A 仅 3% Smax 提升；v18→v22 官方近乎持平）。
- 榜单 1-2 名（88.35 / 86.68）单次提交跃升 +7-10 分，走的就是这个组合。
- SM120 FP4 算力 **593 TFLOPS** = BF16（148）的 4 倍，FP8（296）的 2 倍。NVFP4 是唯一能榨出该算力的路径（见 [SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)）。
- Medusa 100% 针对 decode，并在 **三档** 速度上都有收益 —— 直击 Smax 主导项。

## 1. 目标（双轴并进）

**A 轴 — 量化升级**：将权重从当前 GPTQ W4A16（sparse_qkv_w8）切换到 **NVFP4 + FourOverSix 自适应 scale**，保留 FP8_e5m2 KV cache 与 dense 模式。

**B 轴 — 推测解码**：训练并部署 Medusa 风格的 head（K=1，深度 1 树），通过 GLA-state-aware 的 tree verify 接入 MiniCPM-SALA 混合注意力。

## 2. 合规检查

| 规则 | 状态 |
|------|------|
| 量化必须**现场**完成、5h 总预算内 | NVFP4 GPTQ + 90 校准样本 ≈ 90-180 分钟（与现有 GPTQ_minicpm_sala 校准量级相同）。FourOverSix 是逐 block 的判断（每个 ~ms），总共增加几分钟。✅ |
| 提交包 ≤ 2GB | NVFP4 权重 ≈ 当前 GPTQ 的 0.5×；Medusa head 视宽度 30-100MB。远低于 2GB。✅ |
| 允许推测 head | 官方明确允许（"speculative heads count toward 2GB"）。✅ |
| 可复现 / 可解释 / Apache 2.0 | sglang 上游 Medusa 与 Marlin NVFP4 都是 Apache 2.0；FourOverSix 是开源算法（论文+代码 arXiv:2512.02010）。✅ |
| 评测接口不变 | 仅改权重 + 新增 head；eval 脚本不动。✅ |

## 3. 风险表

| 风险 | 严重度 | 缓解 |
|------|-------|------|
| NVFP4 + Marlin kernel 路径暂不支持我们的 `sparse_qkv_w8` 混合方案 | 高 | A 阶段先出 **统一** NVFP4（去掉 W8 QKV 混合）；对比准确率。底线达标后再回头看混合方案。 |
| FourOverSix 提升不足 → C 跌破 0.96 | 中 | 重新启用 FP16 lm_head + FP8 KV（v22 基线已具备）。极端情况回退 GPTQ W4A16。 |
| Medusa head 训练耗时 / 需 GPU days | 中 | 用最轻的 K=1（单 head，深度 1）—— 参数极少（~20-50M）。8×A100/H100 训练 6-12h 或租用。在 50-200K-token 语料上从主模型蒸馏。 |
| Tree verify GLA state-fork 在长上下文上破坏正确性 | 高 | 严格单元测试：100 条长上下文样本对比 Medusa-on vs Medusa-off 的逐 token 输出，要求 bitwise 一致（Medusa 拒绝则 fallback 到主 token，必须等于纯主路径）。 |
| 加上 head + 新 wheel 后包超 2GB | 低 | NVFP4 比当前 GPTQ 节省 ~3.5GB；head 体积可忽略。 |
| 整体周期太长 | 中 | A 阶段（仅 NVFP4）即可独立发版；B 可后接。先把 A 轴分数锁定。 |

## 4. 分阶段计划（4 阶段，每段独立可发版）

### Phase A —— NVFP4 基线（不带 FourOverSix，不带 Medusa）
**目标**：确认 NVFP4 + Marlin W4A16 路径在我们模型上端到端跑通，准确率 ≥ 78%。

待改文件（估）：
- `benchmark/soar/demo_sala/preprocess_model.py` —— GPTQ 格式切换为 `nvfp4`。
- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` —— 添加 NVFP4 量化配置分支（或使用上游 gptqmodel NVFP4 实现）。
- `benchmark/soar/demo_sala/prepare_env.sh` —— 把 `MODEL_PATH` 指向 NVFP4 模型目录；确保 quant 标志正确；移除 `--quantization gptq`，替换为 sglang 的 NVFP4 标识。
- 添加 env 开关 `SOAR_QUANT_PROFILE={gptq,nvfp4,nvfp4_fos}` 以便本地 A/B。

验证：
1. 90 样本校准在 ~3h 内顺利完成；
2. server 启动；`/v1/models` 报告 nvfp4；
3. 本地准确率 ≥ 78%（基线底）；
4. 本地速度：Smax 预期持平或略快（Marlin NVFP4 与 W4A16 同 kernel 类；SM120 上 GDDR7 BW 主导）。

决策：准确率达标即作为 v23 发版；否则回退 v22。

### Phase B —— FourOverSix 在 NVFP4 之上
**目标**：自适应 scale 选择恢复 ~0.5-1pt 准确率。

待改文件：
- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` —— 在 GPTQ 迭代之前注入 per-block scale 比较：
  ```python
  scale_m6 = compute_block_scale(W_block, M=6)
  scale_m4 = fp8(scale_m6.float() * 1.5)
  err_m6 = mse(W_block, dequant(W_block, scale_m6))
  err_m4 = mse(W_block, dequant(W_block, scale_m4))
  scale = scale_m4 if err_m4 < err_m6 else scale_m6
  # 然后在该 scale 框架内进行 GPTQ
  ```
- env 开关：`SOAR_QUANT_PROFILE=nvfp4_fos`。
- 记录每层 M=4 比例（冠军报告 40-43%）。

验证：
1. M=4 比例在 35-50%；MLP 层 > attn QKV；
2. 准确率较 NVFP4 单跑提升 ≥ 0.3pt；
3. 吞吐与 NVFP4 相同（kernel 不变）。

如果准确率提升被确认，发版 v24。

### Phase C —— Medusa K=1 head 训练（离线，非 fcloud）
**目标**：训练 1 个 Medusa head，把 hidden_size 投射到 vocab，预测 t+1 token，GLA-state-aware（先不做 GLA fork —— 直线单 token）。

硬件：需要 H100/A100 训练机 ~12-24h。Fcloud 仅有 RTX 6000D Blackwell，不适合，需自有/租用机器。

流程：
1. 冻结主模型（NVFP4-FoS 量化版或 BF16 —— 两种顺序都需要试）；
2. 构造数据集：SOAR 分布代理（公开来源长上下文 QA —— RULER、LongBench）；
3. W₁ 全零初始化训练 head 约 5-10K steps，目标 LM-CE loss 作为辅助信号；
4. 蒸馏 checkpoint：~30-80MB。

新增文件：
- `benchmark/soar/demo_sala/medusa_head/` —— head 模块 + 训练脚本；
- `benchmark/soar/demo_sala/medusa_train.py` —— 入口；
- head checkpoint 打入提交 tarball。

验证：保留集长上下文样本上 top-1 接受率 ≥ 0.5。

### Phase D —— Medusa tree-verify + GLA state fork（sglang 服务端）
**目标**：将训好的 head 接入 sglang 的 MiniCPM-SALA 后端，实现正确的 GLA state 分叉。

待改文件（最大改动量）：
- `python/sglang/srt/models/minicpm_sala.py` —— 加载 Medusa head；把最后一层 hidden state 经 head 输出；扩展 forward 接受 `tree_input_ids` 与 `tree_position_ids`。
- `python/sglang/srt/layers/attention/minicpm_backend.py` —— 每次 verify forward：
  - 检测 tree 输入（多个候选位置）；
  - 每个 GLA 层：保存 `h_parent` 快照；每个分支独立从 `h_parent` 跑递推；收集分支输出；
  - 标准 MHA 层：使用上游 tree mask（sglang Eagle/Medusa 路径已支持）。
- `python/sglang/srt/speculative/medusa_*.py` —— 接入 scheduler；产生候选；汇总接受。
- `benchmark/soar/demo_sala/prepare_env.sh` —— 添加 `--speculative-algorithm medusa`、`--speculative-draft-model-path /root/.../medusa_head` 等。

验证：
1. Bitwise-equivalence 测试：100 条 prompt，Medusa-off vs Medusa-on（temperature=0），输出 **必须** 逐 token 一致；
2. 接受率 ≥ 0.4；
3. 本地 Smax ≤ 25s（vs 当前 32.54s）→ 23% 提升目标；
4. 准确率底线保持。

发版 v25。

## 5. 文件改动汇总

| Phase | 文件 | 类型 |
|-------|------|------|
| A | `benchmark/soar/demo_sala/preprocess_model.py` | 编辑 |
| A | `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` | 编辑 |
| A | `benchmark/soar/demo_sala/prepare_env.sh` | 编辑（添加 `SOAR_QUANT_PROFILE` env gate，切换 quant 标志）|
| B | `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py` | 编辑（FourOverSix 块）|
| C | `benchmark/soar/demo_sala/medusa_head/` | 新建 |
| C | `benchmark/soar/demo_sala/medusa_train.py` | 新建 |
| D | `python/sglang/srt/models/minicpm_sala.py` | 编辑（接入 head）|
| D | `python/sglang/srt/layers/attention/minicpm_backend.py` | 编辑（GLA fork）|
| D | `python/sglang/srt/speculative/medusa_*.py` | 编辑/新增 |
| D | `benchmark/soar/demo_sala/prepare_env.sh` | 编辑（speculative 参数）|

## 6. 各阶段测试命令

### Phase A
```
# 离线（本地机器，GPTQ 容器）
python3 benchmark/soar/demo_sala/preprocess_model.py --quant nvfp4 --input <bf16> --output <nvfp4>
# fcloud
SOAR_QUANT_PROFILE=nvfp4 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### Phase B（同 A，profile=nvfp4_fos）

### Phase C（非 fcloud）
```
python3 benchmark/soar/demo_sala/medusa_train.py \
  --base-model <bf16-or-nvfp4-fos> \
  --train-data ruler+longbench-proxy.jsonl \
  --num-heads 1 --depth 1 \
  --output-dir medusa_head/
```

### Phase D
```
SOAR_QUANT_PROFILE=nvfp4_fos SOAR_MEDUSA_ENABLE=1 \
  python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 7. 各阶段成功 / 失败判据

| Phase | 通过 | 失败-回退 |
|-------|------|----------|
| A | acc ≥ 78%、Smax ≤ 35s | 回退 v22；调查 Marlin NVFP4 路径 |
| B | 较 Phase A 提升 ≥ +0.3pt、M=4 比例在 35-50% | 不要 FoS，把 Phase A 作为 v23 发版 |
| C | head 接受率 ≥ 0.4 | 增加数据 / 换冻结顺序后重训 |
| D | T=0 下 Medusa-off vs Medusa-on 逐 token 一致；Smax ≤ 25s；acc ≥ 78% | 撤销 speculative server 参数；把 Phase B 作为 v24 发版 |

## 8. 各阶段回滚

每个阶段以 env-gate 独立发版。最坏情况只需 unset `SOAR_QUANT_PROFILE` / `SOAR_MEDUSA_ENABLE`、重启，回到上一发版。

## 9. 投入估算

- Phase A：3-7 天（主要是 NVFP4 路径调试与精度调）。
- Phase B：1-2 天（小算法添加）。
- Phase C：2-5 天，含数据 + 训练 + 调参。
- Phase D：5-10 天（sglang Medusa wiring + GLA fork 是最难一段）。

## 10. 各阶段失败的下一步

- A 失败（NVFP4 路径坏）：上游 sglang 提 issue；同时尝试 **AWQ-NVFP4** 备选。
- B 失败（FoS 收益边缘）：不阻塞；A 直接发版 v23。
- C 失败（head 接受率不够）：转而尝试 **EAGLE / EAGLE3** —— sglang 已支持，且在长上下文 decode 上有类似收益。
- D 受阻于 GLA fork：先把 Medusa 限制在仅 MHA 层；收益变小但仍正向。

## 11. 交叉引用

- 调研依据：[RESEARCH_week7_champion_review_20260504.zh.md](RESEARCH_week7_champion_review_20260504.zh.md)
- 冠军博客：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- FourOverSix 论文：arXiv:2512.02010
- Medusa 论文：ICML 2024（Cai 等）
- SM120 硬件：[SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)
- 优化目录：[OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- 前置版本：v22（`234f3fed8`）
- 排行榜：team-beta #22 分 30.04；前 5 门槛 50.66（差 +68%）

---

## 待批准

回复 **approve A** 即从 Phase A（NVFP4 基线）开始 —— 风险最低、最容易先发版。
回复 **approve all** 即提前承诺 A-D 完整方案。
回复 **adjust** 给出阶段重排或缩减。

建议：**先批准 A**。仅 NVFP4 一项，如果 Marlin NVFP4 在 SM120 上跑通顺，可能就把分数从 30 推到 40+，为 Phase D 的深度改动赢得余裕。
