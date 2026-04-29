# 提案 — Round 13f-4:在宣布 Round 13f-1 死亡之前先做方差量化

## 状态:PROPOSAL(待审批)

## 背景

Round 13f-1(`SOAR_BACKEND_VARIANT=flashinfer`,无 force-dense,无
dense-as-sparse)测得 ori_acc=76.91%,速度大幅提升(S₁=110.76 / S₈=40.50 /
Smax=33.66;较 Test 12 −9% / −8% / −6%)。13f-2(75.80)和 13f-3(74.73)
继续叠加 flag,准确率反而进一步下降;**但**这三轮全部是单次跑。

把 13f-1 判为 C=0 死亡,所依赖的对比是 13f-1 的 76.91% vs **Test 12 老 fcloud 上的 79.29%**。这个参考来自旧实例。在当前新 fcloud 实例上,
**完全相同的 Test 12 配置**已经被复测过 4 次:

| Test | 日期 | 配置 | acc |
|---|---|---|---|
| Test 12(old fcloud) | 2026-04-12 | 参考 | 79.29% |
| Test 29 | 2026-04-22 | 新 fcloud baseline | 78.73% |
| Test 30 | 2026-04-22 | KV e4m3 | 77.96% |
| Test 33 | 2026-04-22 | 关 torch.compile | 76.98% |
| Test 34a | 2026-04-23 | 新实例 replay | 77.51% |

**新 fcloud 4 轮均值约 77.8%,范围 76.98–78.73%。**
13f-1 的 76.91% 处在这个噪声带的最低边 — 与 Test 33 的 76.98% 仅差 0.07pt。

TEST_RESULTS_TRACKING(Test 33 行)已经写下结论:
*"no single-variable knob fixes the ~77–79% local floor — confirms high
intrinsic eval variance."*

所以在动用昂贵手段(重新校准、重新量化、改 attention kernel)之前,
应当先回答一个问题:**13f-1 与 Test 12 之间的 gap 究竟是真实的,还是噪声?**

## 假设

如果 13f-1 的 76.91% 落在该 fcloud 上 Test 12 真实噪声带内,
13f-1 就是一个可用的提交候选 — 可能是 C=0.92(归一化 92–97%)或
甚至 C=1.0(若官方私有集运气好);速度收益(~9% S₁)直接落袋。

只有当 13f-1 均值可复现地比 Test 12 均值低 ≥1.5pt 时,
重新校准 / 重新量化 才有道理。

## 计划

单 fcloud session,4 轮 baseline / 13f-1 交替的 accuracy(无代码改动,仅
restart 间切换环境变量)。总耗时约 3.5 小时。

| Run | Backend variant | Cmdline 标志 | 预计耗时 |
|---|---|---|---|
| **A1** | Test 12 baseline | `--attention-backend minicpm_flashinfer --force-dense-minicpm --dense-as-sparse` | ~50 分 |
| **A2** | 13f-1 | `--attention-backend flashinfer`(无 force-dense,无 DAS) | ~48 分 |
| **A3** | Test 12 baseline(再跑) | 同 A1 | ~50 分 |
| **A4** | 13f-1(再跑) | 同 A2 | ~48 分 |

每轮 restart server。记录 ori_acc + 各 task acc。

### 统计判定

`gap = mean(A2,A4) − mean(A1,A3)`。

| 结果 | gap | 决策 |
|---|---|---|
| **方差占主** | \|gap\| ≤ 1.0pt | **13f-1 在该 fcloud 上与 Test 12 统计等价。**判 13f-1 为 v20 提交候选(暂定)。速度收益锁定(S1/S8/Smax 已有 110.76/40.50/33.66)。打包提交,让私有集决 C 档位。 |
| **临界** | 1.0–1.5pt | 加跑 A5/A6 收紧 CI。若仍临界,直接同时提交两个变体(v20a=Test 12, v20b=13f-1)做官方 A/B。 |
| **真实回归** | > 1.5pt | 13f-1 acc gap 真实存在。进入 **Phase B(逐样本 diff)**:用 A1+A2 已落盘的 `predictions.jsonl`,样本级对比(1-token 差异 vs 整句差异),零 fcloud 成本。只有当 Phase B 提示"kernel 数值噪声"时,才考虑重新校准。 |

### 规则合规

- 无代码改动(仅环境变量切换)。
- 无 eval 脚本改动。
- 准确率门槛:A2/A4 中至少一次 ≥77% 才算候选(否则 C=0 风险过高,无论方差如何)。
- 满足提交约束(≤2GB,≤5h,无 prefix cache)。

## 测试命令(用户批准后)

```bash
# (用户启动 fcloud)

# A1 — Test 12 baseline
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server   # 无 env -> Test 12 默认
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server | head -1"
# 验证: --attention-backend minicpm_flashinfer --force-dense-minicpm --dense-as-sparse
python3 scripts/fcloud/fcloud_workflow.py accuracy
# 记录 A1 ori_acc + predictions.jsonl 路径

# A2 — 13f-1
python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_BACKEND_VARIANT=flashinfer
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server | head -1"
# 验证: --attention-backend flashinfer (无 --force-dense-minicpm, 无 --dense-as-sparse)
python3 scripts/fcloud/fcloud_workflow.py accuracy

# A3 — Test 12 重跑
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy

# A4 — 13f-1 重跑
python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_BACKEND_VARIANT=flashinfer
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy

python3 scripts/fcloud/fcloud_workflow.py shutdown
```

## 交付

4 轮跑完后,agent 将:
1. 把 R13f4-A1 / A2 / A3 / A4 写入 `TEST_RESULTS_TRACKING.md`。
2. 计算每个变体的 mean ± half-range。
3. 套用上面的决策矩阵,推荐下一步(提交 / 加跑 / 进 Phase B)。

## 为什么先做方差而不是直接重新量化

- **重新校准代价**:5+ 小时 fcloud 时间(校准 + 量化 + 启动测试 + accuracy)。
  当前官方校准脚本走 sparse-mode forward;切到 flashinfer 还要改
  `gptqmodel_minicpm_sala.py`。
- **方差跑代价**:~3.5 小时,无代码改动,任何方向都给出明确结论。
- **信息价值**:重新校准只在"校准看到的激活 ≠ runtime 看到的激活"成立时才有用。
  Phase A 先告诉我们到底有没有 bug。
- **跳过 A 的失败模式**:花 5+ 小时重新校准,得到 acc=77.5%(还是落在
  普通 13f-1 的噪声带内),什么都没结论。

## 回滚

无代码改动,无需回滚。

## 交叉引用

- 13f-1 结果:`R13f1-flashinfer` 行。
- 13f-2 结果:`R13f2-flashinfer-keepforce` 行。
- 13f-3 结果:`R13f3-flashinfer-keepall` 行。
- 新 fcloud baseline 噪声:Test 29/30/33/34a 行。
- 已被否定的 13f 推理路线:[RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md)、[PROPOSAL_round13f2_flashinfer_keep_force_dense.zh.md](PROPOSAL_round13f2_flashinfer_keep_force_dense.zh.md)。
