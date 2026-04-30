# 提案 — Tier 1 调度配置在新长上下文速度集上的复测

**日期**：2026-04-30
**状态**：提案 — 等待用户批准
**前置文档**：`OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` Tier 1 "best so far" 行
**姊妹迭代**：`PROPOSAL_nvfp4_kv_dense_smoke_20260430.{en,zh}.md`（在本次之后执行）

## 1. 背景

目录中记录了一组在**旧速度集**上击败 v18 baseline 的 Tier 1 调度配置：

| 配置 | S1 | S8 | Smax |
|---|---|---|---|
| v20 已上线（`pmr=1, sc=1.0, chunk=32K`） | 121.71 | 44.09 | 35.86 |
| Tier 1 最佳（`pmr=4, sc=0.8, chunk=65K`） | 110.54 | 40.54 | 33.59 |

旧数据下 **S1 −8.2 %**。但 2026-04-15 官方更新速度集为**长上下文为主（68 % 输入 32K–512K）**之后，v20 上线版本被回退回 `pmr=1, sc=1.0, chunk=32K`。在新长上下文速度集上从未有任何官方记录测量过 Tier 1 最佳配置。

在 prefill 占主导的长上下文上，Tier 1 杠杆其实**更划算**：

- `chunked-prefill-size 65536` 在输入 ≥ 64K 时让每次 step 的 prefill 吞吐翻倍（每个请求少一个 chunk）。
- `prefill-max-requests 4` 同时允许 4 个 prefill 在飞，对 S8 / Smax 长输入排队时尤为有利。
- `schedule-conservativeness 0.8` 让请求更早被准入，提升各档利用率。

## 2. 目标

测量 Tier 1 最佳配置在新长上下文速度集上能否重夺旧数据集上的领先；若能则作为 v21 上线。

**预期收益（估计）**：S1 / S8 / Smax 全档 −5 % 至 −15 %。Smax 摆动应比旧数据更大。

## 3. 规则合规

- 不改模型、不改量化、不改 kernel、不改提交包格式。
- 纯调度调参，官方启动器已经通过 `prepare_env.sh` 的 `SGLANG_SERVER_ARGS` 接受。
- 在官方 5 h 预算内。无新依赖。

## 4. 风险

| 风险 | 缓解 |
|---|---|
| `pmr=4 / chunk=65K` 长上下文 Smax 内存溢出（同时驻留更多 KV） | `mem-fraction-static 0.84` 不变；监控 server log；溢出则降级到 `pmr=2 / chunk=65K` |
| 更激进调度导致精度回归 | 这些 flag 不影响精度路径；预期与 v20 同精度（仍跑完整 acc 作为护栏） |
| 本地↔官方比例反转 | 仅汇报本地数字，但承认官方长上下文集更偏 prefill → 收益应 ≥ 本地 |

## 5. 修改文件

| 文件 | 修改 |
|---|---|
| `benchmark/soar/demo_sala/prepare_env.sh` | 新增 env flag `SOAR_TIER1_LONG_CONTEXT`（默认 OFF）。`=1` 时仅覆盖 gptq 分支的 `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --schedule-conservativeness 0.8`。 |
| （无其他文件改动） | |

env 开关保留 v20 为默认，全新 checkout 仍可复现已上线 baseline。

## 6. 验证命令

fcloud 上，在 `python3 scripts/fcloud/fcloud_workflow.py sync` 之后：

```bash
# A) baseline 保险（应与 Test 12 数字匹配）
ssh fcloud "cd /root/submission_sim && unset SOAR_TIER1_LONG_CONTEXT && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# B) Tier 1 候选
ssh fcloud "cd /root/submission_sim && export SOAR_TIER1_LONG_CONTEXT=1 && source prepare_env.sh && grep SGLANG_SERVER_ARGS"
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy   # 精度护栏
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

总 fcloud 时长估计 ~45 min（2 × acc ~10 min + 2 × 速度 ~15 min）。

## 7. 成功 / 失败判据

| 结果 | S1 | S8 | Smax | Acc | 决策 |
|---|---|---|---|---|---|
| **WIN** | ≤ 0.97 × baseline | ≤ 0.97 × baseline | ≤ 1.00 × baseline | ≥ 79 % | 作为 v21 上线，更新目录行。 |
| **PARTIAL** | ≤ 0.97 × baseline | 持平 / 回归 < 2 % | 持平 / 回归 < 2 % | ≥ 79 % | 暂缓；等 #2 NVFP4 结果后再评估。 |
| **REGRESSION** | 任一档回归 ≥ 2 % | — | — | — | 记录并放弃；目录标注 "长上下文已验证回归"。 |
| **OOM / 崩溃** | — | — | — | — | 降级 `pmr=2 / chunk=65K` 试一次；仍坏则放弃。 |

（baseline = 同一天同一 fcloud 实例 `SOAR_TIER1_LONG_CONTEXT=0` 的步骤 A 运行结果，**不是**历史 Test 12 数字 — fcloud 磁盘 I/O 抖动客观存在。）

## 8. 回滚

`unset SOAR_TIER1_LONG_CONTEXT` 并重新 source `prepare_env.sh`。源码无状态变化。git revert 仅一处 `prepare_env.sh` 改动。

## 9. 后续步骤

- WIN → 上线 v21 → 在新 baseline 上跑 #2 NVFP4 KV 提案。
- REGRESSION → 保持 v20 → 在 v20 baseline 上跑 #2 NVFP4 KV 提案。

无论结果如何，下一项都是 #2。
