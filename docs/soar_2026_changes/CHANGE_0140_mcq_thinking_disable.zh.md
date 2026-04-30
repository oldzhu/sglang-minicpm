# CHANGE_0140 — 经聊天模板补丁为 mcq 关闭 `enable_thinking`（阶段 14.1）

**状态**：本地实现完成，待 fcloud 验证
**迭代**：Round 14.1（依据 [PROPOSAL_round14_local_accuracy_improvement.zh.md](PROPOSAL_round14_local_accuracy_improvement.zh.md)）
**基线**：v20（commit `205d8cb91`，`SOAR_BACKEND_VARIANT=flashinfer`）
**作用域**：仅 mcq —— qa/niah/cwe/fwe 不受影响
**改动文件**：`benchmark/soar/demo_sala/preprocess_model.py`、`benchmark/soar/demo_sala/prepare_env.sh`

---

## 1. 背景与动机

评测 harness（[eval_model_001.py:352](../../benchmark/soar/demo_sala/eval_model_001.py#L352)）对所有任务都传 `chat_template_kwargs={"enable_thinking": True}`。对 mcq 而言，这会触发 `<think>...</think>` 推理块，但模型经常在 token 预算内**没能输出闭合的 `</think>`**。Harness 的 `extract_final_answer`（[行 176-178](../../benchmark/soar/demo_sala/eval_model_001.py#L176-L178)）按 `</think>` 切分；若标签缺失，正则 `(?i)ANSWER\s*:\s*([A-D])` 会在整个未闭合（多被截断）的思考块上跑，多数情况下找不到干净字母。

后果：mcq 精度双峰漂移——同一二进制 4 次运行（13f-4 quartet）mcq 在 40/53/60/63% 之间漂，纯粹取决于 `</think>` 是否被采到。早期 Test 29 在幸运 run 已观测到 96.67%。

把 mcq 从 ~55% 均值抬到已观测过的 ~90–96% 区间，整体精度 **+7pt**（mcq 占 5 个等权任务之一）。该修复**对官方 S1/S8/Smax 速度基准零影响**（速度集不含 mcq prompt）。

## 2. 规则合规

- **评测脚本完整性**：`eval_model_001.py` 不动。修复全部落在服务端/模型端。
- **提交包**：补丁在 `preprocess_model.py` 内，该脚本随提交包发布、由官方运行器在提交准备阶段执行；因此**官方推理与本地推理见到相同的 patched chat_template**。
- **无违规手段**：未启用 prefix-cache 私货、未改 harness、未触碰并发参数。
- **可回滚**：`SOAR_DISABLE_MCQ_THINKING=0` 环境变量即可关闭补丁做 A/B。

## 3. 检测信号

`perf_public_set.jsonl` 中**每条** mcq prompt 都包含字面子串：

```
LETTER is one of ABCD
```

它来自指令模板：

> "The last line of your response should be of the following format: 'ANSWER: \$LETTER' (without quotes) where LETTER is one of ABCD."

公共集 30 条 mcq 全部含该短语；非 mcq（qa/niah/cwe/fwe）无一含此短语。公共集误报率 0%。

该指令由 harness 注入，不来自数据，因此**私集预期含同一字符串**（官方任务定义未变；评测组宣布过指令模板跨提交不变）。

## 4. 实现

### 4.1 Jinja 前置块

向模型 `tokenizer_config.json` 中现有 `chat_template` **前置**一段 Jinja：

1. 用 namespace 变量 `_soar_ns.et` 初始化为输入 `enable_thinking`（未定义时默认 `true`）。
2. 遍历 `messages`，若任意消息内容包含 `LETTER is one of ABCD`，将 `_soar_ns.et = false`。
3. 顶层执行 `{% set enable_thinking = _soar_ns.et %}`，让模板正文看到覆盖后的值。

`namespace()` 是必须的：Jinja2 中 `{% set %}` 在 `{% if %}` / `{% for %}` 块内是**块作用域**，无法逃逸。但 namespace 属性突变跨作用域可见；最后顶层 `{% set %}` 才能完成全局再绑定。

```jinja
{# SOAR_MCQ_THINKING_DISABLE_v1 #}
{%- set _soar_ns = namespace(et=(enable_thinking if enable_thinking is defined else true)) -%}
{%- if messages is defined and messages -%}
{%- for _m in messages -%}
{%- set _c = _m['content'] if (_m['content'] is defined and _m['content'] is string) else '' -%}
{%- if 'LETTER is one of ABCD' in _c -%}
{%- set _soar_ns.et = false -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- set enable_thinking = _soar_ns.et -%}
```

注释 `{# SOAR_MCQ_THINKING_DISABLE_v1 #}` 用作幂等标记；在已 patch 的目录上重跑 `preprocess_model.py` 不会重复前置。

### 4.2 `preprocess_model.py` 改动

新增 `_patch_chat_template_for_mcq(dst)` 函数（位于 `main()` 上方），逻辑：

- `SOAR_DISABLE_MCQ_THINKING` 不 truthy → 短路；
- 缺少 `tokenizer_config.json` → 短路；
- 已含标记注释 → 短路；
- 否则前置前置块，原子写回（`tmp_path.replace(tok_path)`）。

在 `main()` 中 copy 与 gptq 两条分支末尾各调用一次，确保所有提交流程都得到 patched 模板。

### 4.3 `prepare_env.sh` 改动

在 v20 后端块附近新增：

```bash
export SOAR_DISABLE_MCQ_THINKING="${SOAR_DISABLE_MCQ_THINKING:-1}"
```

默认开启。A/B 关闭：执行 `preprocess_model.py` 前 `export SOAR_DISABLE_MCQ_THINKING=0`。

## 5. 本地自测

合成模板上的 6 个 Jinja 渲染用例全部通过：

| 用例 | messages | enable_thinking | 期望 | 实际 |
|---|---|---|---|---|
| 1 | mcq | True | OFF | OFF ✓ |
| 2 | mcq | False | OFF | OFF ✓ |
| 3 | qa | True | ON | ON ✓ |
| 4 | qa | False | OFF | OFF ✓ |
| 5 | mcq | 未定义 | OFF | OFF ✓ |
| 6 | qa | 未定义 | ON | ON ✓ |

加 `_patch_chat_template_for_mcq` 4 个单元测试：首次 patch、幂等、env 关闭、文件缺失安全。全部绿。

## 6. fcloud 验证计划

**待用户启动 fcloud 并批准后**做 4 轮 A/B：

| Run | `SOAR_DISABLE_MCQ_THINKING` | 期望 mcq | 期望整体 |
|---|---|---|---|
| R14.1-A1 | 1（默认） | 90–96% | 80–84% |
| R14.1-A2 | 0（对照） | 50–60%（匹配 13f-4） | 75–77% |
| R14.1-A3 | 1 | 复现方差 | 复现 |
| R14.1-A4 | 0 | 复现方差 | 复现 |

采纳判据：mean(R14.1-A1, A3) − mean(R14.1-A2, A4) **≥ +3pt 整体** **且** mcq 均值上抬 **≥ +20pt** **且** 其他任务无 > 1pt 回退。否则回滚并复议阶段 14.2。

预期速度不变。`/root/data/speed_*.jsonl` 不含 mcq prompt，做 1 次 S1 抽查确认无回退即可。

## 7. 回滚

```bash
# 运行时关闭（无需重新构建）
export SOAR_DISABLE_MCQ_THINKING=0

# 或 revert commit
git revert <change-0140-commit>
```

即便补丁在仓库内，未 patch 的模型目录（无标记）会在下次 `preprocess_model.py` 运行时重新被 patch。需净态请从输入目录重新拷贝 `tokenizer_config.json`。

## 8. 结果

待 fcloud 验证后填入。

| Run | 日期 | 整体 | mcq | qa | niah | cwe | fwe | 备注 |
|---|---|---|---|---|---|---|---|---|
| （待定） | | | | | | | | |

## 9. 交叉引用

- 提案：[PROPOSAL_round14_local_accuracy_improvement.zh.md](PROPOSAL_round14_local_accuracy_improvement.zh.md)
- 早期跑飞分析：[PROPOSAL_iteration_A0_mcq_runaway.zh.md](PROPOSAL_iteration_A0_mcq_runaway.zh.md)
- 方差量化：[PROPOSAL_round13f4_variance_quantification.zh.md](PROPOSAL_round13f4_variance_quantification.zh.md)
- v20 基线：prepare_env.sh commit `205d8cb91`
- 测试结果跟踪：[TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md)
