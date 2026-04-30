# CHANGE_0140（续 001）— mcq 关闭思考 v2

本文档延续 `CHANGE_0140_mcq_thinking_disable.zh.md`，原版描述的是 v1 方案。
v1 在 MiniCPM-SALA-90 上完全失效，因为该模型的 `chat_template.jinja` 根本
不读 `enable_thinking`。v2 全面更换策略。

## 背景与动机

- v18 / v20 提交结果显示，mcq 是唯一思考反而有害的任务：长思考会用尽
  `max_tokens` 预算，最终字母答案被截断 → 0 分。
- v1（CHANGE_0140 原版）通过预置 Jinja 段落把 mcq 的 `enable_thinking`
  设为 `false`。`grep "enable_thinking" chat_template.jinja` 验证：上游
  模板**没有**任何根据该变量分支的代码，所以 v1 等于无操作。fcloud 跑
  A1（v1 开启）得到 `mcq=56.67%`、`avg_out_len=10438`，思考确实仍在执行。
- 模型在训练时强制每个 assistant 回合以 `<think>...` 开头。无法用变量
  关掉这一行为；必须改写实际进入模型的 token 序列，让它跳过推理段。

## 合规说明

- **仅修改提交侧**。补丁后的 `chat_template.jinja` 位于打包进提交 tar
  的模型目录里，本地评估脚本 `eval_model_001.py` 不动 — 官方评测调用
  HF `apply_chat_template()` 时会执行同一份补丁后的模板。
- 不偷启 prefix cache，不动评估脚本，不存在准确率舞弊。
- 触发依据是字面子串 `"LETTER is one of ABCD"`，公开集中 30/30 mcq 命中、
  qa/niah/cwe/fwe 0 误触 — 部署前已本地核验。

## 策略 v2 — 预置闭合 `<think>` 块（Qwen3 通用技巧）

模型训练时 assistant 回合的样式固定为：

```
<|im_start|>assistant
<think>
  ...推理内容...
</think>

<最终回答>
<|im_end|>
```

由此得到三个结构性事实：

1. `<|im_start|>assistant\n` 后下一 token 必为 `<think>`。
2. `</think>\n\n` 之后是**最终答案**，简洁、不再有推理。
3. 训练分布中**不存在**在 `</think>` 之后又重新打开 `<think>` 的 turn。

v2 利用第 (2) 条：仅在 mcq prompt 中，在写完 assistant 头之后**预置一个
空的闭合 `<think>\n\n</think>\n\n`**。模型于是处于训练分布里的「思考已结
束」状态，下一 token 直接是答案。我们没有禁用思考，而是让模型以为它已
经思考完毕。

这是 Qwen3 chat-template 文档化的「关闭思考」标准技巧，对所有 Qwen3
衍生且不带 `enable_thinking` 守卫的模板都通用，MiniCPM-SALA-90 正是这
种情况。

## v2 在哪里执行

v2 是**输入 prompt 改造**补丁，运行在请求**入口路径**：

```
client → POST /v1/chat/completions → sglang tokenizer_manager
       → HF tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            └── Jinja 模板在此执行；v2 补丁块触发
       → 返回 prompt token ids → scheduler → 前向 → 采样
       → 输出流回客户端（不被改动）
```

补丁不在 sglang Python 里、不在评估脚本里、也不在输出路径上，只改变模型
看到的输入前缀。

## 触发判别 — 为什么 "LETTER is one of ABCD" 可靠

| 任务 | 公开集中含此子串的样本 | 公开总数 | 备注 |
|------|------------------------|---------|------|
| mcq  | 30 | 30 | 触发短语逐字出现在每条 mcq prompt |
| qa   | 0  | —  | 无误触 |
| niah | 0  | —  | 无误触 |
| cwe  | 0  | —  | 无误触 |
| fwe  | 0  | —  | 无误触 |

类似 "Pick A or B" 这类近似措辞**不**包含完整短语，也不会触发 v2。

## 实现

### 涉及文件（commit `0c7767aa8`）

- `benchmark/soar/demo_sala/preprocess_model.py`
  - 标记常量升至 `{# SOAR_MCQ_THINKING_DISABLE_v2 #}`。
  - 保留 `CHAT_TEMPLATE_MCQ_PATCH_MARKER_V1`，用于在 `--mode off` 时清除
    旧版本 v1 残留。
  - 定义 `CHAT_TEMPLATE_MCQ_PATCH_OLD_TRAILING`（被替换的原始尾部块）和
    `CHAT_TEMPLATE_MCQ_PATCH_NEW_TRAILING`（v2 替换后的尾部块）。
  - 抽出共用辅助函数 `_apply_mcq_patch_to_template(template)` 与
    `_revert_mcq_patch_from_template(template)`，让运行时 A/B 切换工具与
    打包时的 preprocess 共享同一逻辑。
  - `_patch_chat_template_for_mcq(dst)` 现在调用辅助函数，支持两种 HF
    存储布局（外部 `chat_template.jinja` 文件 与 嵌入式
    `tokenizer_config.json["chat_template"]`）。
  - 由环境变量 `SOAR_DISABLE_MCQ_THINKING` 控制（默认 true）。

- `benchmark/soar/demo_sala/toggle_mcq_thinking_patch.py`
  - 重写为从 `preprocess_model.py` 直接导入常量与辅助函数，零重复。
  - `--mode status` 现在区分 ON(v2) / 仅遗留 v1 / OFF。
  - `--mode off` 一次性清除 v2 与遗留 v1。

### 替换目标（未补丁的尾部块）

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

### v2 替换后

```jinja
{# SOAR_MCQ_THINKING_DISABLE_v2 #}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- set _soar_mcq_ns = namespace(disable_think=false) -%}
    {%- if messages is defined and messages -%}
        {%- for _m in messages -%}
            {%- if _m['content'] is defined and _m['content'] is string and 'LETTER is one of ABCD' in _m['content'] -%}
                {%- set _soar_mcq_ns.disable_think = true -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- if _soar_mcq_ns.disable_think -%}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif -%}
{%- endif %}
```

Jinja 提示：`{% set %}` 在 `{% if %}/{% for %}` 内部是块作用域；
`namespace(...)` 的属性变更可见跨域，是标准变通做法。

## 本地校验（任何 fcloud 跑测之前）

针对包含确切 `OLD_TRAILING_BLOCK` 的桩模板做端到端 Jinja2 渲染：

| 测试 | 输入 | 期望尾部 | 结果 |
|------|------|----------|------|
| mcq 渲染 | 含 `LETTER is one of ABCD` 的消息 | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | PASS |
| qa 渲染  | "What is the capital of France?" | `<\|im_start\|>assistant\n` | PASS |
| 误触保护 | "Pick A or B from the options." | `<\|im_start\|>assistant\n`（无 think） | PASS |
| apply 幂等 | 已 v2 补丁的模板 | 辅助函数返回 None | PASS |
| revert | 补丁→原始 | 字节相等 | PASS |
| revert 遗留 v1 | 含 v1 preamble 的模板 | preamble 被剥离 | PASS |

## fcloud A/B 流程

对已量化好的模型目录使用 `toggle_mcq_thinking_patch.py`：

```bash
# 启用 v2
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode on

# 还原（同时清除遗留 v1）
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode off

# 查询状态
python3 toggle_mcq_thinking_patch.py --model-dir <model> --mode status
```

每次切换后重启 sglang server（chat template 仅启动时加载一次）。

## 回滚方法

- **已部署的提交 tar**：在 `prepare_env.sh` 设
  `SOAR_DISABLE_MCQ_THINKING=false`，下次 `prepare_model.sh` 即跳过补丁。
- **已量化的模型目录**：用切换工具 `--mode off`，幂等清除 v2 与 v1 标记。
- **源码层**：回退 commit `0c7767aa8`。

## fcloud 回归测试（v2 启用，2026-04-30）

本地 Jinja2 渲染测试通过后，我们在实际 fcloud 模型上启用 v2
（`MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8`），重启 sglang，跑了
并发=32 的全量公开集准确率评测。重启前对实时模型 tokenizer 渲染验证
mcq prompt 尾部确实是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。

| 任务 | v1（空操作，A1 结果） | **v2 启用（本轮）** | v20 baseline（clean） |
|------|---------------------:|-------------------:|---------------------:|
| mcq  | 56.67%（avg_out 10438）| **46.67%（avg_out 12094）** | （较高；未单列） |
| qa   | ~50%                  | 53.33%             | — |
| niah | 100%                  | 96.67%             | — |
| cwe  | 82.67%                | 82.00%             | — |
| fwe  | 100%                  | 98.89%             | — |
| **平均** | **77.87%**         | **75.51%**         | （acc_ori 80.87、归一化 100） |
| 总耗时 | （未捕获）            | 2880.74 s          | — |

**v2 是回退，不是修复**：

- mcq 准确率从 v1 的 56.67% 跌到 46.67%。
- mcq `avg_out_len` 从 10438 升到 12094 — 即便预置了闭合的 `<think>`，模型
  反而**输出了更多**思考 token。
- 渲染验证证明输入前缀正确，问题出在模型响应本身，而非模板拼接逻辑。

**根因假设（行为佐证）**：MiniCPM-SALA-90 并未把
`<|im_start|>assistant\n<think>\n\n</think>\n\n` 视作 sink state。Qwen3
「闭合空 think」技巧依赖模型在 SFT/蒸馏阶段见过空 `<think>...</think>`
作为「思考已结束」的合法 turn。MiniCPM-SALA-90 反而在我们闭合的
`</think>\n\n` 之后**重新展开推理**（再发一次 `<think>` 或直接写自由
推理文本），结果是 token 更多而不是更少。Qwen3 标准技巧在该模型上不通用。

**v1 看起来「更好」的原因**：v1 完全空操作，思考照旧。v2 主动改了前缀，
落在模型未训练的分布上，反而轻微扰动了 mcq 表现，看起来比 v20 clean 更差。

## 处置

- 本轮跑测后立刻在 fcloud 模型上还原 v2
  （`toggle_mcq_thinking_patch.py --mode off`），chat_template 字节级
  对齐 v20 提交。
- fcloud 已暂停（`pause-instance` 一次 504 后重试成功）。
- 代码保留在仓库中作为**已记录的死路**。toggle 工具 + preprocess 接线
  + 测试仍然是有用的基础设施，未来若再做思考控制可以复用。
- **不会**提交 v2。v20 已在官方拿到 `C=1.0`（归一化 100、`acc_ori`
  80.87），准确率不是瓶颈。

## 状态（对应 v20 提交窗口）

- v20 官方分数：`acc=100.0`（`acc_ori=80.87`）、`C=1.0`，但
  `final_score=32.84` — 准确率已不是瓶颈，速度才是。
- v2 已测试并被否决（本文档）；CHANGE_0140 在此封口。
- 后续工作向速度方向倾斜（S1/S8/Smax），参考
  `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`。

## 后续建议（转向后）

1. v2 已封口；未来若没先解决「模型在 `</think>` 之后重新展开思考」这个
   行为问题，不要再回头测试或启用 v2。
2. 直接进入速度优化：打开
   `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`，按优先
   级挑选聚焦 prefill 吞吐、稀疏注意力、KV cache 效率的方向（这些主导
   官方隐藏长上下文 speed set）。
3. 若未来某轮跌出 `C=1.0`、确实需要 mcq 准确率杠杆，下一个值得尝试的是
   **服务端强制采样 `</think>` 早停**（不同机制，不依赖模型训练分布），
   而不是再做前缀拼接。
