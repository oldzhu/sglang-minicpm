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

## 状态（对应 v20 提交窗口）

- v20 官方分数：`acc=100.0`（`acc_ori=80.87`）、`C=1.0`，但
  `final_score=32.84` — 准确率已不是瓶颈，速度才是。
- 决策：保留 v2 代码已提交，是否在最终提交里**启用** v2 现在变成可选。
  若未来某次提交跌出 `C=1.0`，v2 即第一杠杆。
- 后续工作向速度方向倾斜（S1/S8/Smax），参考
  `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`。

## 后续建议（转向后）

1. 在 fcloud 上以 v2 OFF 重新基线 S1/S8/Smax（先做一轮快速 acc + S1 验证
   作为转向锚点）— 本文档落地后立即跑这一轮。
2. 继续按 GPTQ + FP8 dense 优化目录推进，重点：prefill 吞吐、稀疏注意力、
   KV cache 效率（这些主导官方隐藏长上下文 speed set）。
3. v2 作为安全网保留 — 若未来某轮迭代准确率回退，可在不重量化的前提下
   快速开启。
