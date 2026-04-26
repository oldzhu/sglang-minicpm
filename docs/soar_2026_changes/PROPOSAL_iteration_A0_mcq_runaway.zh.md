# 提案：Iteration A（新优先级）— 修复 mcq/qa 输出失控

**日期**：2026-04-23
**触发事件**：v18 官方结果 → `acc_ori=76.64%, C=0（淘汰）, S1=586s, S8=1089s, Smax=2864s`
**取代优先级**：[PROPOSAL_iteration_A_revised_runtime_tuning.zh.md](PROPOSAL_iteration_A_revised_runtime_tuning.zh.md)（保留为次优先级，后续再做）
**状态**：提案 — 等待用户批准

---

## 1. 从代码审查得到的直接证据

### 评测脚本（`benchmark/soar/demo_sala/eval_model_001.py`）

| 行号 | 代码 | 含义 |
|---|---|---|
| 352 | `outputs = model.generate(inputs, max_out_len=65536)` | **全部 5 个任务统一用 max_tokens=65536**，包含 mcq（应 ≤ 50） |
| 353 | `chat_template_kwargs={"enable_thinking": True}` | **所有任务都开启思考模式**，包括 mcq |
| 79–92 | `_get_potential_stop_words` | 停止词只包含 tokenizer EOS；没有 `</think>` 终止符，也没有按任务区分的停止词 |
| 178 | `extract_final_answer(pred)` 按 `</think>` 切分 | 如果 `</think>` 从未被吐出（思考超过 `max_tokens` 被截断），抽取器直接返回整个思考片段 → mcq 选项抽取失败 |

### 实测行为（Test 34a，150 条样本，concurrency=32）

| 任务 | 准确率 | avg_in tokens | avg_out tokens | 期望 avg_out |
|---|---|---|---|---|
| **mcq** | **56.67%** | 270 | **10,946** ❌ | 1–50 |
| qa | 46.67% | 71,569 | 104 | ✓ |
| niah | 100% | 73,983 | 360 | ✓ |
| cwe | 85.33% | 74,163 | 11,350 | 数 K |
| fwe | 98.89% | 68,154 | 13,806 | 数 K |

每条 mcq 的输出 token 超出期望的 200–1000 倍。在 concurrency=32 无 max_tokens 上限的情况下，这一件事就足以解释：
- **官方 Smax=2864s**：超长 mcq 链堵塞调度队列，阻塞新 prefill
- **官方 S8=1089s**：concurrency=8 下相同效应
- **官方 acc=76.64% → C=0**：mcq 链在 `max_tokens` 被截断，没来得及吐出最终答案字母，打分器返回 0

## 2. 相对之前路线图的战略转向

### v18 结果修正的认知

| 先前认知 | v18 实际表现 |
|---|---|
| 本地 S1/S8/Smax ≈ 110/40/33s 能代表官方 | **错误**：官方是 586/1089/2864s，慢 5–86× |
| 精度下限 77–79% 属于噪声，可以接受 | **错误**：76.64% 官方落在 C=0 的刀口上 |
| 运行时调优（torch.compile、cuda-graph）是速度杠杆 | **错误**：kernel 每步速度不是瓶颈，失控生成才是 |
| Test 29–33 中 mcq 40–96% 的抖动来自评测噪声 | **部分错误**：LOW 端其实是结构性的（思考溢出），HIGH 端是侥幸早早吐出 `</think>` |

### 对优先级顺序的影响

此前所有速度旋钮（torch.compile、Marlin tile、调度）都只影响**每步时延**，都不能减少**总步数**。当 mcq 吐出 10,946 步、而应当吐 10–50 步时，每步快 10% 只能省 10%；但把 mcq 截到 256 步就直接省 97.6%。这个乘法差距压过当前桌面上所有其他优化项。

**推论**：第一个应该问的问题已经不是"如何进前五"，而是"如何让 C ≠ 0"。C ≥ 0.92 之前，速度工作毫无意义。

## 3. 建议修复（两层）

### Layer 1 — 评测脚本侧（只影响本地基准，不影响官方评测）

仅改动 `benchmark/soar/demo_sala/eval_model_001.py`。作用是做**快速本地代理**，验证 server/模型修复是否起作用。不会进入官方提交。

**Fix 1.1 — 按任务配置 `max_out_len`**

| 任务 | 建议 max_out_len | 理由 |
|---|---|---|
| mcq | 1024 | 够做简短推理 + 吐 `ANSWER: X`；即便官方上限不同，也能限制测试成本 |
| qa | 512 | 只需简短事实答案 |
| niah | 1024 | 检索字符串；当前 avg=360 |
| cwe | 16384 | 保留当前长度 |
| fwe | 16384 | 保留当前长度 |
| （默认） | 16384 | 防止未分类任务失控 |

实现方式：建立任务 → `max_out_len` 映射，按任务分组调用 `generate()`（或在 `sampling_kwargs` 中传入 per-request `max_tokens`）。

**Fix 1.2 — 增加思考终止的停止序列**

在 stop list 中加入 `</think>` 和 `<|endoftext|>`。如果模型没有吐出 `</think>` 但吐出了 `<|endoftext|>`（常见替代终止符），仍可中断。风险低。

**本地预期效果**：mcq avg_out 从 11K 降到 ≤1K；整体评测时长从 2817s 降到 ≈800–1200s；如果抽取器能在截断前看到 `ANSWER: X`，精度也会上升。

### Layer 2 — 模型/服务端修复（会影响**官方**评测）

这一层才是真正影响提交分数的修复。有两条候选路径，按顺序验证：

**Fix 2.1 — 在模型 chat template 中关闭默认 `enable_thinking`**（可靠性 HIGH，风险 LOW）

MiniCPM-SALA 的 chat template 内部很可能用 Jinja 变量控制 `enable_thinking`。我们可以在 `preprocess_model.py` 中修改 `tokenizer_config.json` 的 `chat_template` 字段，使其：

- 对 mcq（短 prompt 无长上下文）跳过思考
- 或者始终跳过思考（对长上下文 cwe/fwe 精度可能略降，官方数据待确认）

具体做法需要先检查 fcloud 上模型的 chat_template 字符串。

**Fix 2.2 — 服务端 `--reasoning-parser qwen3`** ❌ **已验证无效（2026-04-26）**

通过审查 `benchmark/soar/demo_sala/eval_model_001.py` 确认：

1. 评测脚本只读取 `choices[0].message.content`（第 40 行），完全不读 `reasoning_content`。
2. 评测脚本自己在客户端调用 `extract_final_answer`（第 176-178 行），用 `pred.split('</think>')[-1]` 取最后一段：
   ```python
   def extract_final_answer(pred):
       parts = pred.split('</think>')
       return parts[-1].strip() if len(parts) > 1 else pred
   ```
3. 所有 scorer（`score_mcq` / `score_exact_match` 等）都已经做了思考剥离。

**结论**：开启 `--reasoning-parser qwen3` 在两种情形下的影响：

| 情形 | 不开（当前） | 开启 reasoning-parser |
|---|---|---|
| 正常输出（含 `</think>`） | content 含完整文本 → split → 取答案 ✅ | content 已被 parser 剥离 → 直接取答案 ✅（等价） |
| **mcq runaway**（思考阶段被 max_tokens 截断，无 `</think>`） | content 是思考残段，正则有时**仍能**抓到中途出现的 `ANSWER: X` → 偶尔得分 | content 变为**空字符串**（全部塞进 reasoning_content），正则一定抓不到 → **必丢分** |

由于官方评测使用同一份 harness（按 copilot-instructions 规则禁止修改 eval 脚本即为此对齐信号），上述结论同样适用于线上。

**因此 Fix 2.2 对准确率净期望为 0 或负，从行动列表中删除，不再尝试。**

**Fix 2.3 — 服务端 max-tokens 截断带按请求采样覆盖**（风险 HIGH）

SGLang 没有 per-task max_tokens 覆盖机制；任何 server-side 截断都会作用于全部任务。**不建议**，会伤害 cwe/fwe。

**Fix 2.4 — 在模型层添加 `</think>` 作为 stop token**（可靠性 MEDIUM，风险 MEDIUM）

修改 preprocessed 模型的 `generation_config.json`，把 `</think>` 加入 `eos_token_id`。如果模型在多数情况下能正确吐出 `</think>`，这一手可以强制停下思考，让评测器拿到 `</think>` 之后的部分。如果模型不吐 `</think>`，则无效。

**Fix 2.5 — 通过服务端 `--max-tokens` 默认值限制**（风险 LOW，收益 LOW）

不确定 SGLang 是否暴露 server-side 默认 max_tokens。其他方案失败后再考虑。

## 4. 执行顺序

### Phase 1 — 本地取证（不用 fcloud）
1. **检查** Test 34a `predictions.jsonl`（请求用户分享 3 条样本节选），或请 fcloud 提供 5 条 mcq 预测，确认失控模式。
2. **检查** fcloud 上模型的 chat_template，判断思考是否总是开启或条件开启。

### Phase 2 — 上线 Layer 1（评测侧）— 验证假设
3. 修改 `eval_model_001.py` 加入 Fix 1.1（按任务 max_out_len）+ Fix 1.2（多停止词）。
4. commit + sync 到 fcloud + 跑精度评测。
5. **成功标准**：mcq avg_out ≤ 2000，整体精度 ≥ 78%，评测时长 ≤ 1500s。

### Phase 3 — 上线 Layer 2（模型/服务端）— 修复官方评测
6. 根据 chat_template 检查结果，选 Fix 2.1 或 Fix 2.2（或 2.4）。
7. 修改 `preprocess_model.py` 或 `prepare_env.sh`，重新量化模型，跑精度 + 速度。
8. **成功标准**：本地 mcq avg_out ≤ 2000（与 Layer 1 结果一致）— 证明修复在模型/服务层面生效，不只是在评测脚本层面。

### Phase 4 — 重新提交 v19
9. 打包新提交。
10. 期望官方结果：`acc_ori ≥ 78%` → `C ≥ 0.96`，`Smax ≤ 1500s`（大约砍半）。

## 5. 风险表

| 风险 | 缓解 |
|---|---|
| 关掉思考损害 cwe/fwe/qa 精度 | 条件关（只在短 prompt 时关），或保留思考但加硬性 max_tokens 上限 |
| 改过的 chat_template 破坏官方 eval loader | 在 `preprocess_model.py` 保留原始 chat_template 作为备选；用环境变量 feature-flag |
| ~~`--reasoning-parser qwen3` 与 MiniCPM token 格式不匹配~~ | ~~本地先比对 token 字符串与 `reasoning_parser.py` 再启用~~ — Fix 2.2 已撤销（见上） |
| 评测侧 Fix（Layer 1）让本地好看但官方仍失败 | Layer 2 明确承担"可迁移"职责；Layer 2 本地复现通过前不重新提交 |
| 官方 eval 脚本与我们的完全不同 | 概率低——local 与 official 精度差距只有 1pt 左右，说明协议相似 |

## 6. 回滚

每层都按 commit 隔离，均可快速还原：
```bash
# Layer 1 还原
git checkout benchmark/soar/demo_sala/eval_model_001.py
# Layer 2 还原（配置改动）
git checkout benchmark/soar/demo_sala/preprocess_model.py benchmark/soar/demo_sala/prepare_env.sh
```
任何一层都不需要 rebuild kernel 或 wheel。

## 7. 对战略路线图的影响

本迭代落地后需要刷新 [STRATEGIC_ROADMAP_TOP5.zh.md](STRATEGIC_ROADMAP_TOP5.zh.md)：
- Iteration A（运行时调优）降级为"C 转正之后再做的免费收尾"
- 新 Iteration A-0（本提案）= mcq/qa 失控修复，**是其余一切工作的前置条件**
- 投机解码（Iteration C）只有在生成长度可控后才有意义
- Phase 4 准备阶段包含：用 `perf_public_set.jsonl` 和不同 concurrency cap 构造 `speed_full_{s1,s8,smax}.jsonl`，让本地速度基准与官方一致

## 8. 待用户确认的问题

1. **是否批准 Phase 1 取证**（看 Test 34a 几条预测 + chat_template，主要靠本地，fcloud 只需要只读查询）？
2. 倾向于先单独上 Layer 1（隔离验证），再上 Layer 2；还是 Layer 1 + Layer 2 一起上一次测试？
3. 是否已经知道 MiniCPM-SALA chat_template 的格式，可以直接决定 Fix 2.1 vs 2.4？
4. 先由用户分享一条 mcq 预测样本再写代码；还是我现在就预先动手实现 Fix 1.1（按任务 max_out_len）？
