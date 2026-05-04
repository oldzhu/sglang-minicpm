# 对话 — Tier 1 长上下文复测 (2026-05-04 ~08:52)

## 主题

速度方向启动：在新的长上下文数据集下重新排序优化目录前 3 名速度候选，
然后通过一行环境变量开关执行 #1（Tier 1 目录配置复测），便于安全回滚。

## 第 1 轮 — Console 配置刷新工具（fcloud 测试前置）

### 用户请求

> "(a) and by the way, before go (a) have you documented the steps to re-Set
> up console config when cookie expired after 1 week, if u haven't, pls create
> a new document for it. it is possible to automize the steps to resetup the
> console config — I mean f12->network->capture post/put request and get
> authorization and cookie and user name and job id automatically"

### Agent 操作

1. 创建双语 JWT 周度刷新手册：
   - [docs/soar_2026_changes/HOWTO_refresh_fcloud_console_config.en.md](../HOWTO_refresh_fcloud_console_config.en.md)
   - [docs/soar_2026_changes/HOWTO_refresh_fcloud_console_config.zh.md](../HOWTO_refresh_fcloud_console_config.zh.md)
   - 章节：快速检查、手动刷新、HAR 半自动刷新、为什么完全自动化不可行
     （3 个架构层面阻塞点）、TL;DR、故障排查。
2. 实现 [scripts/fcloud/refresh_console_config_from_har.py](../../scripts/fcloud/refresh_console_config_from_har.py)
   （约 190 行）— 解析 DevTools 导出的 HAR，提取 authorization / cookie /
   JOB_ID，解码 JWT 获取用户名，备份旧的 `~/.fcloud_console_config`，以
   `0600` 权限写入新配置。
3. 用合成 HAR 做端到端测试（合规 JWT、JSESSIONID cookie、start 端点 URL）；
   刷新摘要正确打印，`.bak` 保留。
4. 更新 `.github/copilot-instructions.md` 与
   `scripts/fcloud/fcloud_console_config.example`，指向新的快捷流程。
5. 提交 `a6b16370e` 已推送至 `minicpm-src/mixed_minicpm_cudagraph`。

### 结果

- 后续每周刷新只需约 30 秒人工时间（F12 → 点击 Start/Pause →
  "Save all as HAR with content" → `python3 refresh_console_config_from_har.py
  console.har`）加上自动解析步骤。
- 已诚实记录的限制：完全无人值守的 F12 抓取不可行 —
  (a) JWT 仅存在于页面 `Authorization` 请求头中（不在 `document.cookie`），
  (b) 任何非 console 来源的跨域 XHR 都无法读到该 header，
  (c) SSO + 2FA 登录本身不能负责任地自动化。

## 第 2 轮 — fcloud Tier 1 复测

### 用户请求

> "go"

（在确认了拟定的 9 步 fcloud 流程，以及环境变量传递方式 `(I)` 之后 —
不需要工具改动；`fcloud_workflow.py` 的 `restart-server --env KEY=VAL` 已存在。）

### Agent 操作 / 时序

| # | 步骤 | 结果 |
|---|------|------|
| 1 | `console-token-info` | JWT 有效，剩余 ~2.91 天 |
| 2 | `start-instance` | HTTP 200 — 任务启动 |
| 3 | 轮询 fcloud_exec 直到就绪 | 第一次尝试即就绪 |
| 4 | `sync` | 拉取 `a6b16370e`；`prepare_env.sh` 已拷贝到 `/root/submission_sim/` |
| 5 | `restart-server`（Run A，env 未设置）| 214s 后 server 就绪 |
| 6 | `speed --variant all` 然后 `speed --variant s1`（找回被滚屏覆盖的 S1）| Run A：S1=110.53s, S8=40.56s, Smax=33.65s |
| 7 | `restart-server --env SOAR_TIER1_LONG_CONTEXT=1` | 214s 后 server 就绪；通过 `/proc/$pid/cmdline` 验证启动参数中确含 `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --schedule-conservativeness 0.8` |
| 8 | `accuracy` | Run B 准确率 = **78.73%**（mcq 56.67、cwe 83.67、fwe 100、niah 100、qa 53.33）|
| 9 | `speed --variant {s1,s8,smax}` | Run B：S1=111.36s, S8=40.49s, Smax=33.62s |
| 10 | `pause-instance` | 第一次 504（上游网关超时），重试 HTTP 200 — 任务已暂停 |
| 11 | 更新 TEST_RESULTS_TRACKING.md | 新增 Tier1-A-baseline + Tier1-B-candidate 两行 |
| 12 | 本对话日志 | 双语写出 |

### 量化结果

| 指标 | Run A（基线）| Run B（TIER1=1）| Δ | 显著? |
|---|---|---|---|---|
| S1 (s) | 110.53 | 111.36 | +0.75% | 否 |
| S8 (s) | 40.56 | 40.49 | −0.17% | 否 |
| Smax (s) | 33.65 | 33.62 | −0.09% | 否 |
| 准确率（原始）| — | 78.73% | — | n/a |
| 准确率（归一化）| — | 98.42% | — | C=0.96 档 |
| mcq | — | 56.67% | （=Test 29）| 噪声内 |
| qa | — | 53.33% | （较 Test 12 −3pt）| 噪声内 |
| niah / fwe | — | 100% / 100% | （=）| — |

### 结论 / 决策

- **Tier 1 参数在本地短上下文速度集上中性。** 这是预期结果：本地
  `speed_*.jsonl` 输入仅约 1K token，远低于 65K chunked-prefill 阈值；
  新参数对这些 prompt 完全不会触发。
- **本地只能验证负向**：短上下文不退化、准确率未崩塌（78.73% 与
  Test 29 完全一致，较 Test 12 下降 0.56pt — 都在 ±1pt 本地噪声底线内）。
- **官方收益本地无法度量**：长上下文官方速度集（68% 输入在 32K–512K 区间）
  正是 chunked-prefill=65K + `prefill-max-req=4` + `sched-cons=0.8` 应当
  发挥作用的场景。
- **决策**：将 `SOAR_TIER1_LONG_CONTEXT=1` 作为 v21 候选发版。该开关在
  `prepare_env.sh` 中默认关闭，因此 `unset SOAR_TIER1_LONG_CONTEXT` 时
  v20 字节级等价复现。打包 v21 时，在 `prepare_env.sh` 顶部 export
  该变量（或一行修改）。

### 后续

- 一旦 #1 上线决策确认，立即：
  - 执行 **#2-A**：在 v21 之上做 `--torch-compile-max-bs` 扫描（先试 bs=16，
    可能 bs=24；用户提到此前 OOM，但本机 fcloud 值得再试）。
  - 在提交后约 24h 关注排行榜更新。
- 本地速度集对 Tier-1 类优化具有误导性。后续针对 prefill 吞吐 /
  chunk size / 调度器的速度提案，应明示 "本地 sweep 无法验证；必须
  发版后实测" — 本轮已踩到该点。
- **本轮无其他无关编辑或仅中文文档变更。**

### 交叉引用

- 提案：[PROPOSAL_tier1_long_context_retest_20260430.zh.md](../PROPOSAL_tier1_long_context_retest_20260430.zh.md)
- 代码提交（env 开关）：`d45b3ff1d` `feat(prepare_env): SOAR_TIER1_LONG_CONTEXT env switch`
- 代码提交（刷新工具）：`a6b16370e` `feat(fcloud): HAR-based console-config refresh + bilingual howto`
- 测试行：TEST_RESULTS_TRACKING.md → `Tier1-A-baseline`、`Tier1-B-candidate`
- 已搁置：[PROPOSAL_nvfp4_kv_dense_smoke_20260430.zh.md](../PROPOSAL_nvfp4_kv_dense_smoke_20260430.zh.md)
  （架构性阻塞，详见 CHANGE_0132 §8 — 由 2-A torch-compile-max-bs sweep 替代）。
