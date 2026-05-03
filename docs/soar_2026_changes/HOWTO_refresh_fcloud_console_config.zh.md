# 操作指南 — 刷新 `~/.fcloud_console_config`（每周更新 JWT）

**状态**：长期参考文档 — 控制台界面变化时随时更新。
**频率**：约每周一次（JWT TTL ≈ 7 天）。
**何时需要**：`start-instance` / `pause-instance` 返回 HTTP 401 / 403，或 `console-token-info` 报 `expired: true`。

## 1. 先做这一步

```bash
python3 scripts/fcloud/fcloud_workflow.py console-token-info
```

如果显示 `expired: false` 且剩余 > 1 天，无需操作。

## 2. 手工刷新（标准流程，~2 分钟）

1. Chrome/Edge 打开 https://console.cnomnibot.com/omnibot ，登录。
2. 按 **F12** → 切到 **Network** 面板，确认录制开启（左上红点亮）。
3. 勾选 **Preserve log**（避免请求被刷掉）。
4. （可选）filter 输入 `start|pause` 缩小范围。
5. 在页面上点任意任务的 **Start** 或 **Pause** 按钮。实例当前是不是真的需要切换无所谓 — 我们只是要让请求触发。
6. 在 Network 面板找到这个请求：
   - 方法：**PUT**
   - URL：`https://console.cnomnibot.com/api/thd/pg/api/v1/jobs/<JOB_ID>/start`（或 `/pause`）
7. 点请求 → **Headers** → **Request Headers**。
8. 把以下四个字段抄到 `~/.fcloud_console_config`：

   | 配置文件字段 | 来源 |
   |---|---|
   | `FCLOUD_CONSOLE_AUTH` | `authorization:` 的值（以 `eyJ...` 开头） |
   | `FCLOUD_CONSOLE_COOKIE` | `cookie:` 的整段值（`lang=...; JSESSIONID=...; ...`） |
   | `FCLOUD_CONSOLE_USERNAME` | 登录邮箱（也在 JWT payload 里；首次配置时手填一次） |
   | `FCLOUD_CONSOLE_JOB_ID` | URL `/jobs/` 与 `/start` 之间的 32 位 hex |

9. 打开 `~/.fcloud_console_config`，替换四个值，保存。
10. 验证：
    ```bash
    python3 scripts/fcloud/fcloud_workflow.py console-token-info
    ```
    应输出 `expired: false`，剩余 ~7 天。

## 3. 半自动刷新 — 用 HAR 文件（推荐）

浏览器可以把 Network 面板整体导出为 HAR（HTTP Archive JSON）。我们写了一个解析器自动提取字段。

### 一次性准备
无。脚本在 [scripts/fcloud/refresh_console_config_from_har.py](scripts/fcloud/refresh_console_config_from_har.py)。

### 刷新流程
1. 浏览器打开 https://console.cnomnibot.com/omnibot 。
2. F12 → Network → 勾 **Preserve log**。
3. 点任意任务的 Start 或 Pause。
4. Network 面板里 **任意行右键 → "Save all as HAR with content"** → 存为 `~/Downloads/console.har`（或别的位置）。
5. 跑：
   ```bash
   python3 scripts/fcloud/refresh_console_config_from_har.py ~/Downloads/console.har
   ```
6. 脚本会：
   - 找到最近一次 PUT 到 `/api/thd/pg/api/v1/jobs/<id>/{start,pause}` 的请求。
   - 抽出 `authorization`、`cookie`、`JOB_ID`。
   - 解码 JWT payload 拿到用户名/邮箱。
   - 把现有 `~/.fcloud_console_config` 备份到 `~/.fcloud_console_config.bak`。
   - 写入新配置，保留其他键（如 `FCLOUD_CONSOLE_BASE_URL`）。
7. 验证：`python3 scripts/fcloud/fcloud_workflow.py console-token-info`。
8. 删除 HAR 文件（它包含活的 JWT 和 cookie — 敏感度等同于配置文件本身）。

### 为什么走 HAR 而不是密码登录脚本
- 控制台登录走 SSO + 偶尔有验证码 / 2FA。
- 在脚本里存邮箱密码本身就是反复出现的凭证泄露风险。
- HAR 导出在 DevTools 里 5 秒搞定，剩下都是确定性解析。
- 我们手抄的就是 HAR 内容 — 只是把抄写动作脚本化。

## 4. 为什么不能完全自动化（"F12 自动捕获"梦想）

用户最初的设想是：**F12 → Network → 自动捕获 POST/PUT，自动拿到 authorization、cookie、username、job id**。三个架构性障碍：

| 障碍 | 原因 |
|---|---|
| `authorization` JWT 不在 `document.cookie` 里 | SPA 在登录后由 JS 内存维护并作为 header 发送。页面外的脚本无法读到。 |
| 浏览器跨域安全 | 即便在 `console.cnomnibot.com` 上挂书签脚本能从 SPA 的 localStorage 拿到 JWT（如果存那里），**也无法读取另一个请求的 `authorization` header**。Chrome 的 webRequest API 仅扩展可用，且 `authorization` 字段在很多场景下被屏蔽。 |
| 登录有 2FA / 验证码 | 即使 Playwright/Puppeteer 也得人工过验证码，自动化的意义就消失。 |

**所以 HAR 导出是合适的自动化粒度**：人来做安全敏感的登录 + 点击；机器做枯燥的解析。

## 5. 速查（TL;DR）

```bash
# 1. F12 → 点 Start/Pause → 存 HAR
# 2. 跑：
python3 scripts/fcloud/refresh_console_config_from_har.py ~/Downloads/console.har

# 3. 验证 + 清理：
python3 scripts/fcloud/fcloud_workflow.py console-token-info
rm ~/Downloads/console.har
```

## 6. 故障排查

| 现象 | 解决 |
|---|---|
| 刷新后 `console-token-info` 仍 expired | 字段抄错。JWT 是 `authorization` 的值，不是 `Set-Cookie`。重对 §2 第 8 步。 |
| HAR 脚本报 "no PUT to /jobs/.../start or /pause found" | 你在点击之前就抓了网络日志。刷新页面，点 Start/Pause，再存 HAR。 |
| `start-instance` 先 504 后重试 200 | 正常 — API 网关偶有超时，已内置重试。 |
| `start-instance` 返回 200 但 UI 上仍 'paused' | 浏览器缓存；刷新控制台页面。服务端状态已变。 |
| HAR 文件 encoding 报错 | 部分浏览器存 HAR 带 `\u0000` 或 BOM。给脚本加 `--encoding utf-8-sig`。 |
