# HOW-TO — Refresh `~/.fcloud_console_config` (weekly JWT)

**Status**: living reference doc — update if console UI changes.
**Frequency**: ~once per week (JWT TTL ≈ 7 days).
**Symptom that you need this**: `start-instance` / `pause-instance` returns HTTP 401 or 403, or `console-token-info` reports `expired: true`.

## 1. Quick check (run first)

```bash
python3 scripts/fcloud/fcloud_workflow.py console-token-info
```

If output shows `expired: false` and remaining > 1 day, no action needed.

## 2. Manual refresh (canonical, ~2 min)

1. Open https://console.cnomnibot.com/omnibot in Chrome/Edge, log in.
2. **F12** → switch to the **Network** tab. Make sure recording is on (the red dot).
3. Tick **Preserve log** (so the request survives a navigation).
4. (Optional) Type `start|pause` in the filter box to narrow results.
5. On the page, click any task's **Start** or **Pause** button. The actual instance state doesn't matter — we just need the request to fire.
6. In the Network panel, find the request:
   - Method: **PUT**
   - URL: `https://console.cnomnibot.com/api/thd/pg/api/v1/jobs/<JOB_ID>/start` (or `/pause`)
7. Click the request → **Headers** tab → **Request Headers**.
8. Copy these four values:

   | Field in `~/.fcloud_console_config` | Source in headers |
   |---|---|
   | `FCLOUD_CONSOLE_AUTH` | value of `authorization:` (starts with `eyJ...`) |
   | `FCLOUD_CONSOLE_COOKIE` | value of `cookie:` (the entire `lang=...; JSESSIONID=...; ...` string) |
   | `FCLOUD_CONSOLE_USERNAME` | your login email (also embedded in JWT payload — first manual entry only) |
   | `FCLOUD_CONSOLE_JOB_ID` | 32-char hex from the URL between `/jobs/` and `/start` |

9. Open `~/.fcloud_console_config`, replace the four values, save.
10. Verify:
    ```bash
    python3 scripts/fcloud/fcloud_workflow.py console-token-info
    ```
    Should print `expired: false` and ~7 days remaining.

## 3. Semi-automated refresh via HAR file (recommended)

Browsers can export the entire Network panel as a HAR (HTTP Archive) JSON. We have a parser that extracts the config fields automatically.

### One-time setup
None. The helper script is at [scripts/fcloud/refresh_console_config_from_har.py](scripts/fcloud/refresh_console_config_from_har.py).

### Refresh procedure
1. Open https://console.cnomnibot.com/omnibot in Chrome/Edge.
2. F12 → Network → tick **Preserve log**.
3. Click any task's Start or Pause button.
4. In the Network panel, **right-click any row → "Save all as HAR with content"** → save as `~/Downloads/console.har` (or anywhere).
5. Run:
   ```bash
   python3 scripts/fcloud/refresh_console_config_from_har.py ~/Downloads/console.har
   ```
6. The script:
   - Locates the latest PUT to `/api/thd/pg/api/v1/jobs/<id>/{start,pause}`.
   - Extracts `authorization`, `cookie`, `JOB_ID`.
   - Decodes the JWT payload to recover the username/email.
   - Backs up your existing `~/.fcloud_console_config` to `~/.fcloud_console_config.bak`.
   - Writes the new config preserving any extra keys (e.g., `FCLOUD_CONSOLE_BASE_URL`).
7. Verify: `python3 scripts/fcloud/fcloud_workflow.py console-token-info`.
8. Delete the HAR file (it contains your live JWT and cookies — same sensitivity as the config itself).

### Why HAR and not a password-based login script
- Logging into the console requires SSO + (sometimes) a captcha or 2FA challenge.
- Storing the user's email/password in a script is a recurring credential-leak risk.
- HAR export takes 5 seconds in DevTools and the rest is fully deterministic parsing.
- HAR is exactly what the manual instructions copy by hand — we just script the copying.

## 4. Why we can't fully automate (the F12 dream)

The user's idea was: "F12 → Network → capture POST/PUT request and get authorization, cookie, username, job id automatically." There are three architectural blockers:

| Blocker | Why |
|---|---|
| `authorization` JWT is not in `document.cookie` | It's set by the SPA's JS in-memory after login and attached as a request header. Page-context JS can read it only if you control the SPA's state store. We don't. |
| Cross-origin browser security | A bookmarklet running on `console.cnomnibot.com` could read the JWT from the SPA's localStorage/sessionStorage IF that's where it's stored, but **a script outside the page can't read network request headers** (Chrome's webRequest API is restricted to extensions, and observers cannot see `authorization` headers without extra permissions). |
| Login flow has 2FA / captcha | Even Playwright/Puppeteer can't auto-login through a captcha challenge without a real human, defeating the automation. |

**Therefore HAR-export is the right tier of automation:** human handles the security-sensitive login + click; the machine handles the boring parsing.

## 5. Cheatsheet (TL;DR)

```bash
# 1. F12 in browser, click any Start/Pause, save HAR
# 2. Run:
python3 scripts/fcloud/refresh_console_config_from_har.py ~/Downloads/console.har

# 3. Verify and clean up:
python3 scripts/fcloud/fcloud_workflow.py console-token-info
rm ~/Downloads/console.har
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `console-token-info` still shows expired after refresh | Wrong field copied. The JWT is the `authorization` value, not `Set-Cookie`. Re-check step 8 in §2. |
| HAR script reports "no PUT to /jobs/.../start or /pause found" | You captured the network log before clicking. Refresh, click Start/Pause, re-save the HAR. |
| `start-instance` returns HTTP 504 then HTTP 200 on retry | Normal — the API gateway sometimes timeouts on first call. Built-in retry handles it. |
| `start-instance` returns HTTP 200 but instance still 'paused' on UI | Browser cache; refresh the console page. State has changed server-side. |
| HAR file rejected — "encoding"-related parse error | Some browsers save HAR with `\u0000` or BOM. Pass `--encoding utf-8-sig` to the script. |
