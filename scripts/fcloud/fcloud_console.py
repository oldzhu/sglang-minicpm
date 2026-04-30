#!/usr/bin/env python3
"""fcloud_console.py - Start/stop a fcloud (omnibot) training task via the console API.

The omnibot console exposes private REST endpoints for the task-list start/stop
buttons.  We capture the JWT once from a browser session and reuse it for ~7
days (token TTL is ~weekly).  When the JWT expires the script exits with a
clear message instructing the user to refresh `~/.fcloud_console_config`.

Endpoints (captured from console.cnomnibot.com SPA):
    PUT https://console.cnomnibot.com/api/thd/pg/api/v1/jobs/{JOB_ID}/start
    PUT https://console.cnomnibot.com/api/thd/pg/api/v1/jobs/{JOB_ID}/pause
    body: {"id": "<JOB_ID>"}

Config file: ~/.fcloud_console_config (KEY=VALUE lines, '#' comments allowed)
    FCLOUD_CONSOLE_BASE_URL=https://console.cnomnibot.com
    FCLOUD_CONSOLE_AUTH=<JWT eyJ...>
    FCLOUD_CONSOLE_COOKIE=<full Cookie header value>
    FCLOUD_CONSOLE_USERNAME=<email>
    FCLOUD_CONSOLE_JOB_ID=<32-char hex id>

Usage:
    python3 scripts/fcloud/fcloud_console.py start
    python3 scripts/fcloud/fcloud_console.py pause
    python3 scripts/fcloud/fcloud_console.py token-info     # decode JWT exp
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

CONFIG_PATH = Path(os.environ.get("FCLOUD_CONSOLE_CONFIG", str(Path.home() / ".fcloud_console_config")))
DEFAULT_BASE_URL = "https://console.cnomnibot.com"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> Dict[str, str]:
    if not CONFIG_PATH.exists():
        sys.exit(
            f"ERROR: {CONFIG_PATH} not found.\n"
            "Create it with the following keys (capture from browser DevTools):\n"
            "  FCLOUD_CONSOLE_AUTH=<JWT from `authorization` header>\n"
            "  FCLOUD_CONSOLE_COOKIE=<full `Cookie` header value>\n"
            "  FCLOUD_CONSOLE_USERNAME=<email>\n"
            "  FCLOUD_CONSOLE_JOB_ID=<32-char job id from URL>\n"
            "Optional: FCLOUD_CONSOLE_BASE_URL (defaults to https://console.cnomnibot.com)\n"
        )
    cfg: Dict[str, str] = {}
    for raw in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip().strip('"').strip("'")
    required = ["FCLOUD_CONSOLE_AUTH", "FCLOUD_CONSOLE_JOB_ID"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        sys.exit(f"ERROR: {CONFIG_PATH} missing keys: {', '.join(missing)}")
    cfg.setdefault("FCLOUD_CONSOLE_BASE_URL", DEFAULT_BASE_URL)
    return cfg


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def decode_jwt_exp(jwt: str) -> Optional[int]:
    parts = jwt.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    # Add padding for urlsafe base64
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        exp = data.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def check_jwt_freshness(jwt: str) -> Dict[str, object]:
    exp = decode_jwt_exp(jwt)
    if exp is None:
        return {"valid_format": False, "exp": None, "expires_at": None, "days_remaining": None}
    now = int(_dt.datetime.now().timestamp())
    return {
        "valid_format": True,
        "exp": exp,
        "expires_at": _dt.datetime.fromtimestamp(exp).isoformat(),
        "days_remaining": round((exp - now) / 86400, 2),
        "expired": exp <= now,
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _build_request(cfg: Dict[str, str], action: str) -> urllib.request.Request:
    job_id = cfg["FCLOUD_CONSOLE_JOB_ID"]
    base = cfg["FCLOUD_CONSOLE_BASE_URL"].rstrip("/")
    url = f"{base}/api/thd/pg/api/v1/jobs/{job_id}/{action}"
    body = json.dumps({"id": job_id}).encode("utf-8")

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN",
        "authorization": cfg["FCLOUD_CONSOLE_AUTH"],
        "content-type": "application/json;charset=UTF-8",
        "ltskey": "lts",
        "origin": base,
        "referer": f"{base}/omnibot",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) fcloud_console.py/1.0",
    }
    if cfg.get("FCLOUD_CONSOLE_COOKIE"):
        headers["Cookie"] = cfg["FCLOUD_CONSOLE_COOKIE"]
    if cfg.get("FCLOUD_CONSOLE_USERNAME"):
        headers["username"] = cfg["FCLOUD_CONSOLE_USERNAME"]

    req = urllib.request.Request(url, data=body, method="PUT", headers=headers)
    return req


def _do_action(cfg: Dict[str, str], action: str) -> Dict[str, object]:
    """Returns {ok, status, body, hint}."""
    fresh = check_jwt_freshness(cfg["FCLOUD_CONSOLE_AUTH"])
    if fresh.get("expired"):
        return {
            "ok": False,
            "status": 0,
            "body": "",
            "hint": (
                f"JWT in {CONFIG_PATH} expired at {fresh['expires_at']}. "
                "Re-capture the `authorization` header from browser DevTools "
                "(F12 -> Network tab -> click any task action -> copy `authorization`)."
            ),
        }

    req = _build_request(cfg, action)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
        return {"ok": 200 <= status < 300, "status": status, "body": body, "hint": ""}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        hint = ""
        if e.code in (401, 403):
            hint = (
                f"HTTP {e.code} likely means the JWT is invalid/expired. "
                f"Refresh {CONFIG_PATH} `FCLOUD_CONSOLE_AUTH` (and Cookie) from "
                "browser DevTools and retry."
            )
        return {"ok": False, "status": e.code, "body": body, "hint": hint}
    except urllib.error.URLError as e:
        return {"ok": False, "status": 0, "body": "", "hint": f"Network error: {e}"}


def start_job(cfg: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    return _do_action(cfg or load_config(), "start")


def pause_job(cfg: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    return _do_action(cfg or load_config(), "pause")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_result(action: str, result: Dict[str, object]) -> int:
    print(f"[fcloud_console][{action}] HTTP {result['status']}")
    if result["body"]:
        # truncate huge bodies
        body = str(result["body"])
        print(f"[fcloud_console][{action}] body: {body[:500]}")
    if result["hint"]:
        print(f"[fcloud_console][{action}] HINT: {result['hint']}")
    if result["ok"]:
        print(f"[fcloud_console][{action}] OK")
        return 0
    print(f"[fcloud_console][{action}] FAILED")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "pause", "stop", "token-info"],
                   help="`stop` is an alias for `pause`")
    args = p.parse_args()

    if args.action == "token-info":
        cfg = load_config()
        info = check_jwt_freshness(cfg["FCLOUD_CONSOLE_AUTH"])
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0 if not info.get("expired", True) else 2

    action = "pause" if args.action == "stop" else args.action
    result = _do_action(load_config(), action)
    return _print_result(action, result)


if __name__ == "__main__":
    raise SystemExit(main())
