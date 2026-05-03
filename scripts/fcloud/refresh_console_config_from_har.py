#!/usr/bin/env python3
"""Refresh ~/.fcloud_console_config from a browser-exported HAR file.

Usage:
    python3 scripts/fcloud/refresh_console_config_from_har.py <path/to/console.har> \
        [--config ~/.fcloud_console_config] [--encoding utf-8] [--dry-run]

Capture procedure (in browser):
    1. Open https://console.cnomnibot.com/omnibot, log in.
    2. F12 -> Network -> tick "Preserve log".
    3. Click any task's Start or Pause button.
    4. Right-click any row in Network -> "Save all as HAR with content".
    5. Run this script with the saved .har path.

What it does:
    - Parses the HAR JSON, finds the latest PUT to /api/thd/pg/api/v1/jobs/<id>/{start,pause}.
    - Extracts authorization, cookie, job_id.
    - Decodes JWT payload to recover username/email (if present).
    - Backs up the existing config to <config>.bak, writes a new config.
    - Preserves all keys not derived from the HAR (e.g. FCLOUD_CONSOLE_BASE_URL).

Security:
    - The HAR file contains live JWT + cookies. Treat it as sensitive as the
      config itself. Delete it after the script succeeds.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

JOB_URL_RE = re.compile(
    r"/api/thd/pg/api/v1/jobs/([0-9a-fA-F]{32})/(start|pause)\b"
)

DERIVED_KEYS = {
    "FCLOUD_CONSOLE_AUTH",
    "FCLOUD_CONSOLE_COOKIE",
    "FCLOUD_CONSOLE_JOB_ID",
    "FCLOUD_CONSOLE_USERNAME",
}


def b64url_decode(seg: str) -> bytes:
    pad = "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg + pad)


def decode_jwt_payload(jwt: str) -> dict:
    parts = jwt.split(".")
    if len(parts) < 2:
        return {}
    try:
        return json.loads(b64url_decode(parts[1]))
    except Exception:
        return {}


def find_request(har: dict) -> dict:
    """Return the latest matching PUT entry, or raise."""
    entries = har.get("log", {}).get("entries", [])
    matches = []
    for e in entries:
        req = e.get("request", {})
        if req.get("method", "").upper() != "PUT":
            continue
        url = req.get("url", "")
        m = JOB_URL_RE.search(url)
        if not m:
            continue
        matches.append((e.get("startedDateTime", ""), e, m.group(1)))
    if not matches:
        raise SystemExit(
            "No PUT to /api/thd/pg/api/v1/jobs/<id>/{start,pause} found in HAR.\n"
            "Capture the click event AFTER opening the Network panel."
        )
    matches.sort(key=lambda t: t[0])
    _, entry, job_id = matches[-1]
    return {"entry": entry, "job_id": job_id}


def header_value(headers: list, name: str) -> str | None:
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            v = h.get("value", "")
            return v if v else None
    return None


def parse_existing(path: Path) -> dict:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def write_config(path: Path, merged: dict[str, str]) -> None:
    # Preserve a stable ordering: known keys first, then anything else.
    order = [
        "FCLOUD_CONSOLE_BASE_URL",
        "FCLOUD_CONSOLE_AUTH",
        "FCLOUD_CONSOLE_COOKIE",
        "FCLOUD_CONSOLE_USERNAME",
        "FCLOUD_CONSOLE_JOB_ID",
    ]
    seen = set()
    lines = []
    for k in order:
        if k in merged:
            lines.append(f"{k}={merged[k]}")
            seen.add(k)
    for k, v in merged.items():
        if k in seen:
            continue
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("har", help="Path to the HAR file exported from DevTools.")
    p.add_argument(
        "--config",
        default=str(Path.home() / ".fcloud_console_config"),
        help="Path to the config file to refresh (default ~/.fcloud_console_config).",
    )
    p.add_argument("--encoding", default="utf-8", help="HAR file encoding (default utf-8).")
    p.add_argument("--dry-run", action="store_true", help="Print only, don't write.")
    args = p.parse_args()

    har_path = Path(args.har).expanduser()
    cfg_path = Path(args.config).expanduser()

    if not har_path.exists():
        print(f"ERROR: HAR file not found: {har_path}", file=sys.stderr)
        return 2

    try:
        har = json.loads(har_path.read_text(encoding=args.encoding))
    except UnicodeDecodeError as e:
        print(
            f"ERROR: HAR decode failed ({e}). Try --encoding utf-8-sig.",
            file=sys.stderr,
        )
        return 2
    except json.JSONDecodeError as e:
        print(f"ERROR: HAR JSON parse failed: {e}", file=sys.stderr)
        return 2

    found = find_request(har)
    entry = found["entry"]
    job_id = found["job_id"]
    headers = entry.get("request", {}).get("headers", [])

    auth = header_value(headers, "authorization")
    cookie = header_value(headers, "cookie")

    if not auth:
        print("ERROR: matched PUT request has no `authorization` header.", file=sys.stderr)
        return 2
    if not cookie:
        print("ERROR: matched PUT request has no `cookie` header.", file=sys.stderr)
        return 2

    jwt = auth.split()[-1]  # strip optional "Bearer " prefix
    payload = decode_jwt_payload(jwt)
    username = (
        payload.get("preferred_username")
        or payload.get("email")
        or payload.get("username")
        or payload.get("name")
        or payload.get("sub")
        or ""
    )

    exp_str = ""
    if "exp" in payload:
        try:
            exp_dt = datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc)
            exp_str = exp_dt.isoformat()
        except Exception:
            pass

    existing = parse_existing(cfg_path)
    merged = dict(existing)  # keep all unrelated keys
    merged["FCLOUD_CONSOLE_AUTH"] = auth
    merged["FCLOUD_CONSOLE_COOKIE"] = cookie
    merged["FCLOUD_CONSOLE_JOB_ID"] = job_id
    if username:
        merged["FCLOUD_CONSOLE_USERNAME"] = username
    elif "FCLOUD_CONSOLE_USERNAME" not in merged:
        print(
            "WARN: could not derive USERNAME from JWT; please set FCLOUD_CONSOLE_USERNAME manually once.",
            file=sys.stderr,
        )
    merged.setdefault("FCLOUD_CONSOLE_BASE_URL", "https://console.cnomnibot.com")

    print("=" * 60)
    print("REFRESH SUMMARY")
    print("=" * 60)
    print(f"  Source HAR : {har_path}")
    print(f"  Target cfg : {cfg_path}")
    print(f"  Job id     : {job_id}")
    print(f"  Username   : {merged.get('FCLOUD_CONSOLE_USERNAME', '<unset>')}")
    print(f"  JWT exp    : {exp_str or '<not in payload>'}")
    print(f"  Auth len   : {len(auth)} chars")
    print(f"  Cookie len : {len(cookie)} chars")

    if args.dry_run:
        print("\n--dry-run set; no file changes.")
        return 0

    if cfg_path.exists():
        bak = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        bak.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nBackup: {bak}")

    write_config(cfg_path, merged)
    cfg_path.chmod(0o600)
    print(f"WROTE  : {cfg_path} (mode 600)")
    print("\nVerify:")
    print("  python3 scripts/fcloud/fcloud_workflow.py console-token-info")
    print("\nThen DELETE the HAR (it contains your live JWT and cookies):")
    print(f"  rm {har_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
