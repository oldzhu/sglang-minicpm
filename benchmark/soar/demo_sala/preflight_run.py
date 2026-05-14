#!/usr/bin/env python3
"""SOAR CHANGE_0165 — Pre-flight runner.

Sends one short prompt to a running sglang server and waits for ~5 decode
tokens.  The server must be launched with SOAR_PREFLIGHT_DUMP_PATH=<file> so
that the worker hooks (NgramWorker / MedusaWorker) write a pickle of the
verify-step state to <file>.

Usage:
    python3 preflight_run.py --url http://127.0.0.1:30000 \
        [--prompt "Capital of France is"] [--max-new-tokens 3]

The script is intentionally minimal — it does NOT touch sglang internals.
All capture happens server-side via the dump hook.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


DEFAULT_PROMPT = (
    "Question: What is the capital city of France?\n"
    "Answer:"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=3)
    args = ap.parse_args()

    # /generate is sglang's native endpoint.
    payload = {
        "text": args.prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url.rstrip("/") + "/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode("utf-8")
    elapsed = time.time() - t0

    print(f"[preflight] {elapsed:.2f}s response:")
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2)[:1200])
    except Exception:
        print(body[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
