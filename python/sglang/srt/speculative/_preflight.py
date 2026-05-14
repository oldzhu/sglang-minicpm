"""SOAR 2026 — Medusa/Ngram pre-flight state dump (CHANGE_0165).

Env-gated helper that captures a snapshot of the verify-step state into a
pickle file.  Used by the pre-flight diff tool to compare NgramWorker (the
reference path) against MedusaWorker (the test path) field-by-field.

Activation:
    SOAR_PREFLIGHT_DUMP_PATH=/tmp/dump.pkl   # absolute path

Behavior:
    - The first call truncates the file and writes a fresh list of records.
    - Subsequent calls append a new record (each is a dict).
    - When env var is unset or empty: noop (zero cost).

Record schema (free-form, but each record must have):
    {
        "tag": "ngram" | "medusa",
        "phase": "pre_verify" | "post_forward" | "post_verify",
        "step_id": int,                 # monotonic per-process counter
        "fields": {                     # arbitrary tensor / scalar dump
            "<name>": <CPU tensor or python primitive>,
            ...
        },
    }
"""
from __future__ import annotations

import os
import pickle
import threading
from typing import Any, Dict, Optional

import torch

_LOCK = threading.Lock()
_STEP_ID = 0
_RECORDS_BUFFER = []  # accumulated in-memory; flushed on each call


def _tensor_to_cpu(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        try:
            return t.detach().cpu().clone()
        except Exception:  # pylint: disable=broad-except
            return repr(t)
    return t


def dump_state(tag: str, phase: str, **fields: Any) -> None:
    """Append a snapshot to SOAR_PREFLIGHT_DUMP_PATH (if set).

    Args:
        tag: "ngram" or "medusa" (identifies the worker).
        phase: "pre_verify" | "post_forward" | "post_verify".
        **fields: tensors / scalars to record.  Tensors are detached + CPU-cloned.
    """
    path = os.environ.get("SOAR_PREFLIGHT_DUMP_PATH", "").strip()
    if not path:
        return

    global _STEP_ID  # pylint: disable=global-statement

    with _LOCK:
        _STEP_ID += 1
        record: Dict[str, Any] = {
            "tag": tag,
            "phase": phase,
            "step_id": _STEP_ID,
            "fields": {k: _tensor_to_cpu(v) for k, v in fields.items()},
        }
        _RECORDS_BUFFER.append(record)

        # Persist every time so a partial run still produces a valid file.
        # On the first write of this process we truncate.
        mode = "wb" if _STEP_ID == 1 else "wb"  # rewrite full buffer each time
        try:
            with open(path, mode) as f:
                pickle.dump(_RECORDS_BUFFER, f)
        except Exception as exc:  # pylint: disable=broad-except
            # Never break the server on dump failure.
            import logging

            logging.getLogger(__name__).warning(
                "preflight: dump_state failed: %s", exc
            )


def reset_for_test() -> None:
    """Test-only helper: clear the in-memory buffer."""
    global _STEP_ID  # pylint: disable=global-statement
    with _LOCK:
        _STEP_ID = 0
        _RECORDS_BUFFER.clear()
