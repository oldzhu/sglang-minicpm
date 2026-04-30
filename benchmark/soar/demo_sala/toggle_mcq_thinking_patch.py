"""In-place toggle for the CHANGE_0140 mcq-thinking chat_template patch.

This is a fcloud A/B testing convenience: it lets us flip an already-quantized
model directory's chat_template between patched (mcq prompts get a pre-seeded
closed `<think>\\n\\n</think>\\n\\n` block right after the assistant header)
and clean, without re-running preprocess_model.py / re-quantization.

Supports both HF storage layouts:
  (1) embedded:  tokenizer_config.json["chat_template"] = "<jinja>"
  (2) external:  chat_template.jinja file (raw jinja source)

The production submission flow (preprocess_model.py) reuses the SAME
constants and helpers from this module.

Usage:
    python toggle_mcq_thinking_patch.py --model-dir <dir> --mode on
    python toggle_mcq_thinking_patch.py --model-dir <dir> --mode off
    python toggle_mcq_thinking_patch.py --model-dir <dir> --mode status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from preprocess_model import (  # noqa: E402
    CHAT_TEMPLATE_MCQ_PATCH_MARKER,
    CHAT_TEMPLATE_MCQ_PATCH_MARKER_V1,
    _apply_mcq_patch_to_template,
    _revert_mcq_patch_from_template,
)


def _resolve_layout(model_dir: Path):
    """Returns (kind, path) where kind in {'jinja','embedded',None}."""
    jinja = model_dir / "chat_template.jinja"
    tok = model_dir / "tokenizer_config.json"
    if jinja.exists():
        return ("jinja", jinja)
    if tok.exists():
        try:
            cfg = json.loads(tok.read_text(encoding="utf-8"))
            t = cfg.get("chat_template")
            if isinstance(t, str) and t.strip():
                return ("embedded", tok)
        except Exception:
            pass
    return (None, None)


def _read_template(kind, path):
    if kind == "jinja":
        return path.read_text(encoding="utf-8")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg.get("chat_template", "") or ""


def _write_template(kind, path, new_template):
    if kind == "jinja":
        tmp = path.with_suffix(".jinja.tmp")
        tmp.write_text(new_template, encoding="utf-8")
        tmp.replace(path)
        return
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["chat_template"] = new_template
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def apply_patch(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "skip: no chat_template.jinja and no embedded chat_template"
    template = _read_template(kind, path)
    if not template.strip():
        return f"skip: {kind} chat_template empty"
    new = _apply_mcq_patch_to_template(template)
    if new is None:
        return "noop: already v2-patched"
    if new == "":
        return "error: trailing add_generation_prompt block not found in expected shape"
    _write_template(kind, path, new)
    return f"patched (v2, {kind})"


def revert_patch(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "skip: no chat_template.jinja and no embedded chat_template"
    template = _read_template(kind, path)
    new = _revert_mcq_patch_from_template(template)
    if new is None:
        return "noop: not patched"
    _write_template(kind, path, new)
    return f"reverted ({kind})"


def status(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "unknown (no chat_template found)"
    template = _read_template(kind, path)
    layout = "chat_template.jinja" if kind == "jinja" else "tokenizer_config.json"
    has_v2 = CHAT_TEMPLATE_MCQ_PATCH_MARKER in template
    has_v1 = CHAT_TEMPLATE_MCQ_PATCH_MARKER_V1 in template
    if has_v2 and has_v1:
        state = "ON (v2; legacy v1 also present)"
    elif has_v2:
        state = "ON (v2 patched)"
    elif has_v1:
        state = "LEGACY v1 only (no-op; run --mode off to clean)"
    else:
        state = "OFF (clean)"
    return f"{state} [{layout}]"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to model directory")
    p.add_argument("--mode", required=True, choices=["on", "off", "status"])
    args = p.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist", file=sys.stderr)
        return 2

    if args.mode == "status":
        print(f"[{model_dir}] mcq-thinking patch: {status(model_dir)}")
        return 0
    if args.mode == "on":
        result = apply_patch(model_dir)
    else:
        result = revert_patch(model_dir)
    print(f"[{model_dir}] mode={args.mode}: {result}")
    print(f"[{model_dir}] now: {status(model_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
