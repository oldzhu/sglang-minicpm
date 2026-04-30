"""In-place toggle for the CHANGE_0140 mcq-thinking chat_template patch.

This is a fcloud A/B testing convenience: it lets us flip an already-quantized
model directory's tokenizer_config.json between patched (enable_thinking=false
for mcq prompts) and unpatched, without re-running preprocess_model.py /
re-quantization.

The production submission flow still goes through preprocess_model.py
unchanged.

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

# Reuse the exact constants used by the production preprocess_model.py to
# guarantee the patched template is byte-identical to the submission path.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from preprocess_model import (  # noqa: E402
    CHAT_TEMPLATE_MCQ_PATCH_MARKER,
    CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE,
)


def _load(tok_path: Path) -> dict:
    with tok_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(tok_path: Path, cfg: dict) -> None:
    tmp_path = tok_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    tmp_path.replace(tok_path)


def apply_patch(tok_path: Path) -> str:
    cfg = _load(tok_path)
    template = cfg.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        return "skip: chat_template missing/empty"
    if CHAT_TEMPLATE_MCQ_PATCH_MARKER in template:
        return "noop: already patched"
    cfg["chat_template"] = CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE + template
    _atomic_write(tok_path, cfg)
    return "patched"


def revert_patch(tok_path: Path) -> str:
    cfg = _load(tok_path)
    template = cfg.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        return "skip: chat_template missing/empty"
    if CHAT_TEMPLATE_MCQ_PATCH_MARKER not in template:
        return "noop: not patched"
    # Remove the preamble. Be permissive: strip the marker line and everything
    # up to and including the trailing top-level rebind line. We rely on the
    # fact that the preamble is contiguous and ends with the rebind line.
    rebind_line = "{%- set enable_thinking = _soar_ns.et -%}\n"
    idx_marker = template.find(CHAT_TEMPLATE_MCQ_PATCH_MARKER)
    if idx_marker != 0:
        # Marker should be at position 0 because we always prepend; if not,
        # fall back to a tolerant strip.
        pass
    idx_rebind = template.find(rebind_line)
    if idx_rebind < 0:
        return "error: marker present but rebind line missing; manual fix needed"
    end = idx_rebind + len(rebind_line)
    new_template = template[:idx_marker] + template[end:]
    cfg["chat_template"] = new_template
    _atomic_write(tok_path, cfg)
    return "reverted"


def status(tok_path: Path) -> str:
    cfg = _load(tok_path)
    template = cfg.get("chat_template", "")
    if not isinstance(template, str):
        return "unknown (chat_template not a string)"
    return "ON (patched)" if CHAT_TEMPLATE_MCQ_PATCH_MARKER in template else "OFF (clean)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to model directory containing tokenizer_config.json")
    p.add_argument("--mode", required=True, choices=["on", "off", "status"])
    args = p.parse_args()

    model_dir = Path(args.model_dir).resolve()
    tok_path = model_dir / "tokenizer_config.json"
    if not tok_path.exists():
        print(f"ERROR: {tok_path} does not exist", file=sys.stderr)
        return 2

    if args.mode == "status":
        print(f"[{model_dir}] mcq-thinking patch: {status(tok_path)}")
        return 0
    if args.mode == "on":
        result = apply_patch(tok_path)
    else:
        result = revert_patch(tok_path)
    print(f"[{model_dir}] mode={args.mode}: {result}")
    print(f"[{model_dir}] now: {status(tok_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
