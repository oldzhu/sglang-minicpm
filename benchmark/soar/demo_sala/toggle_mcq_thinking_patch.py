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


def _atomic_write_text(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def _strip_preamble(template: str) -> str:
    """Remove the CHANGE_0140 preamble from a template string. Tolerant."""
    rebind_line = "{%- set enable_thinking = _soar_ns.et -%}\n"
    idx_marker = template.find(CHAT_TEMPLATE_MCQ_PATCH_MARKER)
    if idx_marker < 0:
        return template
    idx_rebind = template.find(rebind_line, idx_marker)
    if idx_rebind < 0:
        # marker present but rebind missing → corrupted; refuse to touch
        raise RuntimeError("marker present but rebind line missing; manual fix needed")
    end = idx_rebind + len(rebind_line)
    return template[:idx_marker] + template[end:]


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


def apply_patch(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "skip: no chat_template.jinja and no embedded chat_template"
    if kind == "jinja":
        template = path.read_text(encoding="utf-8")
        if not template.strip():
            return "skip: chat_template.jinja empty"
        if CHAT_TEMPLATE_MCQ_PATCH_MARKER in template:
            return "noop: already patched"
        _atomic_write_text(path, CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE + template)
        return "patched (chat_template.jinja)"
    # embedded
    cfg = _load(path)
    template = cfg.get("chat_template", "")
    if CHAT_TEMPLATE_MCQ_PATCH_MARKER in template:
        return "noop: already patched"
    cfg["chat_template"] = CHAT_TEMPLATE_MCQ_PATCH_PREAMBLE + template
    _atomic_write(path, cfg)
    return "patched (tokenizer_config.json)"


def revert_patch(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "skip: no chat_template.jinja and no embedded chat_template"
    if kind == "jinja":
        template = path.read_text(encoding="utf-8")
        if CHAT_TEMPLATE_MCQ_PATCH_MARKER not in template:
            return "noop: not patched"
        _atomic_write_text(path, _strip_preamble(template))
        return "reverted (chat_template.jinja)"
    cfg = _load(path)
    template = cfg.get("chat_template", "")
    if CHAT_TEMPLATE_MCQ_PATCH_MARKER not in template:
        return "noop: not patched"
    cfg["chat_template"] = _strip_preamble(template)
    _atomic_write(path, cfg)
    return "reverted (tokenizer_config.json)"


def status(model_dir: Path) -> str:
    kind, path = _resolve_layout(model_dir)
    if kind is None:
        return "unknown (no chat_template found in either layout)"
    if kind == "jinja":
        template = path.read_text(encoding="utf-8")
    else:
        template = _load(path).get("chat_template", "") or ""
    layout = "chat_template.jinja" if kind == "jinja" else "tokenizer_config.json"
    return f"{'ON (patched)' if CHAT_TEMPLATE_MCQ_PATCH_MARKER in template else 'OFF (clean)'} [{layout}]"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to model directory containing tokenizer_config.json")
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
