"""
Phase B step 1 — probe modelopt 0.43 NVFP4 internals.

Loads MiniCPM-SALA-Copy, runs mtq.quantize with NVFP4_DEFAULT_CFG on a couple
of calibration samples, then dumps the structure of one quantized Linear so we
know what attribute names to read/write in the FourOverSix helper.

Run on fcloud after `source /root/submission_sim/prepare_env.sh` (modelopt must
be installed via SOAR_QUANT_PROFILE=nvfp4 or nvfp4_fos).
"""

import json
import os
import sys
from pathlib import Path

import torch


def main():
    src = Path(os.environ.get("PROBE_SRC", "/root/models/openbmb/MiniCPM-SALA-Copy"))
    out_path = Path(os.environ.get("PROBE_OUT", "/root/probe_nvfp4_internals.txt"))

    print(f"[probe] src={src}")
    print(f"[probe] out={out_path}")

    import modelopt.torch.quantization as mtq
    print(f"[probe] modelopt {mtq.__version__ if hasattr(mtq, '__version__') else '?'}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.eval()

    import copy as _copy
    config = _copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    print("[probe] NVFP4_DEFAULT_CFG keys:", list(config.keys()))
    if "quant_cfg" in config:
        for k, v in list(config["quant_cfg"].items())[:6]:
            print(f"[probe]   quant_cfg[{k!r}] = {v}")

    calibration_texts = [
        "The capital of France is",
        "Photosynthesis converts",
        "In computer science,",
    ]

    def forward_loop(m):
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                m(**inputs)

    print("[probe] running mtq.quantize ...")
    mtq.quantize(model, config, forward_loop=forward_loop)
    print("[probe] mtq.quantize done")

    # Find one quantized Linear-ish module
    target = None
    target_name = None
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_quantizer") or hasattr(mod, "_weight_quantizer"):
            wq = getattr(mod, "weight_quantizer", None) or getattr(mod, "_weight_quantizer", None)
            # Skip disabled ones (e.g. lm_head excluded)
            enabled = getattr(wq, "is_enabled", None)
            if callable(enabled):
                enabled = enabled()
            if enabled is False:
                continue
            target = mod
            target_name = name
            break

    lines = []

    def w(s=""):
        print(s)
        lines.append(s)

    w(f"[probe] target module: {target_name}  type={type(target).__name__}")
    if target is None:
        w("[probe] NO QUANTIZED LINEAR FOUND")
        out_path.write_text("\n".join(lines))
        return

    # Module-level attrs
    w("\n=== target module attrs ===")
    for n in sorted(dir(target)):
        if n.startswith("__"):
            continue
        try:
            v = getattr(target, n)
        except Exception as e:
            w(f"  {n}: <error: {e}>")
            continue
        if callable(v):
            continue
        if isinstance(v, torch.Tensor):
            w(f"  {n}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, (int, float, str, bool, type(None))):
            w(f"  {n}: {type(v).__name__}={v!r}")
        elif isinstance(v, dict):
            w(f"  {n}: dict keys={list(v.keys())[:8]}")
        else:
            w(f"  {n}: {type(v).__name__}")

    # Look at weight_quantizer / _weight_quantizer
    for wq_name in ("weight_quantizer", "_weight_quantizer", "input_quantizer", "_input_quantizer"):
        wq = getattr(target, wq_name, None)
        if wq is None:
            continue
        w(f"\n=== {wq_name} attrs (type={type(wq).__name__}) ===")
        for n in sorted(dir(wq)):
            if n.startswith("__"):
                continue
            try:
                v = getattr(wq, n)
            except Exception as e:
                w(f"  {n}: <error: {e}>")
                continue
            if callable(v):
                continue
            if isinstance(v, torch.Tensor):
                w(f"  {n}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            elif isinstance(v, (int, float, str, bool, type(None))):
                w(f"  {n}: {type(v).__name__}={v!r}")
            elif isinstance(v, dict):
                w(f"  {n}: dict keys={list(v.keys())[:8]}")
            else:
                w(f"  {n}: {type(v).__name__}")

    # Inspect quantized weight tensor's class type (TensorQuantTensor subclass?)
    if hasattr(target, "weight"):
        w_tensor = target.weight
        w(f"\n=== target.weight ===")
        w(f"  type={type(w_tensor).__name__} mro={[c.__name__ for c in type(w_tensor).__mro__]}")
        if isinstance(w_tensor, torch.Tensor):
            w(f"  shape={tuple(w_tensor.shape)} dtype={w_tensor.dtype}")
        for n in sorted(dir(w_tensor)):
            if n.startswith("__"):
                continue
            try:
                v = getattr(w_tensor, n)
            except Exception:
                continue
            if callable(v):
                continue
            if isinstance(v, torch.Tensor):
                w(f"    {n}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, (int, float, str, bool)):
                w(f"    {n}: {v!r}")

    # Probe how export_hf_checkpoint sees this — what's in the state_dict?
    w("\n=== state_dict entries containing target name ===")
    sd = model.state_dict()
    for k, v in sd.items():
        if target_name in k:
            if isinstance(v, torch.Tensor):
                w(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                w(f"  {k}: {type(v).__name__}")

    # Dump distinct module class names that have a weight_quantizer
    w("\n=== distinct quantized module class names ===")
    classes = set()
    for name, mod in model.named_modules():
        if hasattr(mod, "weight_quantizer") or hasattr(mod, "_weight_quantizer"):
            classes.add(type(mod).__name__)
    for c in sorted(classes):
        w(f"  {c}")

    out_path.write_text("\n".join(lines))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
