"""Submission preprocess entry for MiniCPM-SALA.

This script follows SOAR toolkit submission contract:
    python preprocess_model.py --input <raw_model_dir> --output <processed_model_dir>

Modes:
- copy: copy raw model as-is (default)
- gptq: run GPTQ offline quantization with GPTQModel
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def copy_model(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for f in sorted(src.iterdir()):
        if f.name.startswith("."):
            continue
        target = dst / f.name
        if target.exists():
            continue
        if f.is_dir():
            shutil.copytree(f, target)
        else:
            shutil.copy2(f, target)
        count += 1
    return count


def has_command(cmd: str) -> bool:
    try:
        subprocess.run(["bash", "-lc", f"command -v {cmd}"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def gptq_preflight(src: Path) -> None:
    # Validate minimum model artifacts and python dependency availability.
    if not has_command("python3"):
        raise RuntimeError("python3 not found in environment")

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        raise RuntimeError(
            "Input model missing required files for quant-prep: "
            + ", ".join(missing)
        )

    import importlib.util

    spec = importlib.util.find_spec("gptqmodel")
    if spec is None:
        raise RuntimeError(
            "SOAR_QUANT_MODE=gptq but `gptqmodel` is not installed. "
            "Install it in prepare_env.sh (e.g., `uv pip install gptqmodel`) "
            "or use SOAR_QUANT_MODE=copy."
        )


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value}") from exc


def _parse_optional_int_list_env(name: str) -> Optional[List[int]]:
    value = os.environ.get(name)
    if value is None:
        return None
    raw = value.strip()
    if raw == "":
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list, got: {value}") from exc


def _calibration_length_bucket(record: dict) -> str:
    prompt_tokens = record.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        return "len_unknown"
    if prompt_tokens <= 4096:
        return "len_0_4k"
    if prompt_tokens <= 32768:
        return "len_4k_32k"
    if prompt_tokens <= 131072:
        return "len_32k_128k"
    return "len_128k_plus"


def _calibration_bucket_key(
    record: dict,
    task_balance: bool,
    use_prompt_tokens: bool,
) -> str:
    parts: List[str] = []
    if task_balance:
        parts.append(f"task={record.get('task', 'unknown')}")
    if use_prompt_tokens:
        parts.append(_calibration_length_bucket(record))
    if not parts:
        return "all"
    return "|".join(parts)


def _largest_remainder_allocate(capacities: Dict[str, int], total: int) -> Dict[str, int]:
    allocation = {key: 0 for key in capacities}
    if total <= 0:
        return allocation

    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return allocation

    fractional: List[Tuple[float, str]] = []
    assigned = 0
    for key, capacity in capacities.items():
        if capacity <= 0:
            continue
        raw = total * capacity / total_capacity
        whole = min(capacity, int(raw))
        allocation[key] = whole
        assigned += whole
        fractional.append((raw - whole, key))

    remaining = total - assigned
    if remaining <= 0:
        return allocation

    fractional.sort(key=lambda item: (-item[0], item[1]))
    for _, key in fractional:
        if remaining <= 0:
            break
        if allocation[key] >= capacities[key]:
            continue
        allocation[key] += 1
        remaining -= 1

    return allocation


def _select_calibration_records(records: List[dict], max_samples: int) -> Tuple[List[dict], dict]:
    mode = os.environ.get("SOAR_GPTQ_CALIBRATION_SAMPLING", "sequential").strip().lower()
    seed = _parse_int_env("SOAR_GPTQ_CALIBRATION_SEED", 20260320)
    task_balance = _env_truthy("SOAR_GPTQ_CALIBRATION_TASK_BALANCE", default=True)
    use_prompt_tokens = _env_truthy(
        "SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS", default=True
    )

    if mode not in {"sequential", "shuffled", "stratified"}:
        raise ValueError(
            "SOAR_GPTQ_CALIBRATION_SAMPLING must be one of sequential, shuffled, stratified"
        )

    available = len(records)
    if max_samples <= 0 or max_samples >= available:
        selected = list(records)
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    if mode == "sequential":
        selected = list(records[:max_samples])
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    rng = random.Random(seed)

    if mode == "shuffled":
        selected = list(records)
        rng.shuffle(selected)
        selected = selected[:max_samples]
        summary = {
            "mode": mode,
            "seed": seed,
            "available": available,
            "selected": len(selected),
            "task_balance": task_balance,
            "use_prompt_tokens": use_prompt_tokens,
            "selected_buckets": {"all": len(selected)},
        }
        return selected, summary

    buckets: Dict[str, List[dict]] = {}
    for record in records:
        key = _calibration_bucket_key(record, task_balance, use_prompt_tokens)
        buckets.setdefault(key, []).append(record)

    bucket_items = sorted(buckets.items())
    for _, bucket_records in bucket_items:
        rng.shuffle(bucket_records)

    base_counts = {key: 0 for key, _ in bucket_items}
    if max_samples >= len(bucket_items):
        for key, bucket_records in bucket_items:
            if bucket_records:
                base_counts[key] = 1
    else:
        ranked = sorted(bucket_items, key=lambda item: (-len(item[1]), item[0]))
        for key, _ in ranked[:max_samples]:
            base_counts[key] = 1

    remaining = max_samples - sum(base_counts.values())
    capacities = {
        key: max(0, len(bucket_records) - base_counts[key])
        for key, bucket_records in bucket_items
    }
    extra_counts = _largest_remainder_allocate(capacities, remaining)

    selected = []
    selected_buckets: Dict[str, int] = {}
    for key, bucket_records in bucket_items:
        take = min(len(bucket_records), base_counts[key] + extra_counts[key])
        if take <= 0:
            continue
        selected.extend(bucket_records[:take])
        selected_buckets[key] = take

    if len(selected) < max_samples:
        used_ids = {id(record) for record in selected}
        leftovers = [record for record in records if id(record) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    selected = selected[:max_samples]
    summary = {
        "mode": mode,
        "seed": seed,
        "available": available,
        "selected": len(selected),
        "task_balance": task_balance,
        "use_prompt_tokens": use_prompt_tokens,
        "selected_buckets": selected_buckets,
    }
    return selected, summary


def load_calibration_texts(
    path: Path,
    max_samples: int,
    text_field: str,
) -> Tuple[List[str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    records = list(_iter_jsonl(path))
    filtered_records, filter_summary = _filter_calibration_records_by_task(records)
    selected_records, summary = _select_calibration_records(filtered_records, max_samples)

    samples: List[str] = []
    for obj in selected_records:
        text = obj.get(text_field)
        if not text and text_field != "question":
            text = obj.get("question")
        if not text:
            continue
        samples.append(str(text))

    if not samples:
        raise RuntimeError(
            f"No calibration text found in {path}. Checked field='{text_field}' and fallback='question'."
        )
    summary = dict(summary)
    summary.update(filter_summary)
    summary["selected"] = len(samples)
    return samples, summary


def _env_truthy(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_env(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default or [])

    seen = set()
    values: List[str] = []
    for item in raw.split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _normalize_task_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _filter_calibration_records_by_task(records: List[dict]) -> Tuple[List[dict], dict]:
    include_tasks = _parse_csv_env("SOAR_GPTQ_CALIBRATION_TASK_INCLUDE")
    normalized_include = [_normalize_task_name(task) for task in include_tasks if task.strip()]

    if not normalized_include:
        return records, {
            "task_include": [],
            "records_before_task_filter": len(records),
            "records_after_task_filter": len(records),
            "task_filter_applied": False,
        }

    include_set = set(normalized_include)
    filtered = [
        record
        for record in records
        if _normalize_task_name(record.get("task")) in include_set
    ]
    if not filtered:
        raise RuntimeError(
            "Calibration task filter removed all records. "
            f"SOAR_GPTQ_CALIBRATION_TASK_INCLUDE={normalized_include}"
        )

    return filtered, {
        "task_include": normalized_include,
        "records_before_task_filter": len(records),
        "records_after_task_filter": len(filtered),
        "task_filter_applied": True,
    }


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    args: List[Any],
    kwargs: dict,
    optional_keys: List[str],
) -> Any:
    filtered = dict(kwargs)
    try:
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in sig.parameters.values()
        )
        if not has_var_keyword:
            for key in list(filtered.keys()):
                if key not in accepted:
                    filtered.pop(key, None)
    except Exception:
        pass

    try:
        return func(*args, **filtered)
    except TypeError:
        retry_kwargs = dict(filtered)
        for key in optional_keys:
            if key in retry_kwargs:
                retry_kwargs.pop(key, None)
                try:
                    return func(*args, **retry_kwargs)
                except TypeError:
                    continue
        raise


def _include_value_for_attr(attr_name: str, modules: List[str]) -> Any:
    if attr_name in {"inside_layer_modules", "modules_in_block_to_quantize"}:
        return [modules]
    return modules


def _resolve_sparse_layer_ids(model_config: dict) -> List[int]:
    layer_ids = _parse_optional_int_list_env("SOAR_GPTQ_SPARSE_LAYER_IDS")
    if layer_ids is not None:
        return layer_ids

    mixer_types = model_config.get("mixer_types")
    if not isinstance(mixer_types, list):
        return []

    sparse_ids = [
        index
        for index, mixer_type in enumerate(mixer_types)
        if isinstance(mixer_type, str)
        and mixer_type.lower() in {"minicpm4", "minicpm", "standard", "attention", "attn"}
    ]
    return sparse_ids


def _build_dynamic_rules(
    include_modules: List[str],
    exclude_modules: List[str],
    model_config: Optional[dict] = None,
) -> dict:
    dynamic = {}

    for module in exclude_modules:
        escaped = re.escape(module)
        dynamic[rf"-:.*{escaped}.*"] = {}

    mixed_precision_preset = (
        os.environ.get("SOAR_GPTQ_MIXED_PRECISION_PRESET", "o_proj_w8").strip().lower()
    )
    if mixed_precision_preset in {"", "0", "off", "none"}:
        return dynamic

    enable_o_proj_w8 = mixed_precision_preset in {
        "o_proj_w8",
        "sparse_qkv_w8_o_proj_w8",
    }
    enable_sparse_qkv_w8 = mixed_precision_preset in {
        "sparse_qkv_w8",
        "sparse_qkv_w8_o_proj_w8",
    }

    if not enable_o_proj_w8 and not enable_sparse_qkv_w8:
        raise ValueError(
            "Unsupported SOAR_GPTQ_MIXED_PRECISION_PRESET: "
            f"{mixed_precision_preset}. Supported values: o_proj_w8, sparse_qkv_w8, sparse_qkv_w8_o_proj_w8, off"
        )

    if enable_o_proj_w8:
        target_module = "self_attn.o_proj"
        if target_module not in exclude_modules and (
            not include_modules or target_module in include_modules
        ):
            dynamic[rf"+:.*{re.escape(target_module)}.*"] = {
                "bits": _parse_int_env("SOAR_GPTQ_O_PROJ_BITS", 8),
                "group_size": _parse_int_env("SOAR_GPTQ_O_PROJ_GROUP_SIZE", 128),
            }

    if enable_sparse_qkv_w8:
        if model_config is None:
            raise ValueError(
                "SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8 requires model_config"
            )

        sparse_layer_ids = _resolve_sparse_layer_ids(model_config)
        if not sparse_layer_ids:
            raise ValueError(
                "SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8 could not resolve sparse layers. "
                "Set SOAR_GPTQ_SPARSE_LAYER_IDS explicitly or verify config.mixer_types."
            )

        target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        enabled_modules = [
            module
            for module in target_modules
            if module not in exclude_modules and (not include_modules or module in include_modules)
        ]
        if not enabled_modules:
            return dynamic

        bits = _parse_int_env("SOAR_GPTQ_SPARSE_QKV_BITS", 8)
        group_size = _parse_int_env("SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE", 128)
        for layer_id in sparse_layer_ids:
            for module in enabled_modules:
                dynamic[rf"+:.*layers\.{layer_id}\.{re.escape(module)}.*"] = {
                    "bits": bits,
                    "group_size": group_size,
                }
            # GPTQ preprocess sees unfused q/k/v module names, while SGLang runtime
            # instantiates the fused qkv_proj parameter. Emit both names so the saved
            # quantize config keeps preprocess-time and runtime-time packing aligned.
            dynamic[rf"+:.*layers\.{layer_id}\.self_attn\.qkv_proj.*"] = {
                "bits": bits,
                "group_size": group_size,
            }

    return dynamic


def _is_module_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    patterns = [
        "layer module item",
        "not found in model",
        "module mismatch",
        "incompatible module",
    ]
    return any(pattern in text for pattern in patterns)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _rope_debug_snapshot(config: dict) -> dict:
    return {
        "rope_scaling_present": "rope_scaling" in config,
        "rope_type_present": "rope_type" in config,
        "rope_scaling": config.get("rope_scaling"),
        "rope_type": config.get("rope_type"),
    }


def _format_rope_debug(prefix: str, snapshot: dict) -> str:
    return (
        f"[preprocess][rope-debug] {prefix} "
        f"rope_scaling_present={snapshot['rope_scaling_present']} "
        f"rope_type_present={snapshot['rope_type_present']} "
        f"rope_scaling={json.dumps(snapshot['rope_scaling'], ensure_ascii=False, sort_keys=True)} "
        f"rope_type={json.dumps(snapshot['rope_type'], ensure_ascii=False)}"
    )


def _config_attr_snapshot(config: Any) -> dict:
    return {
        "model_type": getattr(config, "model_type", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "rope_parameters": getattr(config, "rope_parameters", None),
        "rope_type": getattr(config, "rope_type", None),
        "rope_theta": getattr(config, "rope_theta", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def _format_config_attr_debug(prefix: str, snapshot: dict) -> str:
    return (
        f"[preprocess][rope-debug] {prefix} "
        f"model_type={json.dumps(snapshot['model_type'], ensure_ascii=False)} "
        f"rope_scaling={json.dumps(snapshot['rope_scaling'], ensure_ascii=False, sort_keys=True)} "
        f"rope_parameters={json.dumps(snapshot['rope_parameters'], ensure_ascii=False, sort_keys=True)} "
        f"rope_type={json.dumps(snapshot['rope_type'], ensure_ascii=False)} "
        f"rope_theta={json.dumps(snapshot['rope_theta'], ensure_ascii=False)} "
        f"max_position_embeddings={json.dumps(snapshot['max_position_embeddings'], ensure_ascii=False)}"
    )


def _clear_minicpm_default_rope_state(config: Any) -> List[str]:
    changes: List[str] = []
    if getattr(config, "model_type", None) != "minicpm_sala":
        return changes

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if scaling_type == "default":
            config.rope_scaling = None
            changes.append("cleared in-memory rope_scaling default marker for MiniCPM-SALA")

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_type") == "default":
        config.rope_parameters = None
        changes.append("cleared in-memory rope_parameters default marker for MiniCPM-SALA")

    if getattr(config, "rope_type", None) == "default":
        config.rope_type = None
        changes.append("cleared in-memory top-level rope_type default marker for MiniCPM-SALA")

    return changes


def _print_dependency_versions(prefix: str) -> None:
    import gptqmodel
    import transformers

    print(
        f"[preprocess] {prefix} "
        f"python={json.dumps(os.sys.version.split()[0])} "
        f"gptqmodel={json.dumps(getattr(gptqmodel, '__version__', 'unknown'))} "
        f"transformers={json.dumps(getattr(transformers, '__version__', 'unknown'))}"
    )


def _log_minicpm_in_memory_config(
    config: Any,
    pre_label: str,
    post_label: str,
    middle_label: Optional[str] = None,
    middle_snapshot: Optional[dict] = None,
) -> List[str]:
    pre_snapshot = _config_attr_snapshot(config)
    changes = _clear_minicpm_default_rope_state(config)
    patched_snapshot = _config_attr_snapshot(config)

    if getattr(config, "model_type", None) != "minicpm_sala":
        return changes

    debug_in_memory_config = _env_truthy("SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG", default=True)
    if not (debug_in_memory_config or changes):
        return changes

    print(_format_config_attr_debug(pre_label, pre_snapshot))
    if middle_label is not None and middle_snapshot is not None:
        print(_format_config_attr_debug(middle_label, middle_snapshot))
    if changes:
        print(
            "[preprocess][rope-debug] in_memory_patch "
            f"changes={json.dumps(changes, ensure_ascii=False)}"
        )
    print(_format_config_attr_debug(post_label, patched_snapshot))
    return changes


def _install_gptqmodel_minicpm_rope_patch() -> None:
    import transformers
    from gptqmodel.utils import hf as gptq_hf
    from transformers import modeling_utils as modeling_utils

    _print_dependency_versions("GPTQ dependency versions")

    original_normalize = getattr(gptq_hf, "normalize_hf_config_compat", None)
    original_build_shell_model = getattr(gptq_hf, "build_shell_model", None)
    original_from_pretrained = modeling_utils.PreTrainedModel.from_pretrained.__func__
    original_from_config = modeling_utils.PreTrainedModel._from_config.__func__
    installed_hooks: List[str] = []

    if original_normalize is not None:
        if not getattr(original_normalize, "_soar_minicpm_patch_installed", False):
            def wrapped_normalize_hf_config_compat(config: Any, *args: Any, **kwargs: Any) -> None:
                original_normalize(config, *args, **kwargs)
                original_post_snapshot = _config_attr_snapshot(config)
                _log_minicpm_in_memory_config(
                    config,
                    pre_label="in_memory_config pre_normalize",
                    middle_label="in_memory_config post_normalize_pre_patch",
                    middle_snapshot=original_post_snapshot,
                    post_label="in_memory_config post_patch",
                )
                return None

            wrapped_normalize_hf_config_compat._soar_minicpm_patch_installed = True
            gptq_hf.normalize_hf_config_compat = wrapped_normalize_hf_config_compat
            installed_hooks.append("normalize_hf_config_compat")

    if original_build_shell_model is not None:
        if not getattr(original_build_shell_model, "_soar_minicpm_patch_installed", False):
            def wrapped_build_shell_model(*args: Any, **kwargs: Any) -> Any:
                config = kwargs.get("config")
                if config is None and len(args) >= 2:
                    config = args[1]
                if config is not None:
                    _log_minicpm_in_memory_config(
                        config,
                        pre_label="in_memory_config build_shell_model_entry_pre_patch",
                        post_label="in_memory_config build_shell_model_entry_post_patch",
                    )
                return original_build_shell_model(*args, **kwargs)

            wrapped_build_shell_model._soar_minicpm_patch_installed = True
            gptq_hf.build_shell_model = wrapped_build_shell_model
            installed_hooks.append("build_shell_model")

    if not getattr(original_from_pretrained, "_soar_minicpm_patch_installed", False):
        def wrapped_from_pretrained(cls: Any, pretrained_model_name_or_path: Any, *model_args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config")
            if getattr(config, "model_type", None) == "minicpm_sala":
                _print_dependency_versions("GPTQ dependency versions pre_transformers_from_pretrained")
                _log_minicpm_in_memory_config(
                    config,
                    pre_label="transformers_from_pretrained_entry_pre_patch",
                    post_label="transformers_from_pretrained_entry_post_patch",
                )
            return original_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

        wrapped_from_pretrained._soar_minicpm_patch_installed = True
        modeling_utils.PreTrainedModel.from_pretrained = classmethod(wrapped_from_pretrained)
        installed_hooks.append("transformers.from_pretrained")

    if not getattr(original_from_config, "_soar_minicpm_patch_installed", False):
        def wrapped_from_config(cls: Any, config: Any, **kwargs: Any) -> Any:
            if getattr(config, "model_type", None) == "minicpm_sala":
                _print_dependency_versions("GPTQ dependency versions pre_transformers_from_config")
                _log_minicpm_in_memory_config(
                    config,
                    pre_label="transformers_from_config_entry_pre_patch",
                    post_label="transformers_from_config_entry_post_patch",
                )
            return original_from_config(cls, config, **kwargs)

        wrapped_from_config._soar_minicpm_patch_installed = True
        modeling_utils.PreTrainedModel._from_config = classmethod(wrapped_from_config)
        installed_hooks.append("transformers._from_config")

    if installed_hooks:
        print(
            "[preprocess] Installed MiniCPM-SALA load compatibility hooks "
            f"hooks={json.dumps(installed_hooks)}"
        )
        return

    print(
        "[preprocess] MiniCPM-SALA load compatibility hooks already installed"
    )


def _sanitize_model_config_for_gptq(config: dict) -> Tuple[dict, List[str], dict, dict]:
    sanitized = dict(config)
    changes: List[str] = []
    raw_snapshot = _rope_debug_snapshot(config)
    force_null_rope_scaling = _env_truthy(
        "SOAR_GPTQ_FORCE_NULL_ROPE_SCALING", default=False
    )

    rope_scaling = sanitized.get("rope_scaling")
    has_default_rope_marker = isinstance(rope_scaling, dict) and (
        rope_scaling.get("type") == "default"
        or rope_scaling.get("rope_type") == "default"
    )
    if has_default_rope_marker:
        if force_null_rope_scaling:
            sanitized["rope_scaling"] = None
            changes.append(
                "set rope_scaling=null because default rope scaling markers are unsupported by current MiniCPM-SALA GPTQ shell load path"
            )
        else:
            sanitized.pop("rope_scaling", None)
            changes.append(
                "removed rope_scaling because default rope scaling markers are unsupported by current MiniCPM-SALA GPTQ shell load path"
            )

    if force_null_rope_scaling and not has_default_rope_marker:
        sanitized["rope_scaling"] = None
        changes.append(
            "set rope_scaling=null due to SOAR_GPTQ_FORCE_NULL_ROPE_SCALING workaround"
        )

    if "rope_type" in sanitized:
        sanitized.pop("rope_type", None)
        changes.append("removed top-level rope_type from GPTQ temp config")

    sanitized_snapshot = _rope_debug_snapshot(sanitized)
    return sanitized, changes, raw_snapshot, sanitized_snapshot


def _prepare_gptq_load_source(
    src: Path,
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory[str]], dict]:
    config_path = src / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sanitized_config, changes, raw_snapshot, sanitized_snapshot = _sanitize_model_config_for_gptq(config)
    debug_info = {
        "raw": raw_snapshot,
        "sanitized": sanitized_snapshot,
        "changes": changes,
        "used_temp_source": bool(changes),
        "load_src": str(src),
    }

    print(_format_rope_debug("raw_config", raw_snapshot))
    if not changes:
        print(_format_rope_debug("sanitized_config source=raw", sanitized_snapshot))
        return src, None, debug_info

    temp_dir = tempfile.TemporaryDirectory(prefix="soar_gptq_model_")
    temp_root = Path(temp_dir.name)
    debug_info["load_src"] = str(temp_root)

    for entry in sorted(src.iterdir()):
        if entry.name.startswith("."):
            continue
        target = temp_root / entry.name
        if entry.name == "config.json":
            target.write_text(
                json.dumps(sanitized_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            continue
        _safe_symlink_or_copy(entry, target)

    print(
        "[preprocess] GPTQ load-source config sanitization "
        f"source={src} temp_source={temp_root} changes={changes}"
    )
    print(_format_rope_debug(f"sanitized_config source={temp_root}", sanitized_snapshot))
    return temp_root, temp_dir, debug_info


def _restore_gptq_source_metadata(model: Any, src: Path) -> None:
    src_str = str(src)

    if hasattr(model, "model_local_path"):
        try:
            model.model_local_path = src_str
        except Exception:
            pass

    if hasattr(model, "model_name_or_path"):
        try:
            model.model_name_or_path = src_str
        except Exception:
            pass


def run_gptq_quantization(
    src: Path,
    dst: Path,
    bits: int,
    group_size: int,
    calibration_file: Path,
    calibration_samples: int,
    calibration_field: str,
    batch_size: int,
) -> None:
    # Import lazily so copy mode does not require gptq dependencies.
    from gptqmodel import GPTQModel, QuantizeConfig

    from gptqmodel_minicpm_sala import register_minicpm_sala_gptq_model

    register_minicpm_sala_gptq_model()
    _install_gptqmodel_minicpm_rope_patch()

    calibration_texts, calibration_summary = load_calibration_texts(
        calibration_file,
        max_samples=calibration_samples,
        text_field=calibration_field,
    )
    model_config = json.loads((src / "config.json").read_text(encoding="utf-8"))

    default_include = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    default_exclude = ["self_attn.o_gate", "self_attn.z_proj"]

    layer_aware = _env_truthy("SOAR_GPTQ_LAYER_AWARE", default=True)
    include_modules = _parse_csv_env("SOAR_GPTQ_INCLUDE_MODULES", default=default_include)
    exclude_modules = _parse_csv_env("SOAR_GPTQ_EXCLUDE_MODULES", default=default_exclude)
    exclude_set = set(exclude_modules)
    include_modules = [module for module in include_modules if module not in exclude_set]

    dynamic_rules = (
        _build_dynamic_rules(include_modules, exclude_modules, model_config=model_config)
        if layer_aware
        else None
    )
    quant_config = QuantizeConfig(bits=bits, group_size=group_size, dynamic=dynamic_rules)

    trust_remote_code = _env_truthy("SOAR_TRUST_REMOTE_CODE", default=True)
    attn_impl = os.environ.get("SOAR_GPTQ_ATTN_IMPL", "flash_attention_2").strip()
    print(
        "[preprocess] GPTQ start "
        f"bits={bits} group_size={group_size} "
        f"calibration_samples={len(calibration_texts)} batch_size={batch_size} "
        f"calibration_sampling={json.dumps(calibration_summary, sort_keys=True)} "
        f"trust_remote_code={trust_remote_code} attn_impl={attn_impl} "
        f"layer_aware={layer_aware} include={include_modules} exclude={exclude_modules} "
        f"dynamic_rules={dynamic_rules}"
    )
    print("[preprocess] GPTQ custom model support enabled for model_type=minicpm_sala")

    load_src, load_src_tmpdir, load_debug_info = _prepare_gptq_load_source(src)

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_impl,
    }
    try:
        try:
            model = _call_with_supported_kwargs(
                GPTQModel.load,
                [str(load_src), quant_config],
                load_kwargs,
                optional_keys=["attn_implementation"],
            )
        except Exception as exc:
            print(
                "[preprocess][rope-debug] load_failed "
                f"load_src={load_debug_info['load_src']} "
                f"used_temp_source={load_debug_info['used_temp_source']} "
                f"changes={json.dumps(load_debug_info['changes'], ensure_ascii=False)} "
                f"raw_rope_scaling={json.dumps(load_debug_info['raw']['rope_scaling'], ensure_ascii=False, sort_keys=True)} "
                f"raw_rope_type={json.dumps(load_debug_info['raw']['rope_type'], ensure_ascii=False)} "
                f"sanitized_rope_scaling={json.dumps(load_debug_info['sanitized']['rope_scaling'], ensure_ascii=False, sort_keys=True)} "
                f"sanitized_rope_type={json.dumps(load_debug_info['sanitized']['rope_type'], ensure_ascii=False)} "
                f"exc_type={type(exc).__name__} exc={exc}"
            )
            raise
        _restore_gptq_source_metadata(model, src)
        if layer_aware:
            try:
                print(
                    "[preprocess] GPTQ resolved modules "
                    f"simple_layer_modules={model.simple_layer_modules(model.model.config, model.quantize_config)}"
                )
            except Exception as debug_exc:
                print(f"[preprocess] GPTQ module debug unavailable: {debug_exc}")

        try:
            _call_with_supported_kwargs(
                model.quantize,
                [calibration_texts],
                {"batch_size": batch_size},
                optional_keys=["batch_size"],
            )
        except Exception as exc:
            if not (layer_aware and _is_module_mismatch_error(exc)):
                raise

            retry_include = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
            retry_exclude = ["self_attn.o_gate", "self_attn.z_proj"]
            print(
                "[preprocess] GPTQ retry after module mismatch "
                f"error={exc} retry_include={retry_include} retry_exclude={retry_exclude}"
            )

            retry_dynamic = _build_dynamic_rules(
                retry_include,
                retry_exclude,
                model_config=model_config,
            )
            retry_config = QuantizeConfig(
                bits=bits,
                group_size=group_size,
                dynamic=retry_dynamic,
            )
            print(
                "[preprocess] GPTQ retry dynamic "
                f"dynamic_rules={retry_dynamic}"
            )

            retry_model = _call_with_supported_kwargs(
                GPTQModel.load,
                [str(load_src), retry_config],
                load_kwargs,
                optional_keys=["attn_implementation"],
            )
            _restore_gptq_source_metadata(retry_model, src)
            try:
                print(
                    "[preprocess] GPTQ retry resolved modules "
                    f"simple_layer_modules={retry_model.simple_layer_modules(retry_model.model.config, retry_model.quantize_config)}"
                )
            except Exception as debug_exc:
                print(f"[preprocess] GPTQ retry module debug unavailable: {debug_exc}")
            _call_with_supported_kwargs(
                retry_model.quantize,
                [calibration_texts],
                {"batch_size": batch_size},
                optional_keys=["batch_size"],
            )
            model = retry_model

        dst.mkdir(parents=True, exist_ok=True)
        model.save(str(dst))
    finally:
        if load_src_tmpdir is not None:
            load_src_tmpdir.cleanup()

    if not (dst / "quantize_config.json").exists():
        raise RuntimeError(
            "GPTQ output missing quantize_config.json, which is required by SGLang loader."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["copy", "gptq"],
        default=None,
        help="Preprocess mode. If unset, reads SOAR_QUANT_MODE (default: copy).",
    )
    parser.add_argument(
        "--calibration-file",
        default=os.environ.get(
            "SOAR_GPTQ_CALIBRATION_FILE",
            str(Path(__file__).resolve().parent / "perf_public_set.jsonl"),
        ),
        help="JSONL calibration file path for GPTQ mode.",
    )
    parser.add_argument(
        "--calibration-field",
        default=os.environ.get("SOAR_GPTQ_CALIBRATION_FIELD", "question"),
        help="Text field name used from calibration JSONL records.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_CALIBRATION_SAMPLES", "32")),
        help="Maximum number of calibration samples.",
    )
    parser.add_argument(
        "--gptq-bits",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_BITS", "4")),
        help="GPTQ quantization bits.",
    )
    parser.add_argument(
        "--gptq-group-size",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_GROUP_SIZE", "128")),
        help="GPTQ group size.",
    )
    parser.add_argument(
        "--gptq-batch-size",
        type=int,
        default=int(os.environ.get("SOAR_GPTQ_BATCH_SIZE", "2")),
        help="Calibration batch size for GPTQModel.quantize.",
    )
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")

    mode = args.mode or os.environ.get("SOAR_QUANT_MODE", "copy")
    mode = mode.strip().lower()
    if mode not in {"copy", "gptq"}:
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    if mode == "gptq":
        gptq_preflight(src)
        if not args.calibration_file:
            raise RuntimeError(
                "GPTQ mode requires --calibration-file (or SOAR_GPTQ_CALIBRATION_FILE)."
            )
        run_gptq_quantization(
            src=src,
            dst=dst,
            bits=args.gptq_bits,
            group_size=args.gptq_group_size,
            calibration_file=Path(args.calibration_file).resolve(),
            calibration_samples=args.calibration_samples,
            calibration_field=args.calibration_field,
            batch_size=args.gptq_batch_size,
        )
        print(f"[preprocess] mode={mode} done - quantized model saved to {dst}")
        return

    count = copy_model(src, dst)

    print(f"[preprocess] mode={mode} done - copied {count} files from {src} to {dst}")


if __name__ == "__main__":
    main()
