#!/usr/bin/env python3
"""SOAR CHANGE_0165 — Pre-flight diff tool.

Loads two pickles produced by `_preflight.dump_state` (one from NGRAM run, one
from MEDUSA run) and prints a per-field comparison table.

Usage:
    python3 preflight_diff.py --ngram /tmp/dump_ngram.pkl \
        --medusa /tmp/dump_medusa.pkl [--phase pre_verify|post_forward|post_verify|all]

For each phase, the tool walks the FIRST record from each side (i.e. first
verify step in the run) and reports per-field:
    - shape
    - dtype
    - first-N values (or scalar repr)
    - "differs": True/False with summary

Output is plain text intended for inclusion in CHANGE_0165 §5.N.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from typing import Any, Dict, List

import torch


PHASE_ORDER = ["pre_prepare_for_verify", "pre_verify", "post_forward", "post_verify"]


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        recs = pickle.load(f)
    if not isinstance(recs, list):
        raise ValueError(f"{path}: expected list of records, got {type(recs)}")
    return recs


def _first_by_phase(recs: List[Dict[str, Any]], phase: str) -> Dict[str, Any] | None:
    for r in recs:
        if r.get("phase") == phase:
            return r
    return None


def _summarize(v: Any, head: int = 8) -> str:
    if isinstance(v, torch.Tensor):
        flat = v.flatten()
        n = flat.numel()
        head_vals = flat[: min(head, n)].tolist()
        tail_note = "" if n <= head else f", ...({n - head} more)"
        return f"tensor{tuple(v.shape)} dtype={v.dtype} values={head_vals}{tail_note}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)}) head={list(v)[:head]}"
    return repr(v)


def _equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        # tensors of different dtypes / shapes are also "different"
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            pass
        else:
            return False
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        try:
            return torch.equal(a, b)
        except Exception:
            return False
    return a == b


def _diff_summary(a: Any, b: Any) -> str:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f"SHAPE: {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.dtype != b.dtype:
            return f"DTYPE: {a.dtype} vs {b.dtype}"
        try:
            diff_mask = (a != b)
            n_diff = int(diff_mask.sum().item())
            n_total = a.numel()
            if n_diff == 0:
                return "EQUAL"
            # show first 4 diff positions
            idxs = diff_mask.nonzero(as_tuple=False)[:4].tolist()
            return f"VALUES: {n_diff}/{n_total} differ; first positions={idxs}"
        except Exception as e:  # noqa
            return f"COMPARE_ERR: {e}"
    return f"VALUES: a={a!r} vs b={b!r}"


def _print_table(ngram_rec, medusa_rec) -> int:
    """Print per-field diff; return number of differing fields."""
    if ngram_rec is None and medusa_rec is None:
        print("  (no records on either side)")
        return 0
    if ngram_rec is None:
        print(f"  WARN: ngram side missing this phase. medusa keys={list((medusa_rec or {}).get('fields', {}).keys())}")
        return -1
    if medusa_rec is None:
        print(f"  WARN: medusa side missing this phase. ngram keys={list((ngram_rec or {}).get('fields', {}).keys())}")
        return -1

    a_fields = ngram_rec["fields"]
    b_fields = medusa_rec["fields"]
    all_keys = sorted(set(a_fields) | set(b_fields))

    n_diff = 0
    print(f"  {'field':<32} {'ngram':<6} {'medusa':<6} {'status':<60}")
    print(f"  {'-'*32} {'-'*6} {'-'*6} {'-'*60}")
    for k in all_keys:
        a = a_fields.get(k, "<MISSING>")
        b = b_fields.get(k, "<MISSING>")
        if a == "<MISSING>" or b == "<MISSING>":
            n_diff += 1
            print(f"  {k:<32} {'+' if a!='<MISSING>' else '-':<6} {'+' if b!='<MISSING>' else '-':<6} ONE_SIDE_MISSING")
            continue
        same = _equal(a, b)
        status = "EQUAL" if same else _diff_summary(a, b)
        if not same:
            n_diff += 1
        marker_a = "+"
        marker_b = "+"
        print(f"  {k:<32} {marker_a:<6} {marker_b:<6} {status:<60}")

    print("")
    print("  Details (per-field summaries):")
    for k in all_keys:
        a = a_fields.get(k, "<MISSING>")
        b = b_fields.get(k, "<MISSING>")
        print(f"    [{k}]")
        print(f"      ngram : {_summarize(a)}")
        print(f"      medusa: {_summarize(b)}")
    return n_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram", required=True, help="pickle from NGRAM run")
    ap.add_argument("--medusa", required=True, help="pickle from MEDUSA run")
    ap.add_argument(
        "--phase", default="all",
        choices=["all", "pre_verify", "post_forward", "post_verify"],
    )
    args = ap.parse_args()

    ngram_recs = _load(args.ngram)
    medusa_recs = _load(args.medusa)
    print(f"loaded ngram  : {len(ngram_recs)} records  ({args.ngram})")
    print(f"loaded medusa : {len(medusa_recs)} records  ({args.medusa})")
    print("")

    phases = PHASE_ORDER if args.phase == "all" else [args.phase]
    total_diff = 0
    for p in phases:
        print(f"=== PHASE: {p} ===")
        a = _first_by_phase(ngram_recs, p)
        b = _first_by_phase(medusa_recs, p)
        n = _print_table(a, b)
        if n > 0:
            total_diff += n
        print("")

    print(f"TOTAL differing fields across phases: {total_diff}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
