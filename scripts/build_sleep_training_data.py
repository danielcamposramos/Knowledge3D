#!/usr/bin/env python3
"""
Bootstrap training data for the Sleep Keeper (keep/discard/uncertain).

Heuristic labels:
  - keep: success + neural + zero mismatches
  - discard: failed or policy mismatches (configurable)
  - uncertain: success but non-neural or mixed traces
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LABELS = {
    "discard": 0,
    "keep": 1,
    "uncertain": 2,
}


def _iter_log_entries(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _is_noble_failure(
    entry: Dict[str, Any],
    *,
    min_steps: int,
    max_mismatches: int,
) -> bool:
    if bool(entry.get("success", False)):
        return False
    meta = entry.get("metadata") or {}
    policy_mode = str(meta.get("policy_mode", entry.get("policy_mode", "heuristic")))
    if policy_mode not in {"neural", "mixed"}:
        return False
    mismatches = int(meta.get("policy_mismatches", entry.get("policy_mismatches", 0)))
    steps = entry.get("step_sequence") or []
    if len(steps) < int(min_steps):
        return False
    if mismatches > int(max_mismatches):
        return False
    return True


def _label_entry(
    entry: Dict[str, Any],
    *,
    strict_mismatch: bool,
    min_steps: int,
    max_mismatches: int,
) -> Tuple[str, bool]:
    success = bool(entry.get("success", False))
    meta = entry.get("metadata") or {}
    policy_mode = str(meta.get("policy_mode", entry.get("policy_mode", "heuristic")))
    mismatches = int(meta.get("policy_mismatches", entry.get("policy_mismatches", 0)))

    if not success:
        if _is_noble_failure(entry, min_steps=min_steps, max_mismatches=max_mismatches):
            return "keep", True
        return "discard", False
    if strict_mismatch and mismatches > 0:
        return "discard", False
    if policy_mode == "neural" and mismatches == 0:
        return "keep", False
    if policy_mode == "mixed":
        return "uncertain", False
    if policy_mode == "heuristic":
        return "uncertain", False
    return "uncertain", False


def _entry_to_sample(entry: Dict[str, Any], label: str, *, negative_wisdom: bool) -> Dict[str, Any]:
    return {
        "text": entry.get("problem_text", ""),
        "label": LABELS[label],
        "label_name": label,
        "negative_wisdom": bool(negative_wisdom),
        "trace_id": entry.get("trace_id", ""),
        "success": bool(entry.get("success", False)),
        "policy_mode": (entry.get("metadata") or {}).get("policy_mode", entry.get("policy_mode", "")),
        "policy_mismatches": (entry.get("metadata") or {}).get("policy_mismatches", entry.get("policy_mismatches", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sleep Keeper training data.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Log Galaxy JSONL files to bootstrap from.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--dust-out",
        default=None,
        help="Optional JSONL path for discarded samples (dust).",
    )
    parser.add_argument(
        "--strict-mismatch",
        action="store_true",
        help="Treat any policy mismatch as discard.",
    )
    parser.add_argument(
        "--noble-min-steps",
        type=int,
        default=3,
        help="Minimum steps to count a failed trace as a noble failure.",
    )
    parser.add_argument(
        "--noble-max-mismatches",
        type=int,
        default=1,
        help="Max policy mismatches allowed for noble failures.",
    )
    args = parser.parse_args()

    counts = {"keep": 0, "discard": 0, "uncertain": 0, "keep_negative": 0}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dust_handle = None
    if args.dust_out:
        dust_path = Path(args.dust_out)
        dust_path.parent.mkdir(parents=True, exist_ok=True)
        dust_handle = dust_path.open("w", encoding="utf-8")

    with output_path.open("w", encoding="utf-8") as out:
        for entry in _iter_log_entries(args.logs):
            label, negative_wisdom = _label_entry(
                entry,
                strict_mismatch=bool(args.strict_mismatch),
                min_steps=int(args.noble_min_steps),
                max_mismatches=int(args.noble_max_mismatches),
            )
            if label not in LABELS:
                continue
            sample = _entry_to_sample(entry, label, negative_wisdom=negative_wisdom)
            if not sample["text"]:
                continue
            out.write(json.dumps(sample, ensure_ascii=True) + "\n")
            counts[label] += 1
            if negative_wisdom:
                counts["keep_negative"] += 1
            if dust_handle and label == "discard":
                dust_handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

    if dust_handle:
        dust_handle.close()

    total = sum(counts.values())
    print("[SleepKeeper] Training samples:", total)
    for key in ("keep", "discard", "uncertain", "keep_negative"):
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()
