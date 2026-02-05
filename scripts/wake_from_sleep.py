#!/usr/bin/env python3
"""
Wake from Sleep: convert SleepGalaxy decisions into training datasets.

Outputs:
  - Positive LogGalaxy JSONL (keep, not negative_wisdom)
  - Negative LogGalaxy JSONL (keep + negative_wisdom) for anti-patterns
  - Optional serialized .bin/.json for positives (NavigationDataset ready)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from knowledge3d.training.math_benchmarks.log_galaxy_serializer import serialize_log_galaxy


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_log_index(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for entry in _iter_jsonl(path):
            trace_id = str(entry.get("trace_id", ""))
            if not trace_id:
                continue
            if trace_id not in index:
                index[trace_id] = entry
    return index


def _write_jsonl(entries: List[Dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _sanitize_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    reserved = {"honest", "hallucination", "heuristic"}
    for step in steps:
        rule = str(step.get("rule", "")).strip().lower()
        if not rule or rule in reserved:
            continue
        clean_step = dict(step)
        clean_step.pop("status", None)
        cleaned.append(clean_step)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Wake datasets from SleepGalaxy.")
    parser.add_argument("--sleep", required=True, help="SleepGalaxy JSONL path.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="LogGalaxy JSONL files to resolve trace_id -> step_sequence.",
    )
    parser.add_argument("--positive-out", required=True, help="Output JSONL for positive traces.")
    parser.add_argument("--negative-out", required=True, help="Output JSONL for negative wisdom traces.")
    parser.add_argument(
        "--serialize",
        action="store_true",
        help="Serialize positive JSONL into .bin/.json using LogGalaxy serializer.",
    )
    args = parser.parse_args()

    log_index = _load_log_index(args.logs)

    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    missing = 0

    for entry in _iter_jsonl(args.sleep):
        decision = str(entry.get("decision", "")).lower()
        if decision != "keep":
            continue
        trace_id = str(entry.get("trace_id", ""))
        log_entry = log_index.get(trace_id)
        if not log_entry:
            missing += 1
            continue
        meta = dict(log_entry.get("metadata") or {})
        meta["sleep_decision"] = decision
        if entry.get("metadata", {}).get("negative_wisdom"):
            meta["negative_wisdom"] = True
        log_entry = dict(log_entry)
        log_entry["metadata"] = meta
        log_entry["step_sequence"] = _sanitize_steps(log_entry.get("step_sequence") or [])
        if meta.get("negative_wisdom"):
            negatives.append(log_entry)
        else:
            positives.append(log_entry)

    _write_jsonl(positives, args.positive_out)
    _write_jsonl(negatives, args.negative_out)

    print("[Wake] Positive traces:", len(positives))
    print("[Wake] Negative wisdom traces:", len(negatives))
    if missing:
        print(f"[Wake] Missing trace ids: {missing}")

    if args.serialize:
        prefix = args.positive_out.replace(".jsonl", "")
        serialize_log_galaxy(jsonl_path=args.positive_out, output_prefix=prefix)
        print(f"[Wake] Serialized: {prefix}.bin / {prefix}.json")


if __name__ == "__main__":
    main()
