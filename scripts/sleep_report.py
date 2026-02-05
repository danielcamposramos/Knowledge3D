#!/usr/bin/env python3
"""
Summarize Sleep Keeper decisions for a Sleep Galaxy JSONL.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_entries(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    decision_scores: Dict[str, List[float]] = {}
    policy_modes: Counter[str] = Counter()

    for entry in entries:
        decision = str(entry.get("decision", "unknown")).lower()
        decision_counts[decision] += 1
        decision_scores.setdefault(decision, []).append(_safe_float(entry.get("decision_score", 0.0)))

        meta = entry.get("metadata") or {}
        mode = str(meta.get("policy_mode", "unknown"))
        policy_modes[mode] += 1

    summary = {
        "total": len(entries),
        "decision_counts": dict(decision_counts),
        "decision_avg_score": {k: _mean(v) for k, v in decision_scores.items()},
        "policy_modes": dict(policy_modes),
    }
    return summary


def _render(summary: Dict[str, Any], *, dust_count: int = 0) -> None:
    total = summary.get("total", 0)
    print("=== Sleep Keeper Report ===")
    print(f"Total entries: {total}")
    if dust_count:
        print(f"Dust entries: {dust_count}")

    decision_counts = summary.get("decision_counts", {})
    decision_avg = summary.get("decision_avg_score", {})
    print("\nDecision breakdown:")
    for decision, count in sorted(decision_counts.items(), key=lambda x: (-x[1], x[0])):
        pct = (count / total * 100.0) if total else 0.0
        avg = decision_avg.get(decision, 0.0)
        print(f"  {decision}: {count} ({pct:.1f}%) avg_score={avg:.3f}")

    policy_modes = summary.get("policy_modes", {})
    if policy_modes:
        print("\nPolicy modes:")
        for mode, count in sorted(policy_modes.items(), key=lambda x: (-x[1], x[0])):
            pct = (count / total * 100.0) if total else 0.0
            print(f"  {mode}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SleepGalaxy decisions.")
    parser.add_argument("--input", required=True, help="SleepGalaxy JSONL path.")
    parser.add_argument("--dust", default=None, help="Optional dust JSONL path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"SleepGalaxy JSONL not found: {input_path}")

    entries = list(_iter_entries(str(input_path)))
    summary = _summarize(entries)

    dust_count = 0
    if args.dust:
        dust_path = Path(args.dust)
        if dust_path.exists():
            dust_count = sum(1 for _ in _iter_entries(str(dust_path)))

    _render(summary, dust_count=dust_count)


if __name__ == "__main__":
    main()
