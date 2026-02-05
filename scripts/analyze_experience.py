#!/usr/bin/env python3
"""
Analyze Log Galaxy traces for autonomy and drift metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, Iterable, Tuple


def _iter_traces(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_policy_meta(trace: Dict[str, Any]) -> Tuple[int, int, int, str]:
    meta = trace.get("metadata") or {}
    steps = trace.get("step_sequence") or []
    
    honest_steps = 0
    hallucination_steps = 0
    heuristic_steps = 0
    
    for s in steps:
        status = s.get("status", "heuristic")
        if status == "honest":
            honest_steps += 1
        elif status == "hallucination":
            hallucination_steps += 1
        else:
            heuristic_steps += 1
            
    policy_mode = str(meta.get("policy_mode", trace.get("policy_mode", "heuristic")))
    return honest_steps, hallucination_steps, len(steps), policy_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Log Galaxy neural experience.")
    parser.add_argument("log_path", help="Path to log galaxy JSONL file.")
    args = parser.parse_args()

    totals = {
        "traces": 0,
        "success": 0,
        "honest_steps": 0,
        "hallucination_steps": 0,
        "total_steps": 0,
    }
    mode_counts: Counter[str] = Counter()

    for trace in _iter_traces(args.log_path):
        totals["traces"] += 1
        if trace.get("success"):
            totals["success"] += 1
        honest, hallucination, total_steps, policy_mode = _extract_policy_meta(trace)
        totals["honest_steps"] += honest
        totals["hallucination_steps"] += hallucination
        totals["total_steps"] += total_steps
        mode_counts[policy_mode] += 1

    if totals["traces"] == 0:
        print("No traces found.")
        return

    autonomy = totals["honest_steps"] / max(1, totals["total_steps"])
    policy_activity = totals["honest_steps"] + totals["hallucination_steps"]
    drift = totals["hallucination_steps"] / max(1, policy_activity)
    success_rate = totals["success"] / totals["traces"]

    print("=== Experience Analysis ===")
    print(f"Traces: {totals['traces']}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Autonomy (policy steps / total steps): {autonomy:.2%}")
    print(f"Drift (mismatches / policy steps): {drift:.2%}")
    if mode_counts:
        print("Policy modes:")
        for mode, count in mode_counts.most_common():
            print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
