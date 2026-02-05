#!/usr/bin/env python3
"""
Generate verification dataset for confidence calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

# Allow running as a script without requiring `PYTHONPATH=.`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verification_loop import VerificationLoop


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


def _extract_rules(step_sequence: List[Dict[str, Any]]) -> List[str]:
    rules = []
    for step in step_sequence:
        rule = step.get("rule")
        if rule:
            rules.append(str(rule))
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verification dataset from Log Galaxy JSONL.")
    parser.add_argument(
        "--input",
        default="data/log_galaxy_neural_v5.jsonl",
        help="Log Galaxy JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/verification_train_v1.jsonl",
        help="Verification dataset JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of traces (0 = all).")
    parser.add_argument("--quiet", action="store_true", help="Silence verification solver logs.")
    args = parser.parse_args()

    verifier = VerificationLoop(quiet=bool(args.quiet))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for entry in _iter_jsonl(args.input):
            if args.limit and total >= args.limit:
                break
            total += 1
            problem_text = str(entry.get("problem_text") or "").strip()
            if not problem_text:
                continue
            step_sequence = entry.get("step_sequence") or []
            predicted_rules = _extract_rules(step_sequence)
            if not predicted_rules:
                continue
            correctness = verifier.verify_rule_sequence(problem_text, predicted_rules)
            payload = {
                "trace_id": entry.get("trace_id"),
                "problem_text": problem_text,
                "predicted_rules": predicted_rules,
                "correctness_labels": correctness,
                "success": entry.get("success"),
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1

    print(f"[VerificationDataset] Read: {total}")
    print(f"[VerificationDataset] Written: {written}")
    print(f"[VerificationDataset] Output: {output}")


if __name__ == "__main__":
    main()
