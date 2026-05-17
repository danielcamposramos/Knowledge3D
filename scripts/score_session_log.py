#!/usr/bin/env python3
"""Post-hoc scorer for natural-query session logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _truth_lookup(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else list(payload.get("rows") or payload.get("results") or [])
    lookup: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id") or row.get("task_id") or row.get("item_id") or "").strip()
        if not row_id:
            continue
        lookup[row_id] = row.get("expected") if "expected" in row else row.get("correct_answer", row.get("answer"))
    return lookup


def _normalise_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value).strip()
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_log", help="JSONL emitted by the natural-query adapter/log path")
    parser.add_argument("--truth", default=None, help="Optional JSON/JSONL reference file keyed by id/task_id/item_id")
    args = parser.parse_args()

    session_rows = _read_jsonl(Path(args.session_log))
    truth = _truth_lookup(Path(args.truth)) if args.truth else {}

    by_suite: dict[str, dict[str, Any]] = {}
    by_meaning: dict[str, dict[str, Any]] = {}
    scored_rows: list[dict[str, Any]] = []
    for row in session_rows:
        row_id = str(row.get("item_id") or row.get("task_id") or row.get("id") or "").strip()
        expected = row.get("expected")
        if expected is None and row_id in truth:
            expected = truth[row_id]
        predicted = row.get("normalized_answer", row.get("predicted", row.get("answer")))
        correct = _normalise_answer(predicted) == _normalise_answer(expected) if expected is not None else bool(row.get("correct", False))
        suite = str(row.get("suite") or "unknown")
        meaning = str(row.get("route_family") or "unknown")
        by_suite.setdefault(suite, {"total": 0, "correct": 0})
        by_suite[suite]["total"] += 1
        by_suite[suite]["correct"] += int(correct)
        by_meaning.setdefault(meaning, {"total": 0, "correct": 0})
        by_meaning[meaning]["total"] += 1
        by_meaning[meaning]["correct"] += int(correct)
        scored_rows.append(
            {
                "item_id": row_id,
                "suite": suite,
                "route_family": meaning,
                "expected": expected,
                "predicted": predicted,
                "correct": bool(correct),
            }
        )

    summary = {
        "session_log": str(args.session_log),
        "truth_file": str(args.truth) if args.truth else None,
        "total": len(scored_rows),
        "correct": sum(1 for row in scored_rows if bool(row["correct"])),
        "accuracy": (
            sum(1 for row in scored_rows if bool(row["correct"])) / len(scored_rows)
            if scored_rows
            else 0.0
        ),
        "per_suite": {
            key: {
                **value,
                "accuracy": (value["correct"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_suite.items())
        },
        "per_meaning_class": {
            key: {
                **value,
                "accuracy": (value["correct"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_meaning.items())
        },
        "rows": scored_rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
