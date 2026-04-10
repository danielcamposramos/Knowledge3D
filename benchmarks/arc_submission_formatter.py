"""ARC submission formatting helpers for ARC competition artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Grid = list[list[int]]
Submission = dict[str, list[dict[str, Grid | None]]]


def format_arc_submission(
    results: list[dict[str, Any]],
    *,
    attempt_slots: int = 2,
    duplicate_primary_attempt: bool = False,
) -> Submission:
    """Convert ARC benchmark rows into competition-style submission payloads."""
    if attempt_slots < 1:
        raise ValueError("attempt_slots must be >= 1")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row in results:
        task_id = str(row["task_id"])
        sample_index = int(row.get("sample_index", 0) or 0)
        grouped.setdefault(task_id, []).append((sample_index, row))

    submission: Submission = {}
    for task_id, sample_rows in grouped.items():
        entries: list[dict[str, Grid | None]] = []
        for _, row in sorted(sample_rows, key=lambda item: item[0]):
            predictions = _row_predictions(row)
            primary = predictions[0] if predictions else None
            secondary = predictions[1] if len(predictions) > 1 else None
            if primary is not None and not _is_grid(primary):
                raise ValueError(f"Task {task_id} has invalid primary grid: {primary!r}")
            if secondary is not None and not _is_grid(secondary):
                raise ValueError(f"Task {task_id} has invalid secondary grid: {secondary!r}")

            attempts: dict[str, Grid | None] = {"attempt_1": primary}
            fallback_grid = row.get("input_grid")
            if fallback_grid is not None and not _is_grid(fallback_grid):
                raise ValueError(f"Task {task_id} has invalid input grid fallback: {fallback_grid!r}")
            if attempts["attempt_1"] is None and _is_grid(fallback_grid):
                attempts["attempt_1"] = fallback_grid
            for index in range(2, attempt_slots + 1):
                value = secondary
                if value is None:
                    if duplicate_primary_attempt:
                        value = primary
                    elif _is_grid(fallback_grid):
                        value = fallback_grid
                attempts[f"attempt_{index}"] = value
            entries.append(attempts)
        submission[task_id] = entries
    return submission


def validate_arc_submission(submission: Submission) -> list[str]:
    errors: list[str] = []
    for task_id, entries in submission.items():
        if not isinstance(task_id, str) or not task_id:
            errors.append("Submission keys must be non-empty task ids")
            continue
        if not isinstance(entries, list) or not entries:
            errors.append(f"{task_id}: submission must contain at least one test entry")
            continue
        for sample_index, attempt_map in enumerate(entries):
            if not isinstance(attempt_map, dict):
                errors.append(f"{task_id}[{sample_index}]: attempts entry must be an object")
                continue
            for attempt_name, grid in attempt_map.items():
                if not attempt_name.startswith("attempt_"):
                    errors.append(f"{task_id}[{sample_index}]: invalid attempt key {attempt_name}")
                    continue
                if grid is not None and not _is_grid(grid):
                    errors.append(f"{task_id}[{sample_index}]: {attempt_name} is not a valid grid")
    return errors


def write_arc_submission(output_path: str | Path, submission: Submission) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    return target


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if not isinstance(row, list):
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _row_predictions(row: dict[str, Any]) -> list[Grid]:
    predictions: list[Grid] = []
    raw_predictions = row.get("predictions")
    if isinstance(raw_predictions, list):
        for value in raw_predictions:
            if _is_grid(value):
                predictions.append(value)
    primary = row.get("predicted")
    if _is_grid(primary) and primary not in predictions:
        predictions.insert(0, primary)
    secondary = row.get("secondary_prediction")
    if _is_grid(secondary) and secondary not in predictions:
        predictions.append(secondary)
    return predictions


__all__ = [
    "format_arc_submission",
    "validate_arc_submission",
    "write_arc_submission",
]
