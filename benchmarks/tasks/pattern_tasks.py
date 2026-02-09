"""Deterministic pattern-completion task generation helpers."""

from __future__ import annotations

import random
from typing import Any


PATTERN_TYPES = (
    "ALTERNATING_NEXT",
    "ARITHMETIC_NEXT",
    "GEOMETRIC_NEXT",
    "MIRROR_COMPLETE",
    "ROW_TILE",
)


def solve_pattern_task(task: dict[str, Any]) -> Any:
    pattern_type = str(task.get("pattern_type", "")).upper()
    if pattern_type == "ALTERNATING_NEXT":
        seq = list(task["input"]["sequence"])
        if len(seq) < 2:
            return 0
        return seq[-2]
    if pattern_type == "ARITHMETIC_NEXT":
        seq = list(task["input"]["sequence"])
        if len(seq) < 2:
            return 0
        step = seq[1] - seq[0]
        return seq[-1] + step
    if pattern_type == "GEOMETRIC_NEXT":
        seq = list(task["input"]["sequence"])
        if len(seq) < 2 or seq[0] == 0:
            return 0
        ratio = seq[1] // seq[0]
        return seq[-1] * ratio
    if pattern_type == "MIRROR_COMPLETE":
        half = list(task["input"]["half"])
        return half + list(reversed(half))
    if pattern_type == "ROW_TILE":
        row = list(task["input"]["row"])
        length = int(task["input"]["length"])
        out = [row[i % len(row)] for i in range(length)]
        return out
    return task.get("input")


def generate_pattern_tasks(count: int, seed: int = 1339) -> list[dict[str, Any]]:
    """Generate deterministic sequence/pattern completion tasks."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    for idx in range(max(0, int(count))):
        pattern_type = PATTERN_TYPES[idx % len(PATTERN_TYPES)]
        task: dict[str, Any] = {
            "id": f"pat_{idx:04d}",
            "category": "pattern_completion",
            "pattern_type": pattern_type,
        }

        if pattern_type == "ALTERNATING_NEXT":
            a = rng.randint(1, 5)
            b = rng.randint(6, 9)
            seq = [a, b, a, b]
            task["input"] = {"sequence": seq}
            task["query"] = "predict next value for alternating sequence"
        elif pattern_type == "ARITHMETIC_NEXT":
            start = rng.randint(1, 10)
            step = rng.randint(1, 4)
            seq = [start + step * j for j in range(4)]
            task["input"] = {"sequence": seq}
            task["query"] = "predict next value for arithmetic progression"
        elif pattern_type == "GEOMETRIC_NEXT":
            start = rng.randint(1, 4)
            ratio = rng.choice([2, 3])
            seq = [start * (ratio**j) for j in range(4)]
            task["input"] = {"sequence": seq}
            task["query"] = "predict next value for geometric progression"
        elif pattern_type == "MIRROR_COMPLETE":
            half = [rng.randint(0, 9) for _ in range(3)]
            task["input"] = {"half": half}
            task["query"] = "complete mirrored row from first half"
        else:
            row = [rng.randint(0, 9) for _ in range(3)]
            length = rng.randint(6, 9)
            task["input"] = {"row": row, "length": length}
            task["query"] = "tile row pattern to target length"

        task["expected"] = solve_pattern_task(task)
        tasks.append(task)

    return tasks

