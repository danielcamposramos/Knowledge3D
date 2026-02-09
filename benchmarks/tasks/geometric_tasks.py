"""Deterministic geometric task generation helpers."""

from __future__ import annotations

import random
from typing import Any


OPS = ("ROTATE_90", "ROTATE_180", "MIRROR_H", "MIRROR_V", "TRANSPOSE")


def apply_geometric_op(grid: list[list[int]], operation: str) -> list[list[int]]:
    rows = [list(map(int, row)) for row in grid]
    op = operation.upper()
    if op == "ROTATE_90":
        return [list(col) for col in zip(*rows[::-1])]
    if op == "ROTATE_180":
        return [list(reversed(row)) for row in reversed(rows)]
    if op == "MIRROR_H":
        return [list(reversed(row)) for row in rows]
    if op == "MIRROR_V":
        return list(reversed(rows))
    if op == "TRANSPOSE":
        return [list(col) for col in zip(*rows)]
    return rows


def _random_grid(rng: random.Random, *, min_side: int = 2, max_side: int = 5) -> list[list[int]]:
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


def generate_geometric_tasks(count: int, seed: int = 1337) -> list[dict[str, Any]]:
    """Generate deterministic geometric transformation tasks."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for idx in range(max(0, int(count))):
        op = OPS[idx % len(OPS)]
        grid = _random_grid(rng)
        expected = apply_geometric_op(grid, op)
        tasks.append(
            {
                "id": f"geom_{idx:04d}",
                "category": "geometric_transforms",
                "operation": op,
                "input": grid,
                "expected": expected,
                "query": f"apply {op.lower()} to grid",
            }
        )
    return tasks

