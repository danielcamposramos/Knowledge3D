"""Deterministic arithmetic task generation helpers."""

from __future__ import annotations

import random
from typing import Any


OPS = ("COUNT_VALUE", "SUM_ALL", "MAX_VALUE", "MIN_VALUE", "UNIQUE_COUNT")


def evaluate_arithmetic_task(grid: list[list[int]], operation: str, value: int | None = None) -> int:
    flat = [int(cell) for row in grid for cell in row]
    op = operation.upper()
    if op == "COUNT_VALUE":
        if value is None:
            return 0
        return sum(1 for cell in flat if cell == int(value))
    if op == "SUM_ALL":
        return sum(flat)
    if op == "MAX_VALUE":
        return max(flat) if flat else 0
    if op == "MIN_VALUE":
        return min(flat) if flat else 0
    if op == "UNIQUE_COUNT":
        return len(set(flat))
    return 0


def _random_grid(rng: random.Random, *, min_side: int = 2, max_side: int = 5) -> list[list[int]]:
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


def generate_arithmetic_tasks(count: int, seed: int = 1338) -> list[dict[str, Any]]:
    """Generate deterministic grid arithmetic tasks."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for idx in range(max(0, int(count))):
        op = OPS[idx % len(OPS)]
        grid = _random_grid(rng)
        value = None
        if op == "COUNT_VALUE":
            row = rng.choice(grid)
            value = int(rng.choice(row))
        expected = evaluate_arithmetic_task(grid, op, value=value)
        task: dict[str, Any] = {
            "id": f"arith_{idx:04d}",
            "category": "grid_arithmetic",
            "operation": op,
            "input": grid,
            "expected": expected,
            "query": f"apply {op.lower()} on grid",
        }
        if value is not None:
            task["value"] = value
            task["query"] += f" with value={value}"
        tasks.append(task)
    return tasks

