"""Deterministic compositional task generation helpers."""

from __future__ import annotations

import random
from typing import Any

from .geometric_tasks import OPS as GEOMETRIC_OPS
from .geometric_tasks import apply_geometric_op


def generate_compositional_tasks(count: int, seed: int = 1340) -> list[dict[str, Any]]:
    """Generate deterministic multi-step transformation tasks."""
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for idx in range(max(0, int(count))):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        grid = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
        ops = [GEOMETRIC_OPS[(idx + j) % len(GEOMETRIC_OPS)] for j in range(2)]
        transformed = [row[:] for row in grid]
        for op in ops:
            transformed = apply_geometric_op(transformed, op)
        tasks.append(
            {
                "id": f"comp_{idx:04d}",
                "category": "compositional",
                "input": grid,
                "operations": ops,
                "expected": transformed,
                "query": f"apply {' then '.join(op.lower() for op in ops)} to grid",
            }
        )
    return tasks

