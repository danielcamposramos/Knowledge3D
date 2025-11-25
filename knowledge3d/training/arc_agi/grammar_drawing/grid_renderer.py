"""Render drawing grammar outputs to ARC-like grids (placeholder)."""

from __future__ import annotations

from typing import List

import numpy as np


def render_to_grid(width: int = 3, height: int = 3, color: int = 1) -> List[List[int]]:
    """Return a minimal filled grid placeholder."""
    grid = np.zeros((height, width), dtype=int)
    if height > 0 and width > 0:
        grid[height // 2, width // 2] = color
    return grid.tolist()


__all__ = ["render_to_grid"]
