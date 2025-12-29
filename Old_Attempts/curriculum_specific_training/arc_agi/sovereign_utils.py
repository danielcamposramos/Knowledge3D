"""Sovereign, numpy-free helpers for ARC-AGI grid and vector operations."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T", int, float, bool)


def grid_shape(grid: Sequence[Sequence[T]]) -> Tuple[int, int]:
    """Return (height, width) for a 2D grid-like object."""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    return h, w


def to_int_grid(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Deep-copy grid to a List[List[int]]."""
    return [[int(cell) for cell in row] for row in grid]


def copy_grid(grid: Sequence[Sequence[T]]) -> List[List[T]]:
    """Deep-copy a grid."""
    return [list(row) for row in grid]


def zeros1d(length: int, fill: float = 0.0) -> List[float]:
    """Create a 1D list filled with a value."""
    return [fill for _ in range(max(0, length))]


def zeros2d(height: int, width: int, fill: T = 0) -> List[List[T]]:
    """Create a 2D list filled with a value."""
    return [[fill for _ in range(max(0, width))] for _ in range(max(0, height))]


def zeros_like_grid(grid: Sequence[Sequence[T]], fill: T = 0) -> List[List[T]]:
    """Create a zero-filled grid with the same shape as input."""
    h, w = grid_shape(grid)
    return zeros2d(h, w, fill)


def flatten(grid: Sequence[Sequence[T]]) -> List[T]:
    """Flatten a 2D grid into a 1D list."""
    return [cell for row in grid for cell in row]


def max_abs(values: Iterable[float]) -> float:
    """Return maximum absolute value from iterable (0.0 if empty)."""
    return max((abs(float(v)) for v in values), default=0.0)


def pad_or_truncate(seq: Sequence[T], length: int, fill: T = 0) -> List[T]:
    """Pad sequence with fill or truncate to a target length."""
    data = list(seq)
    if len(data) >= length:
        return data[:length]
    return data + [fill for _ in range(length - len(data))]


def unique_nonzero(grid: Sequence[Sequence[int]]) -> List[int]:
    """Return sorted unique non-zero values from a grid."""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(int(cell))
    return sorted(colors)


def unique_counts(values: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Return (unique_values, counts) similar to numpy.unique(return_counts=True)."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    uniques = sorted(counts.keys())
    return uniques, [counts[u] for u in uniques]


def count_nonzero_grid(grid: Sequence[Sequence[int]]) -> int:
    """Count non-zero entries in a grid."""
    return sum(1 for cell in flatten(grid) if cell != 0)


def bounding_box_nonzero(grid: Sequence[Sequence[int]]) -> Tuple[int, int, int, int] | None:
    """Return bounding box (y0, y1, x0, x1) for non-zero cells, or None if empty."""
    h, w = grid_shape(grid)
    min_y, max_y = h, -1
    min_x, max_x = w, -1
    for y in range(h):
        for x in range(w):
            if grid[y][x] != 0:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
    if max_y == -1:
        return None
    return min_y, max_y, min_x, max_x


def rotate_cw(grid: Sequence[Sequence[int]], times: int = 1) -> List[List[int]]:
    """Rotate grid clockwise by 90 degrees * times."""
    result = to_int_grid(grid)
    times %= 4
    for _ in range(times):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def rotate_ccw(grid: Sequence[Sequence[int]], times: int = 1) -> List[List[int]]:
    """Rotate grid counter-clockwise by 90 degrees * times."""
    result = to_int_grid(grid)
    times %= 4
    for _ in range(times):
        result = [list(row) for row in zip(*result)][::-1]
    return result


def flip_horizontal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Flip grid left-right."""
    return [list(reversed(row)) for row in grid]


def flip_vertical(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Flip grid top-bottom."""
    return [list(row) for row in reversed(grid)]


def translate_grid(grid: Sequence[Sequence[int]], dx: int, dy: int, fill: int = 0) -> List[List[int]]:
    """Translate grid by dx, dy with fill value outside bounds."""
    h, w = grid_shape(grid)
    out = zeros2d(h, w, fill)
    for y in range(h):
        for x in range(w):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = int(grid[y][x])
    return out


def grids_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    """Check structural equality between two grids."""
    ha, wa = grid_shape(a)
    hb, wb = grid_shape(b)
    if ha != hb or wa != wb:
        return False
    for row_a, row_b in zip(a, b):
        if list(row_a) != list(row_b):
            return False
    return True


def mask_any(mask: Sequence[Sequence[bool]]) -> bool:
    """True if any cell in mask is True."""
    return any(any(row) for row in mask)


def mask_sum(mask: Sequence[Sequence[bool]]) -> int:
    """Count True values in mask."""
    return sum(1 for row in mask for cell in row if cell)


def mask_nonzero_positions(mask: Sequence[Sequence[bool]]) -> List[Tuple[int, int]]:
    """Return list of (y, x) where mask is True."""
    return [(y, x) for y, row in enumerate(mask) for x, val in enumerate(row) if val]


def translate_mask(mask: Sequence[Sequence[bool]], dx: int, dy: int) -> List[List[bool]]:
    """Translate boolean mask by dx, dy."""
    h, w = grid_shape(mask)
    out = zeros2d(h, w, False)
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = True
    return out


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean (0.0 if empty)."""
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def std(values: Sequence[float]) -> float:
    """Population standard deviation (0.0 if empty)."""
    vals = list(values)
    if not vals:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two sequences."""
    return sum(float(x) * float(y) for x, y in zip(a, b))


def l2_norm(values: Sequence[float]) -> float:
    """L2 norm of a sequence."""
    return math.sqrt(sum(float(v) * float(v) for v in values))


def pad_vector(vec: Sequence[float], length: int) -> List[float]:
    """Pad or truncate vector to target length."""
    return pad_or_truncate([float(v) for v in vec], length, 0.0)


def most_common_value(values: Sequence[int]) -> int:
    """Return value with highest frequency (smallest value wins ties)."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return 0
    best_val = None
    best_count = -1
    for v in sorted(counts.keys()):
        c = counts[v]
        if c > best_count:
            best_val = v
            best_count = c
    return int(best_val)


def is_grid(obj: object) -> bool:
    """Return True if obj looks like a 2D grid (list/tuple of list/tuple)."""
    if not isinstance(obj, (list, tuple)):
        return False
    if not obj:
        return True
    first_row_len = None
    for row in obj:
        if not isinstance(row, (list, tuple)):
            return False
        if first_row_len is None:
            first_row_len = len(row)
        elif len(row) != first_row_len:
            return False
    return True
