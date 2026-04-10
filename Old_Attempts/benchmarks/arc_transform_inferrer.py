"""Infer simple ARC transformations from training pairs and apply them."""

from __future__ import annotations

import json
from typing import Any


Grid = list[list[int]]

TRANSFORM_IDENTITY = "identity"
TRANSFORM_FLIP_H = "flip_h"
TRANSFORM_FLIP_V = "flip_v"
TRANSFORM_ROT90 = "rot90"
TRANSFORM_ROT180 = "rot180"
TRANSFORM_ROT270 = "rot270"
TRANSFORM_COLOR_PERM = "color_perm"
TRANSFORM_TILE_2X = "tile_2x"
TRANSFORM_TILE_3X = "tile_3x"
TRANSFORM_SCALE_2X = "scale_2x"
TRANSFORM_SCALE_3X = "scale_3x"
TRANSFORM_UNKNOWN = "nearest_training_pair"


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)


def _rectangular(grid: Grid) -> bool:
    if not isinstance(grid, list):
        return False
    if not grid:
        return True
    width = len(grid[0])
    return all(isinstance(row, list) and len(row) == width for row in grid)


def _flip_h(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_v(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def _rot90(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]


def _rot180(grid: Grid) -> Grid:
    return _flip_v(_flip_h(grid))


def _rot270(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]


def _tile(grid: Grid, factor: int) -> Grid:
    tiled_rows = [list(row) * factor for row in grid]
    out: Grid = []
    for _ in range(factor):
        out.extend(_copy_grid(tiled_rows))
    return out


def _scale(grid: Grid, factor: int) -> Grid:
    out: Grid = []
    for row in grid:
        expanded_row: list[int] = []
        for cell in row:
            expanded_row.extend([int(cell)] * factor)
        for _ in range(factor):
            out.append(list(expanded_row))
    return out


def _color_perm(input_grid: Grid, output_grid: Grid) -> dict[int, int] | None:
    if _shape(input_grid) != _shape(output_grid):
        return None
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for input_row, output_row in zip(input_grid, output_grid):
        for src, dst in zip(input_row, output_row):
            src = int(src)
            dst = int(dst)
            if src in mapping and mapping[src] != dst:
                return None
            if dst in reverse and reverse[dst] != src:
                return None
            mapping[src] = dst
            reverse[dst] = src
    return mapping


def detect_transform(input_grid: Grid, output_grid: Grid) -> dict[str, Any]:
    if not (_rectangular(input_grid) and _rectangular(output_grid)):
        return {"type": TRANSFORM_UNKNOWN}

    if output_grid == input_grid:
        return {"type": TRANSFORM_IDENTITY}
    if _flip_h(input_grid) == output_grid:
        return {"type": TRANSFORM_FLIP_H}
    if _flip_v(input_grid) == output_grid:
        return {"type": TRANSFORM_FLIP_V}
    if _rot90(input_grid) == output_grid:
        return {"type": TRANSFORM_ROT90}
    if _rot180(input_grid) == output_grid:
        return {"type": TRANSFORM_ROT180}
    if _rot270(input_grid) == output_grid:
        return {"type": TRANSFORM_ROT270}

    mapping = _color_perm(input_grid, output_grid)
    if mapping is not None:
        return {"type": TRANSFORM_COLOR_PERM, "mapping": mapping}

    if _tile(input_grid, 2) == output_grid:
        return {"type": TRANSFORM_TILE_2X}
    if _tile(input_grid, 3) == output_grid:
        return {"type": TRANSFORM_TILE_3X}
    if _scale(input_grid, 2) == output_grid:
        return {"type": TRANSFORM_SCALE_2X}
    if _scale(input_grid, 3) == output_grid:
        return {"type": TRANSFORM_SCALE_3X}

    return {"type": TRANSFORM_UNKNOWN}


def _transform_signature(transform: dict[str, Any]) -> tuple[Any, ...]:
    kind = str(transform.get("type", TRANSFORM_UNKNOWN))
    if kind == TRANSFORM_COLOR_PERM:
        mapping = transform.get("mapping") or {}
        return (kind, tuple(sorted((int(k), int(v)) for k, v in dict(mapping).items())))
    return (kind,)


def infer_task_transform(training_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not training_pairs:
        return {"type": TRANSFORM_UNKNOWN}

    detected: list[dict[str, Any]] = []
    for pair in training_pairs:
        input_grid = _copy_grid(list(pair.get("input") or []))
        output_grid = _copy_grid(list(pair.get("output") or []))
        detected.append(detect_transform(input_grid, output_grid))

    grouped: dict[tuple[Any, ...], tuple[dict[str, Any], int]] = {}
    for transform in detected:
        signature = _transform_signature(transform)
        exemplar, count = grouped.get(signature, (transform, 0))
        grouped[signature] = (exemplar, count + 1)

    if len(grouped) == 1:
        return dict(next(iter(grouped.values()))[0])

    best_transform, best_count = max(grouped.values(), key=lambda item: item[1])
    if best_count * 3 >= len(training_pairs) * 2:
        result = dict(best_transform)
        result["confidence"] = "majority"
        return result
    return {"type": TRANSFORM_UNKNOWN}


def apply_transform(grid: Grid, transform: dict[str, Any]) -> Grid:
    kind = str(transform.get("type", TRANSFORM_UNKNOWN))
    if kind == TRANSFORM_IDENTITY:
        return _copy_grid(grid)
    if kind == TRANSFORM_FLIP_H:
        return _flip_h(grid)
    if kind == TRANSFORM_FLIP_V:
        return _flip_v(grid)
    if kind == TRANSFORM_ROT90:
        return _rot90(grid)
    if kind == TRANSFORM_ROT180:
        return _rot180(grid)
    if kind == TRANSFORM_ROT270:
        return _rot270(grid)
    if kind == TRANSFORM_COLOR_PERM:
        mapping = {int(k): int(v) for k, v in dict(transform.get("mapping") or {}).items()}
        return [[mapping.get(int(cell), int(cell)) for cell in row] for row in grid]
    if kind == TRANSFORM_TILE_2X:
        return _tile(grid, 2)
    if kind == TRANSFORM_TILE_3X:
        return _tile(grid, 3)
    if kind == TRANSFORM_SCALE_2X:
        return _scale(grid, 2)
    if kind == TRANSFORM_SCALE_3X:
        return _scale(grid, 3)
    raise ValueError("use nearest_training_pair fallback path")


def transform_type_to_rpn(transform: dict[str, Any]) -> str:
    kind = str(transform.get("type", TRANSFORM_UNKNOWN))
    if kind == TRANSFORM_IDENTITY:
        return "GRID IDENTITY_TRANSFORM"
    if kind == TRANSFORM_FLIP_H:
        return "GRID FLIP_H"
    if kind == TRANSFORM_FLIP_V:
        return "GRID FLIP_V"
    if kind == TRANSFORM_ROT90:
        return "GRID ROT90"
    if kind == TRANSFORM_ROT180:
        return "GRID ROT180"
    if kind == TRANSFORM_ROT270:
        return "GRID ROT270"
    if kind == TRANSFORM_COLOR_PERM:
        mapping = json.dumps(transform.get("mapping") or {}, sort_keys=True, separators=(",", ":"))
        return f"GRID {mapping} COLOR_PERM"
    if kind == TRANSFORM_TILE_2X:
        return "GRID 2 2 TILE"
    if kind == TRANSFORM_TILE_3X:
        return "GRID 3 3 TILE"
    if kind == TRANSFORM_SCALE_2X:
        return "GRID 2 SCALE_UNIFORM"
    if kind == TRANSFORM_SCALE_3X:
        return "GRID 3 SCALE_UNIFORM"
    return "GRID NEAREST_TRAINING_PAIR"


__all__ = [
    "Grid",
    "TRANSFORM_COLOR_PERM",
    "TRANSFORM_FLIP_H",
    "TRANSFORM_FLIP_V",
    "TRANSFORM_IDENTITY",
    "TRANSFORM_ROT90",
    "TRANSFORM_ROT180",
    "TRANSFORM_ROT270",
    "TRANSFORM_SCALE_2X",
    "TRANSFORM_SCALE_3X",
    "TRANSFORM_TILE_2X",
    "TRANSFORM_TILE_3X",
    "TRANSFORM_UNKNOWN",
    "apply_transform",
    "detect_transform",
    "infer_task_transform",
    "transform_type_to_rpn",
]
