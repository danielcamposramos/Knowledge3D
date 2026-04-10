"""Seed ARC task demonstration pairs as Grammar Galaxy rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

from knowledge3d.cranium.ptx_runtime.rpn_opcodes import OP_POLY_BUILD
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


ARC_GRID_POLY_SYMBOL = 0xA7C0
ARC_MAX_HEIGHT = 30
ARC_MAX_WIDTH = 30
ARC_MAX_CELLS = ARC_MAX_HEIGHT * ARC_MAX_WIDTH


def _stable_task_id(task_json: dict[str, Any]) -> str:
    explicit = str(task_json.get("task_id") or task_json.get("id") or "").strip()
    if explicit:
        return explicit
    digest = hashlib.sha256(
        json.dumps(task_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"arc_{digest[:12]}"


def _normalize_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        return []
    rows: list[list[int]] = []
    width = None
    for row in grid:
        if not isinstance(row, list):
            raise ValueError("ARC grid rows must be lists")
        normalized = [int(cell) for cell in row]
        if width is None:
            width = len(normalized)
        if len(normalized) != width:
            raise ValueError("ARC grid must be rectangular")
        rows.append(normalized)
    if width is None:
        return []
    if len(rows) > ARC_MAX_HEIGHT or width > ARC_MAX_WIDTH:
        raise ValueError(f"ARC grid exceeds {ARC_MAX_HEIGHT}x{ARC_MAX_WIDTH}")
    return rows


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    return height, width


def _float_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def grid_to_rpn(grid: list[list[int]]) -> str:
    """Encode a 2D ARC grid as a deterministic text program."""
    normalized = _normalize_grid(grid)
    height, width = _grid_shape(normalized)
    tokens = [f"GRID_BEGIN {height} {width}"]
    for row in normalized:
        tokens.append(f"ROW_BEGIN {' '.join(str(int(value)) for value in row)} ROW_END")
    tokens.append("GRID_END")
    return " ".join(tokens)


def rpn_to_grid(text: str) -> list[list[int]] | None:
    tokens = [token for token in str(text or "").strip().split() if token]
    if len(tokens) < 4 or tokens[0] != "GRID_BEGIN" or tokens[-1] != "GRID_END":
        return None
    try:
        height = int(tokens[1])
        width = int(tokens[2])
    except Exception:
        return None
    rows: list[list[int]] = []
    index = 3
    while index < len(tokens) - 1:
        if tokens[index] != "ROW_BEGIN":
            return None
        index += 1
        row: list[int] = []
        while index < len(tokens) - 1 and tokens[index] != "ROW_END":
            try:
                row.append(int(tokens[index]))
            except Exception:
                return None
            index += 1
        if index >= len(tokens) - 1 or tokens[index] != "ROW_END":
            return None
        rows.append(row)
        index += 1
    if len(rows) != height or any(len(row) != width for row in rows):
        return None
    return rows


def grid_to_program_words(grid: list[list[int]]) -> list[int]:
    """Encode a grid into a compact STAR leaf program."""
    normalized = _normalize_grid(grid)
    height, width = _grid_shape(normalized)
    coeffs: list[float] = [float(height), float(width)]
    for row in normalized:
        coeffs.extend(float(int(cell)) for cell in row)
    if len(coeffs) - 2 > ARC_MAX_CELLS:
        raise ValueError(f"ARC grid exceeds {ARC_MAX_CELLS} cells")
    words = [int(OP_POLY_BUILD), int(ARC_GRID_POLY_SYMBOL), int(len(coeffs))]
    words.extend(_float_word(value) for value in coeffs)
    return words


def grid_rpn_to_program_words(text: str) -> list[int]:
    """Compile the portable text form into the live compact grid program."""
    grid = rpn_to_grid(text)
    if grid is None:
        raise ValueError("RPN text does not encode a valid ARC grid")
    return grid_to_program_words(grid)


def pair_to_grammar_rule(
    task_id: str,
    pair_idx: int,
    input_grid: list[list[int]],
    output_grid: list[list[int]],
) -> MeaningCentricStar:
    input_rpn = grid_to_rpn(input_grid)
    output_rpn = grid_to_rpn(output_grid)
    return MeaningCentricStar(
        star_id=f"spatial_grid_transform:{task_id}:pair{int(pair_idx)}",
        meaning_class="spatial_grid_transform",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn=input_rpn,
        behavior_rpn=output_rpn,
        taxonomy_refs=["spatial_grid_transform", "two_dimensional", "input_output_pair"],
        grammar_refs=["game2d_transformation"],
        confidence=1,
        polarity=1,
    )


def _arc_provenance_from_star(star: MeaningCentricStar) -> dict[str, Any]:
    star_id = str(star.star_id or "").strip()
    task_id = ""
    pair_idx: int | None = None
    parts = star_id.split(":")
    if len(parts) >= 3 and parts[0] == "spatial_grid_transform":
        task_id = str(parts[1]).strip()
        tail = str(parts[2]).strip().lower()
        if tail.startswith("pair"):
            try:
                pair_idx = int(tail[4:])
            except Exception:
                pair_idx = None
    metadata: dict[str, Any] = {"benchmark_source": "arc_agi_2"}
    if task_id:
        metadata["task_id"] = task_id
    if pair_idx is not None:
        metadata["pair_idx"] = int(pair_idx)
    return metadata


def load_task_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ARC task at {path} must decode to an object")
    payload = dict(payload)
    payload.setdefault("task_id", Path(path).stem)
    return payload


def _store_stars(galaxy_manager: Any, stars: list[MeaningCentricStar]) -> None:
    if galaxy_manager is None or not stars:
        return
    sync_ctx = getattr(galaxy_manager, "bulk_disk_sync", None)
    if callable(sync_ctx):
        context = sync_ctx()
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        for star in stars:
            if hasattr(galaxy_manager, "store_meaning_star"):
                galaxy_manager.store_meaning_star(
                    "Grammar",
                    star,
                    category="meaning_star",
                    metadata={
                        "bootstrap": "arc_r0_task_seeder",
                        **_arc_provenance_from_star(star),
                    },
                )


def seed_task(
    task_json: dict[str, Any],
    galaxy_manager=None,
    *,
    task_id: str | None = None,
) -> list[MeaningCentricStar]:
    """Parse one ARC task payload and return Grammar rules for all train pairs."""
    payload = dict(task_json or {})
    resolved_task_id = str(task_id or _stable_task_id(payload))
    train_rows = list(payload.get("train") or [])
    stars: list[MeaningCentricStar] = []
    for pair_idx, row in enumerate(train_rows):
        if not isinstance(row, dict):
            continue
        input_grid = _normalize_grid(row.get("input"))
        output_grid = _normalize_grid(row.get("output"))
        stars.append(pair_to_grammar_rule(resolved_task_id, pair_idx, input_grid, output_grid))
    _store_stars(galaxy_manager, stars)
    return stars


def seed_tasks_directory(tasks_dir: str | Path, galaxy_manager=None) -> dict[str, list[MeaningCentricStar]]:
    """Walk a directory of ARC task files and seed all of them."""
    root = Path(tasks_dir)
    seeded: dict[str, list[MeaningCentricStar]] = {}
    for path in sorted(root.glob("*.json")):
        task_json = load_task_json(path)
        task_id = _stable_task_id(task_json)
        seeded[task_id] = seed_task(task_json, galaxy_manager=galaxy_manager, task_id=task_id)
    return seeded


__all__ = [
    "ARC_GRID_POLY_SYMBOL",
    "ARC_MAX_CELLS",
    "grid_rpn_to_program_words",
    "grid_to_program_words",
    "grid_to_rpn",
    "load_task_json",
    "pair_to_grammar_rule",
    "rpn_to_grid",
    "seed_task",
    "seed_tasks_directory",
]
