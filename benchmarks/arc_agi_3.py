"""K3D sovereign ARC-AGI-3 agent — thin I/O adapter over Knowledgeverse."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.arc3_episode_galaxy import ARC3EpisodeGalaxy
from knowledge3d.knowledgeverse.arc3_frame_encoder import ARC3FrameEncoder
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.game2d_wine import arc3_game_envelope
from benchmarks.arc3_game_mechanics_seeder import seed_game_mechanics


ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]
RESET_ACTION_NAME = "RESET"
RESET_ACTION_LABEL = "Reset"
ARC3_ROUTE_GALAXIES: list[str] = []
def _normalize_grid(value: Any) -> list[list[int]]:
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        return [[int(cell) for cell in row] for row in value]
    return [[]]


def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _gameplay_grid(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not isinstance(grid[0], list):
        return [[]]
    if len(grid) <= 8:
        return _clone_grid(grid)
    if len(grid) >= 60:
        return _clone_grid(grid[:55])
    return _clone_grid(grid[:-5])


def _same_gameplay_state(left: list[list[int]] | None, right: list[list[int]] | None) -> bool:
    if left is None or right is None:
        return False
    return _gameplay_grid(left) == _gameplay_grid(right)


def _foreground_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    if not grid or not isinstance(grid[0], list):
        return None
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            cell = int(value)
            counts[cell] = counts.get(cell, 0) + 1
    background = max(counts, key=lambda key: counts[key]) if counts else 0
    foreground_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if int(value) != int(background)
    ]
    if not foreground_cells:
        return None
    avg_row = sum(float(row_index) for row_index, _ in foreground_cells) / float(len(foreground_cells))
    avg_col = sum(float(col_index) for _, col_index in foreground_cells) / float(len(foreground_cells))
    return avg_row, avg_col


def _focus_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    if not grid or not isinstance(grid[0], list):
        return None
    background = _background_value(grid)
    non_background_counts: Counter[int] = Counter(
        int(value) for row in grid for value in row if int(value) != int(background)
    )
    if not non_background_counts:
        return None

    # Prefer salient rare colors, which in live ARC3 more often correspond to
    # the avatar, key state, or new interactive blocks than the broad terrain.
    preferred_colors = [0, 1, 6, 9, 11, 12, 15]
    candidate_colors = [
        color
        for color in preferred_colors
        if color in non_background_counts and non_background_counts[color] <= 128
    ]
    if not candidate_colors:
        rarest_count = min(non_background_counts.values())
        candidate_colors = [
            color
            for color, count in non_background_counts.items()
            if count == rarest_count or count <= max(rarest_count * 2, 64)
        ]

    cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if int(value) in set(candidate_colors)
    ]
    if cells:
        avg_row = sum(float(row_index) for row_index, _ in cells) / float(len(cells))
        avg_col = sum(float(col_index) for _, col_index in cells) / float(len(cells))
        return avg_row, avg_col
    return _foreground_centroid(grid)


def _components_for_colors(grid: list[list[int]], colors: set[int]) -> list[dict[str, Any]]:
    if not grid or not isinstance(grid[0], list):
        return []
    rows = len(grid)
    cols = len(grid[0])
    allowed = {int(color) for color in colors}
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []

    for row in range(rows):
        for col in range(cols):
            value = int(grid[row][col])
            if value not in allowed or (row, col) in seen:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            seen.add((row, col))
            points: list[tuple[int, int]] = []
            color_counts: Counter[int] = Counter()
            while queue:
                current_row, current_col = queue.popleft()
                current_value = int(grid[current_row][current_col])
                points.append((current_row, current_col))
                color_counts[current_value] += 1
                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if (next_row, next_col) in seen:
                        continue
                    if int(grid[next_row][next_col]) not in allowed:
                        continue
                    seen.add((next_row, next_col))
                    queue.append((next_row, next_col))
            if not points:
                continue
            row_values = [point[0] for point in points]
            col_values = [point[1] for point in points]
            components.append(
                {
                    "size": len(points),
                    "centroid": (
                        sum(float(point[0]) for point in points) / float(len(points)),
                        sum(float(point[1]) for point in points) / float(len(points)),
                    ),
                    "bbox": (
                        min(row_values),
                        min(col_values),
                        max(row_values),
                        max(col_values),
                    ),
                    "colors": set(color_counts.keys()),
                    "points": tuple(points),
                }
            )
    return components


def _avatar_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    if not grid or not isinstance(grid[0], list):
        return None
    rows = len(grid)
    cols = len(grid[0])
    center_row = (rows - 1) / 2.0
    center_col = (cols - 1) / 2.0
    candidates = [
        component
        for component in _components_for_colors(grid, {0, 1})
        if component["size"] <= 8 and {0, 1}.issubset(component["colors"])
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda component: (
            abs(int(component["size"]) - 5),
            abs(float(component["centroid"][0]) - center_row)
            + abs(float(component["centroid"][1]) - center_col),
        ),
    )
    return float(best["centroid"][0]), float(best["centroid"][1])


def _component_dimensions(component: dict[str, Any]) -> tuple[int, int]:
    min_row, min_col, max_row, max_col = component["bbox"]
    return int(max_row) - int(min_row) + 1, int(max_col) - int(min_col) + 1


def _position_label(
    rows: int,
    cols: int,
    centroid: tuple[float, float],
) -> str:
    row_ratio = float(centroid[0]) / float(max(1, rows - 1))
    col_ratio = float(centroid[1]) / float(max(1, cols - 1))
    if row_ratio <= 0.2:
        vertical = "top"
    elif row_ratio <= 0.4:
        vertical = "upper"
    elif row_ratio < 0.6:
        vertical = "center"
    elif row_ratio < 0.8:
        vertical = "lower"
    else:
        vertical = "bottom"
    if col_ratio <= 0.2:
        horizontal = "left"
    elif col_ratio < 0.8:
        horizontal = "center"
    else:
        horizontal = "right"
    return f"{vertical}_{horizontal}"


def _monochrome_components(grid: list[list[int]]) -> list[dict[str, Any]]:
    gameplay = _gameplay_grid(grid)
    if not gameplay or not gameplay[0]:
        return []
    rows = len(gameplay)
    cols = len(gameplay[0])
    background = _background_value(gameplay)
    colors = sorted({int(value) for row in gameplay for value in row if int(value) != background})
    components: list[dict[str, Any]] = []
    for color in colors:
        for component in _components_for_colors(gameplay, {int(color)}):
            height, width = _component_dimensions(component)
            payload = dict(component)
            payload["primary_color"] = int(color)
            payload["height"] = int(height)
            payload["width"] = int(width)
            payload["position_label"] = _position_label(rows, cols, payload["centroid"])
            payload["approximate_position"] = str(payload["position_label"])
            components.append(payload)
    return components


def _diagnose_frame_colors(grid: list[list[int]]) -> list[dict[str, Any]]:
    gameplay = _gameplay_grid(grid)
    if not gameplay or not gameplay[0]:
        return []
    background = _background_value(gameplay)
    rows = len(gameplay)
    cols = len(gameplay[0])
    diagnostics: list[dict[str, Any]] = []
    for component in _monochrome_components(gameplay):
        color = int(component["primary_color"])
        if color == background:
            continue
        diagnostics.append(
            {
                "color": int(color),
                "pixels": int(component["size"]),
                "centroid": (
                    float(component["centroid"][0]),
                    float(component["centroid"][1]),
                ),
                "position_label": str(component["position_label"]),
                "bbox": tuple(int(value) for value in component["bbox"]),
                "height": int(component["height"]),
                "width": int(component["width"]),
                "covers_ratio": float(component["size"]) / float(max(1, rows * cols)),
            }
        )
    diagnostics.sort(
        key=lambda row: (
            float(row["centroid"][0]),
            -int(row["pixels"]),
            int(row["color"]),
        )
    )
    return diagnostics


def _detect_static_objects(
    grid: list[list[int]],
    avatar_centroid: tuple[float, float] | None,
    *,
    known_objects: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    gameplay = _gameplay_grid(grid)
    if not gameplay or not gameplay[0]:
        return []
    rows = len(gameplay)
    cols = len(gameplay[0])
    avatar_row = int(round(float(avatar_centroid[0]))) if avatar_centroid is not None else -999
    avatar_col = int(round(float(avatar_centroid[1]))) if avatar_centroid is not None else -999
    background = _background_value(gameplay)
    discovered: list[dict[str, Any]] = []
    for component in _monochrome_components(gameplay):
        color = int(component["primary_color"])
        if color == background:
            continue
        size = int(component["size"])
        height = int(component["height"])
        width = int(component["width"])
        if size <= 1:
            continue
        if size > max(512, int(rows * cols * 0.2)):
            continue
        if height >= int(rows * 0.75) or width >= int(cols * 0.75):
            continue
        centroid = (
            float(component["centroid"][0]),
            float(component["centroid"][1]),
        )
        centroid_row = int(round(centroid[0]))
        centroid_col = int(round(centroid[1]))
        if abs(centroid_row - avatar_row) <= 1 and abs(centroid_col - avatar_col) <= 1 and color in {0, 1}:
            continue
        position_label = str(component["position_label"])
        known = dict((known_objects or {}).get((color, position_label), {}))
        semantic_hint = str(known.get("semantic_hint") or "").strip().lower()
        if not semantic_hint:
            if centroid[0] <= float(rows) * 0.45 and size >= 12 and height >= 4:
                semantic_hint = "door"
            elif size <= 20 and height <= 6 and width <= 6:
                semantic_hint = "switch"
            elif centroid[0] >= float(rows) * 0.6 and 4 <= size <= 36:
                semantic_hint = "recharge"
        discovered.append(
            {
                "type": "ARC3_OBJECT",
                "color": int(color),
                "centroid": centroid,
                "size": int(size),
                "height": int(height),
                "width": int(width),
                "position_label": position_label,
                "approximate_position": position_label,
                "semantic_hint": semantic_hint,
                "behavior": str(known.get("behavior") or "unknown"),
                "interaction_tested": bool(known.get("interaction_tested", False)),
                "interaction_result": known.get("interaction_result"),
                "moves_with_avatar": False,
                "points": tuple(component["points"]),
            }
        )
    discovered.sort(
        key=lambda row: (
            0 if str(row.get("semantic_hint", "")).strip() else 1,
            float(abs(float(row["centroid"][0]) - float(avatar_centroid[0]))) + float(abs(float(row["centroid"][1]) - float(avatar_centroid[1])))
            if avatar_centroid is not None
            else float(row["size"]),
        )
    )
    return discovered


def _adjacent_colors(
    grid: list[list[int]],
    avatar_centroid: tuple[float, float] | None,
) -> list[int]:
    if avatar_centroid is None or not grid or not grid[0]:
        return []
    row = int(round(float(avatar_centroid[0])))
    col = int(round(float(avatar_centroid[1])))
    colors: list[int] = []
    seen: set[int] = set()
    for next_row, next_col in (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        if not (0 <= next_row < len(grid) and 0 <= next_col < len(grid[0])):
            continue
        color = int(grid[next_row][next_col])
        if color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _background_value(grid: list[list[int]]) -> int:
    counts: Counter[int] = Counter(int(value) for row in grid for value in row)
    return int(counts.most_common(1)[0][0]) if counts else 0


def _frame_state(grid: list[list[int]]) -> str:
    if not grid or not grid[0]:
        return "unknown"
    rows = len(grid)
    cols = len(grid[0]) if grid and isinstance(grid[0], list) else 0
    counts: Counter[int] = Counter(int(value) for row in grid for value in row)
    total = max(1, sum(counts.values()))
    dominant_color, dominant_count = counts.most_common(1)[0]
    dominant_ratio = float(dominant_count) / float(total)
    distinct_colors = len(counts)
    normal_gameplay_colors = {3, 4, 5}
    if (
        rows >= 32
        and cols >= 32
        and dominant_color not in normal_gameplay_colors
        and dominant_ratio >= 0.8
        and distinct_colors <= 4
    ):
        return "transition"
    return "gameplay"


def _flash_semantics(grid: list[list[int]]) -> str:
    if not grid or not grid[0]:
        return ""
    counts: Counter[int] = Counter(int(value) for row in grid for value in row)
    total = max(1, sum(counts.values()))
    dominant_color, dominant_count = counts.most_common(1)[0]
    dominant_ratio = float(dominant_count) / float(total)
    if dominant_ratio < 0.8:
        return ""
    if dominant_color in {8, 11}:
        return "failure"
    if dominant_color in {6, 9, 15}:
        return "success"
    return "transition"


def _movement_budget_snapshot(grid: list[list[int]]) -> dict[str, Any] | None:
    if not grid or not grid[0] or len(grid) < 3:
        return None
    rows = [len(grid) - 3, len(grid) - 2]
    track_colors = {3, 11}
    cells = [
        int(grid[row_index][col_index])
        for row_index in rows
        for col_index in range(len(grid[row_index]))
        if int(grid[row_index][col_index]) in track_colors
    ]
    if len(cells) < 8:
        return None
    row_count = len(rows)
    capacity_units = max(1, len(cells) // row_count)
    remaining_units = sum(1 for value in cells if value == 11) // row_count
    spent_units = sum(1 for value in cells if value == 3) // row_count
    fraction = float(remaining_units) / float(capacity_units)
    if fraction <= 0.15 or remaining_units <= 4:
        bucket = "critical"
    elif fraction <= 0.35 or remaining_units <= 12:
        bucket = "low"
    else:
        bucket = "healthy"
    return {
        "remaining_units": int(remaining_units),
        "capacity_units": int(capacity_units),
        "spent_units": int(spent_units),
        "fraction": float(fraction),
        "bucket": bucket,
    }


def _lives_remaining(grid: list[list[int]]) -> int | None:
    if not grid or not grid[0] or len(grid) < 3:
        return None
    rows = [len(grid) - 3, len(grid) - 2]
    right_start = max(0, len(grid[0]) - 10)
    red_cells = 0
    for row_index in rows:
        for col_index in range(right_start, len(grid[row_index])):
            if int(grid[row_index][col_index]) == 8:
                red_cells += 1
    if red_cells < 4:
        return None
    return max(0, int(round(float(red_cells) / 4.0)))


def _reference_box_visible(grid: list[list[int]]) -> bool:
    if not grid or not grid[0] or len(grid) < 8:
        return False
    max_row = len(grid) - 3
    focus_rows = range(max(0, max_row - 5), max_row)
    focus_cols = range(0, min(12, len(grid[0])))
    blue_cells = sum(
        1
        for row_index in focus_rows
        for col_index in focus_cols
        if int(grid[row_index][col_index]) == 9
    )
    return blue_cells >= 6


def _estimate_grid_steps(
    source: tuple[float, float] | None,
    target: tuple[float, float] | None,
) -> int | None:
    if source is None or target is None:
        return None
    return int(round(abs(float(target[0]) - float(source[0])) + abs(float(target[1]) - float(source[1]))))


def _should_force_reset(
    *,
    budget_snapshot: dict[str, Any] | None,
    avatar_centroid: tuple[float, float] | None,
    target_centroid: tuple[float, float] | None,
    target_label: str,
) -> bool:
    if not budget_snapshot:
        return False
    remaining_units = int(budget_snapshot.get("remaining_units", 0))
    bucket = str(budget_snapshot.get("bucket", ""))
    if bucket not in {"low", "critical"}:
        return False
    if target_label == "recharge":
        estimated_steps = _estimate_grid_steps(avatar_centroid, target_centroid)
        return estimated_steps is not None and estimated_steps > max(1, remaining_units - 1)
    if bucket != "critical":
        return False
    estimated_steps = _estimate_grid_steps(avatar_centroid, target_centroid)
    if estimated_steps is None:
        return remaining_units <= 4
    return estimated_steps > max(1, remaining_units - 2)


def _available_action_indices(available_actions: list[Any] | None) -> list[int]:
    if not isinstance(available_actions, list):
        return []
    numeric_values: list[int] = []
    for item in available_actions:
        if isinstance(item, int):
            numeric_values.append(int(item))
        elif isinstance(item, str) and str(item).strip().isdigit():
            numeric_values.append(int(str(item).strip()))
    # Live ARC3 server exposes 1-based ids (1..7); local ARC3 benchmark uses 0-based ids (0..6).
    use_one_based = bool(numeric_values) and 0 not in numeric_values and all(
        1 <= value <= len(ACTION_NAMES) for value in numeric_values
    )

    indices: list[int] = []
    seen: set[int] = set()
    for item in available_actions:
        resolved: int | None = None
        if isinstance(item, int):
            numeric = int(item)
            if use_one_based and 1 <= numeric <= len(ACTION_NAMES):
                resolved = numeric - 1
            elif 0 <= numeric < len(ACTION_NAMES):
                resolved = numeric
        elif isinstance(item, str):
            text = str(item).strip()
            if text in ACTION_NAMES:
                resolved = ACTION_NAMES.index(text)
            elif text in ACTION_LABELS:
                resolved = ACTION_LABELS.index(text)
            elif text.isdigit():
                numeric = int(text)
                if use_one_based and 1 <= numeric <= len(ACTION_NAMES):
                    resolved = numeric - 1
                elif 0 <= numeric < len(ACTION_NAMES):
                    resolved = numeric
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        indices.append(resolved)
    return indices


def _frame_to_query_text(
    frame: list[list[int]],
    goal_frame: list[list[int]] | None,
    available_actions: list[Any] | None = None,
    *,
    frame_state: str = "gameplay",
    fresh_context: bool = False,
    budget_snapshot: dict[str, Any] | None = None,
    lives_remaining: int | None = None,
    reference_box_visible: bool = False,
    flash_semantics: str = "",
    avatar_centroid: tuple[float, float] | None = None,
    visible_objects: list[dict[str, Any]] | None = None,
    episode_context: dict[str, Any] | None = None,
    stuck_signal: bool = False,
    centroid_drift: float = 0.0,
) -> str:
    normalized_goal = _normalize_grid(goal_frame) if goal_frame is not None else [[]]
    rows = len(frame)
    cols = len(frame[0]) if rows and isinstance(frame[0], list) else 0
    goal_state = "goal present" if normalized_goal != [[]] else "goal absent"
    current_centroid = avatar_centroid
    if current_centroid is None and frame_state != "transition":
        current_centroid = _avatar_centroid(frame) or _focus_centroid(frame)
    state_tokens: list[str] = [f"arc3 game frame {rows}x{cols}", goal_state]
    background = _background_value(frame) if rows and cols else 0
    state_tokens.append(f"background color {int(background)}")
    if frame_state == "transition":
        if flash_semantics == "failure":
            state_tokens.extend(
                [
                    "screen flash failure",
                    "yellow flash failure",
                    "movement budget depletion penalty",
                ]
            )
        elif flash_semantics == "success":
            state_tokens.extend(
                [
                    "screen transition uniform color",
                    "green flash success",
                    "level progression success",
                ]
            )
        else:
            state_tokens.extend(
                [
                    "screen transition uniform color",
                    "transition animation continue",
                ]
            )
    elif fresh_context:
        state_tokens.extend(
            [
                "post transition new context",
                "re perceive fresh layout",
                "new level gameplay",
            ]
        )
    if current_centroid is not None and frame_state != "transition":
        state_tokens.append(
            f"avatar at row {int(round(float(current_centroid[0])))} col {int(round(float(current_centroid[1])))}"
        )
    if budget_snapshot:
        fraction = budget_snapshot.get("fraction")
        try:
            state_tokens.append(f"movement budget {float(fraction) * 100.0:.0f} percent remaining")
        except Exception:
            pass
    if lives_remaining is not None:
        lives_word = {0: "zero", 1: "one", 2: "two", 3: "three"}.get(int(lives_remaining), str(int(lives_remaining)))
        state_tokens.extend(
            [
                "lives system",
                "lives visual indicator",
                f"{lives_word} lives remaining",
            ]
        )
    if reference_box_visible:
        state_tokens.append("reference box current state visible")
    if current_centroid is not None and frame_state != "transition":
        state_tokens.extend(["avatar identity", "walkable surface"])
    if centroid_drift >= 0.0:
        state_tokens.append(f"centroid drift {float(centroid_drift):.1f}")
    object_tokens: list[str] = []
    for obj in list(visible_objects or [])[:8]:
        color = int(obj.get("color", -1))
        if color < 0:
            continue
        centroid = obj.get("centroid")
        row = col = -1
        if isinstance(centroid, (tuple, list)) and len(centroid) == 2:
            row = int(round(float(centroid[0])))
            col = int(round(float(centroid[1])))
        size = int(obj.get("size", 0) or 0)
        semantic_hint = str(obj.get("semantic_hint") or obj.get("behavior") or "").strip().lower()
        tested = bool(obj.get("interaction_tested", False))
        object_tokens.append(
            " ".join(
                token
                for token in (
                    "visible object",
                    f"color {color}",
                    f"row {row}" if row >= 0 else "",
                    f"col {col}" if col >= 0 else "",
                    f"size {size}" if size > 0 else "",
                    semantic_hint,
                    "tested" if tested else "untested",
                )
                if token
            )
        )
    if current_centroid is not None and frame_state != "transition" and rows > 0 and cols > 0:
        row = int(round(float(current_centroid[0])))
        col = int(round(float(current_centroid[1])))
        neighbors = {
            "up": (row - 1, col),
            "down": (row + 1, col),
            "left": (row, col - 1),
            "right": (row, col + 1),
        }
        adjacent_tokens: list[str] = []
        adjacent_object_tokens: list[str] = []
        known_objects = {
            (int(obj.get("color", -1)), int(round(float(obj["centroid"][0]))), int(round(float(obj["centroid"][1])))): obj
            for obj in list(visible_objects or [])
            if isinstance(obj.get("centroid"), (tuple, list)) and len(obj["centroid"]) == 2
        }
        for direction, (next_row, next_col) in neighbors.items():
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            color = int(frame[next_row][next_col])
            adjacent_tokens.append(f"{direction} {color}")
            for obj in list(visible_objects or []):
                centroid = obj.get("centroid")
                if not (isinstance(centroid, (tuple, list)) and len(centroid) == 2):
                    continue
                obj_row = int(round(float(centroid[0])))
                obj_col = int(round(float(centroid[1])))
                distance = abs(obj_row - row) + abs(obj_col - col)
                if distance != 1:
                    continue
                adjacent_object_tokens.append(
                    " ".join(
                        token
                        for token in (
                            "object adjacent to avatar",
                            f"color {int(obj.get('color', -1))}",
                            f"distance {distance}",
                            "tested" if bool(obj.get("interaction_tested", False)) else "untested",
                            str(obj.get("semantic_hint") or obj.get("behavior") or "").strip().lower(),
                        )
                        if token
                    )
                )
        if adjacent_tokens:
            state_tokens.append("adjacent cells " + " ".join(adjacent_tokens))
        state_tokens.extend(adjacent_object_tokens)
    action_tokens: list[str] = []
    for action_index in _available_action_indices(available_actions):
        action_tokens.append(ACTION_NAMES[int(action_index)].lower())
    actions_text = " ".join(action_tokens) if action_tokens else "actions unknown"
    position_text = " ".join(state_tokens + object_tokens)
    return (
        f"{position_text} "
        f"available actions {actions_text} "
        "levels navigation visual"
    ).strip()


def _result_has_direct_action(result: dict[str, Any] | None) -> bool:
    packet = _normalized_runtime_result(result)
    if not packet:
        return False
    raw_action_name = str(packet.get("action_name", "")).strip().upper()
    has_click_payload = False
    action_input = packet.get("action_input")
    if isinstance(action_input, dict) and isinstance(action_input.get("x"), (int, float)) and isinstance(action_input.get("y"), (int, float)):
        has_click_payload = True
    elif isinstance(packet.get("x"), (int, float)) and isinstance(packet.get("y"), (int, float)):
        has_click_payload = True
    if raw_action_name == RESET_ACTION_NAME:
        return True
    if raw_action_name in ACTION_NAMES:
        return raw_action_name != "ACTION6" or has_click_payload
    raw_action_index = packet.get("action_index")
    if isinstance(raw_action_index, int) and -1 <= int(raw_action_index) < len(ACTION_NAMES):
        return int(raw_action_index) != 5 or has_click_payload
    if has_click_payload:
        return True
    return _normalize_grid(packet.get("output_grid")) != [[]]


def _normalized_runtime_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    task_result = result.get("task_result")
    if not isinstance(task_result, dict):
        return dict(result)
    merged = dict(task_result)
    for key, value in result.items():
        if key == "task_result":
            continue
        merged.setdefault(key, value)
    route_payload = task_result.get("route")
    if not isinstance(route_payload, dict) and isinstance(result.get("route"), dict):
        merged["route"] = dict(result["route"])
    return merged


def _derive_action_from_result(
    frame: list[list[int]],
    result: dict[str, Any],
    *,
    goal_frame: list[list[int]] | None = None,
) -> tuple[int | str | None, dict[str, int]]:
    packet = _normalized_runtime_result(result)
    action_input = packet.get("action_input")
    click_payload = (
        {"x": int(action_input["x"]), "y": int(action_input["y"])}
        if isinstance(action_input, dict)
        and isinstance(action_input.get("x"), (int, float))
        and isinstance(action_input.get("y"), (int, float))
        else {}
    )
    raw_action_name = str(packet.get("action_name", "")).strip().upper()
    if raw_action_name == RESET_ACTION_NAME:
        return RESET_ACTION_NAME, {}
    if raw_action_name in ACTION_NAMES:
        if raw_action_name == "ACTION6" and not click_payload and not (
            isinstance(packet.get("x"), (int, float)) and isinstance(packet.get("y"), (int, float))
        ):
            return None, {}
        return ACTION_NAMES.index(raw_action_name), dict(click_payload)

    raw_action_index = packet.get("action_index")
    if isinstance(raw_action_index, int):
        if int(raw_action_index) < 0:
            return RESET_ACTION_NAME, {}
        if int(raw_action_index) < len(ACTION_NAMES):
            if int(raw_action_index) == 5 and not click_payload and not (
                isinstance(packet.get("x"), (int, float)) and isinstance(packet.get("y"), (int, float))
            ):
                return None, {}
            return int(raw_action_index), dict(click_payload)

    if click_payload:
        return 5, dict(click_payload)

    if isinstance(packet.get("x"), (int, float)) and isinstance(packet.get("y"), (int, float)):
        return 5, {"x": int(packet["x"]), "y": int(packet["y"])}

    return None, {}


class K3DARC3Agent:
    """ARC-AGI-3 agent — thin I/O wrapper over Knowledgeverse.execute_task()."""

    def __init__(
        self,
        max_actions: int = 10000,
        log_path: str | Path | None = None,
        knowledgeverse: Knowledgeverse | None = None,
    ) -> None:
        self.max_actions = int(max_actions)
        self.log_path = Path(log_path) if log_path else None
        self.kv = knowledgeverse or Knowledgeverse()
        self.tablet_boundary = self._build_tablet_boundary()
        self.action_history: list[dict[str, Any]] = []
        self._last_levels_completed = 0
        self._last_frame: list[list[int]] | None = None
        self._needs_reperceive = False
        self._attempt_actions = 0
        self._step_count = 0
        self._visited_cells: set[tuple[int, int]] = set()
        self._known_objects: dict[tuple[int, str], dict[str, Any]] = {}
        self._frame_color_diagnostics: list[dict[str, Any]] = []
        self._color_diagnostics_games: set[str] = set()
        self._game_id = "unknown"
        self._frame_encoder = ARC3FrameEncoder()
        self._episode_galaxy = ARC3EpisodeGalaxy(game_id=self._game_id, knowledgeverse=self.kv)
        if isinstance(self.kv, Knowledgeverse):
            self.kv.ensure_default_galaxies_loaded()
        self._game_mechanics_star_count = int(seed_game_mechanics(self.kv))

    def _build_tablet_boundary(self) -> HeadlessTabletMPC:
        if isinstance(self.kv, Knowledgeverse):
            return HeadlessTabletMPC(knowledgeverse=self.kv)

        def _handler(payload: dict[str, Any]) -> dict[str, Any]:
            task = dict(payload.get("task") or {})
            route: dict[str, Any] = {
                "specialist": str(payload.get("specialist") or "visual"),
                "domain_hint": str(payload.get("domain_hint") or task.get("domain_hint") or "game_2d"),
            }
            solved = self.kv.execute_task(
                task=task,
                route=route,
                specialist=str(route["specialist"]),
                domain_hint=str(route["domain_hint"]),
                use_enriched=bool(payload.get("use_enriched", True)),
            )
            return {"status": "ok", "route": route, "task_result": dict(solved or {})}

        return HeadlessTabletMPC(command_handler=_handler)

    def reset_attempt_state(self) -> None:
        """Reset per-attempt shell state while keeping episode memory intact."""
        self._last_levels_completed = 0
        self._last_frame = None
        self._needs_reperceive = False
        self._attempt_actions = 0

    def _record_visited_cell(self, avatar_centroid: tuple[float, float] | None) -> None:
        if avatar_centroid is None:
            return
        self._visited_cells.add(
            (
                int(round(float(avatar_centroid[0]))),
                int(round(float(avatar_centroid[1]))),
            )
        )

    def _diagnose_frame_colors_once(self, grid: list[list[int]], *, game_id: str) -> None:
        game_key = str(game_id or "unknown")
        if game_key in self._color_diagnostics_games:
            return
        diagnostics = _diagnose_frame_colors(grid)
        if not diagnostics:
            return
        self._color_diagnostics_games.add(game_key)
        self._frame_color_diagnostics = list(diagnostics)
        print(f"[ARC3] gameplay color diagnostics for {game_key}:")
        for row in diagnostics:
            centroid = row["centroid"]
            print(
                "[ARC3] "
                f"color={int(row['color'])} pixels={int(row['pixels'])} "
                f"centroid=({float(centroid[0]):.1f}, {float(centroid[1]):.1f}) "
                f"position={row['position_label']} bbox={tuple(row['bbox'])}"
            )

    def _merge_visible_objects(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for obj in list(objects or []):
            color = int(obj.get("color", -1))
            position = str(obj.get("approximate_position") or obj.get("position_label") or "").strip()
            if color < 0 or not position:
                continue
            key = (color, position)
            record = self._known_objects.setdefault(
                key,
                {
                    "color": int(color),
                    "approximate_position": position,
                    "position_label": position,
                    "interaction_tested": False,
                    "semantic_hint": str(obj.get("semantic_hint") or ""),
                    "behavior": str(obj.get("behavior") or "unknown"),
                },
            )
            record["color"] = int(color)
            record["position_label"] = str(obj.get("position_label") or position)
            record["approximate_position"] = position
            if str(obj.get("semantic_hint") or "").strip():
                record["semantic_hint"] = str(obj.get("semantic_hint") or "").strip()
            if str(obj.get("behavior") or "").strip():
                record["behavior"] = str(obj.get("behavior") or "").strip()
            if "centroid" in obj:
                record["centroid"] = tuple(obj.get("centroid") or ())
            if "size" in obj:
                record["size"] = int(obj.get("size", 0) or 0)
            if "height" in obj:
                record["height"] = int(obj.get("height", 0) or 0)
            if "width" in obj:
                record["width"] = int(obj.get("width", 0) or 0)
            if "points" in obj:
                record["points"] = tuple(obj.get("points") or ())
            if "interaction_result" in obj and obj.get("interaction_result") is not None:
                record["interaction_result"] = obj.get("interaction_result")
            if "interaction_tested" in obj:
                record["interaction_tested"] = bool(obj.get("interaction_tested"))
            merged_obj = dict(record)
            merged_obj["moves_with_avatar"] = bool(obj.get("moves_with_avatar", False))
            merged.append(merged_obj)
            self._episode_galaxy.seed_object(merged_obj)
        return merged

    def get_episode_context(self) -> dict[str, Any]:
        return self._episode_galaxy.get_episode_context(last_n=10)

    def run_deep_consolidation(self) -> dict[str, Any]:
        return self._episode_galaxy.run_deep_consolidation()

    def choose_action(
        self,
        frame: list[list[int]],
        *,
        goal_frame: list[list[int]] | None = None,
        task_data: dict[str, Any] | None = None,
        available_actions: list[Any] | None = None,
        game_id: str | None = None,
        levels_completed: int = 0,
        episode_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate ARC-3 state through the canonical GAME_2D tablet/WINE boundary."""
        normalized_frame = _normalize_grid(frame)
        normalized_goal = _normalize_grid(goal_frame) if goal_frame is not None else [[]]
        gameplay_frame = _gameplay_grid(normalized_frame)
        gameplay_goal = _gameplay_grid(normalized_goal) if normalized_goal != [[]] else [[]]
        frame_state = _frame_state(normalized_frame)
        flash_semantics = _flash_semantics(normalized_frame)
        budget_snapshot = _movement_budget_snapshot(normalized_frame)
        lives_remaining = _lives_remaining(normalized_frame)
        reference_box_visible = _reference_box_visible(normalized_frame)
        task_context = dict(task_data or {}) if isinstance(task_data, dict) else {}
        resolved_game_id = str(game_id or self._game_id or "unknown")
        self._game_id = resolved_game_id
        if self._episode_galaxy.game_id != resolved_game_id:
            self._episode_galaxy.game_id = resolved_game_id
        world_model = task_context.get("world_model")
        avatar_centroid = None if frame_state == "transition" else (_avatar_centroid(normalized_frame) or _focus_centroid(normalized_frame))
        self._record_visited_cell(avatar_centroid)
        if frame_state == "gameplay":
            self._diagnose_frame_colors_once(normalized_frame, game_id=resolved_game_id)
        visible_objects = (
            self._merge_visible_objects(
                _detect_static_objects(
                    normalized_frame,
                    avatar_centroid,
                    known_objects=self._known_objects,
                )
            )
            if frame_state == "gameplay"
            else []
        )
        valid_action_indices = _available_action_indices(available_actions)
        centroid_drift = 0.0
        recent_centroids: list[tuple[float, float]] = []
        if avatar_centroid is not None and len(self.action_history) >= 5:
            for previous_record in self.action_history[-5:]:
                rc = previous_record.get("avatar_centroid")
                if isinstance(rc, (tuple, list)) and len(rc) == 2:
                    recent_centroids.append((float(rc[0]), float(rc[1])))
            if len(recent_centroids) >= 3:
                centroid_drift = max(
                    abs(float(rc[0]) - float(avatar_centroid[0])) + abs(float(rc[1]) - float(avatar_centroid[1]))
                    for rc in recent_centroids
                )
        stuck_signal = bool(avatar_centroid is not None and len(self.action_history) >= 5 and centroid_drift < 2.0)
        fresh_context = False
        if self._needs_reperceive:
            if frame_state != "transition":
                fresh_context = True
        envelope = arc3_game_envelope(
            task_id=str(
                task_context.get("task_id")
                or task_context.get("id")
                or f"arc3_live_{len(self.action_history) + 1:04d}"
            ),
            frame=normalized_frame,
            goal_frame=normalized_goal if normalized_goal != [[]] else None,
            available_actions=list(available_actions or []),
            action_options=list(ACTION_NAMES),
            training_examples=list(task_context.get("train") or []),
            query=_frame_to_query_text(
                normalized_frame,
                normalized_goal,
                available_actions=available_actions,
                frame_state=frame_state,
                fresh_context=fresh_context,
                budget_snapshot=budget_snapshot,
                lives_remaining=lives_remaining,
                reference_box_visible=reference_box_visible,
                flash_semantics=flash_semantics,
                avatar_centroid=avatar_centroid,
                visible_objects=visible_objects,
                episode_context=dict(episode_context or {}),
                stuck_signal=stuck_signal,
                centroid_drift=centroid_drift,
            ),
            query_embedding=self._frame_encoder.encode(gameplay_frame),
            goal_embedding=(
                self._frame_encoder.encode(gameplay_goal)
                if gameplay_goal != [[]]
                else None
            ),
            step_count=int(self._step_count),
            game_id=resolved_game_id,
            levels_completed=int(levels_completed),
            world_model=dict(world_model or {}) if isinstance(world_model, dict) else {},
            episode_context=dict(episode_context or {}),
            task_context_extras={
                "stuck_signal": bool(stuck_signal),
                "centroid_drift": float(centroid_drift),
            },
            metadata={
                "task_data": task_context,
                "game_id": resolved_game_id,
                "levels_completed": int(levels_completed),
            },
        )
        try:
            raw_tablet_result = self.tablet_boundary.submit(envelope, use_enriched=True)
            tablet_result = dict(raw_tablet_result or {})
            response_packet = dict(tablet_result.get("response") or {})
            emitted = dict(tablet_result.get("emitted") or response_packet or {})
            task_result = dict(emitted.get("task_result") or response_packet.get("task_result") or {})
        except Exception as exc:
            raise RuntimeError(f"arc3_tablet_boundary_failed:{exc}") from exc
        game_action = dict(emitted.get("game_action") or {}) if isinstance(emitted.get("game_action"), dict) else {}
        action_input = dict(game_action.get("action_input") or {}) if isinstance(game_action.get("action_input"), dict) else {}
        runtime_result = _normalized_runtime_result(
            {
                **dict(response_packet or {}),
                **{key: value for key, value in dict(emitted or {}).items() if key != "task_result"},
                **(
                    {"action_input": dict(action_input)}
                    if action_input
                    else {}
                ),
                **(
                    {"action_index": game_action.get("action_index")}
                    if game_action.get("action_index") is not None
                    else {}
                ),
                **(
                    {"action_name": game_action.get("action_name")}
                    if str(game_action.get("action_name") or "").strip()
                    else {}
                ),
                "task_result": dict(task_result or {}),
            }
        )
        action_choice, payload = _derive_action_from_result(
            normalized_frame,
            dict(runtime_result or {}),
            goal_frame=normalized_goal,
        )
        if action_choice is None:
            raise RuntimeError("arc3_sovereign_action_not_materialized")
        click_reason = ""
        strategic_reset = False
        if action_choice == RESET_ACTION_NAME:
            if valid_action_indices and 6 not in set(valid_action_indices):
                raise RuntimeError("arc3_reset_not_available")
            payload = {}
            click_reason = "tablet_boundary_reset"
            self._needs_reperceive = True
            strategic_reset = True
            action_index = 6
            action_name = ACTION_NAMES[action_index]
            action_label = ACTION_LABELS[action_index]
        else:
            action_index = int(action_choice)
            if valid_action_indices and action_index not in set(valid_action_indices):
                raise RuntimeError(f"arc3_action_not_available:{action_index}")
            action_name = ACTION_NAMES[action_index]
            action_label = ACTION_LABELS[action_index]
        if action_name == "ACTION6":
            if not {"x", "y"} <= set(payload):
                raise RuntimeError("arc3_click_payload_missing")
            click_reason = "tablet_boundary_click"
        else:
            payload = {}
        match = (runtime_result or {}).get("match") if isinstance((runtime_result or {}).get("match"), dict) else {}
        jarvis_brief = (
            (runtime_result or {}).get("jarvis_brief")
            if isinstance((runtime_result or {}).get("jarvis_brief"), dict)
            else {}
        )
        record = {
            "action": action_name,
            "action_index": int(action_index),
            "label": action_label,
            "confidence": float((runtime_result or {}).get("confidence", (runtime_result or {}).get("similarity", 0.0))),
            "converged": int((runtime_result or {}).get("convergence_signal", (runtime_result or {}).get("converged", 0))),
            "iterations_used": int((runtime_result or {}).get("iterations_used", 0)),
            "frame_number": len(self.action_history) + 1,
            "gpu_execution": bool((runtime_result or {}).get("gpu_execution", False)),
            "solver": str((runtime_result or {}).get("solver", "tablet_boundary")),
            "task_result": dict(runtime_result or {}),
            "available_actions": list(available_actions or []),
            "click_reason": click_reason,
            "frame_state": frame_state,
            "fresh_context": fresh_context,
            "game_id": resolved_game_id,
            "levels_completed": int(levels_completed),
            "movement_budget": dict(budget_snapshot or {}),
            "lives_remaining": lives_remaining,
            "reference_box_visible": bool(reference_box_visible),
            "flash_semantics": flash_semantics,
            "attempt_actions": int(self._attempt_actions),
            "frame_unchanged": bool(self._last_frame is not None and _same_gameplay_state(normalized_frame, self._last_frame)),
            "blocked_actions": [],
            "matched_star_id": str(match.get("id", "")),
            "matched_star_galaxy": str(match.get("galaxy", "")),
            "teacher_route_galaxies": list((runtime_result or {}).get("teacher_route_galaxies") or []),
            "winning_program_id": str((runtime_result or {}).get("winning_program_id", "")),
            "galaxy_contribution": dict((runtime_result or {}).get("galaxy_contribution") or {}),
            "jarvis_worker_count": int(jarvis_brief.get("worker_count", 0) or 0),
            "jarvis_planned_swarm_groups": int(jarvis_brief.get("planned_swarm_groups", 0) or 0),
            "spatial_plan_target": "",
            "spatial_plan_path_length": 0,
            "spatial_plan_fallback": False,
            "visible_object_count": len(list(visible_objects or [])),
            "visited_cells_count": len(self._visited_cells),
            "frame_color_diagnostics": list(self._frame_color_diagnostics),
            "route_family": str((runtime_result or {}).get("route_family") or emitted.get("route_family") or ""),
            "failure_code": str((runtime_result or {}).get("failure_code") or emitted.get("failure_code") or ""),
            "actual_result_kind": str((runtime_result or {}).get("actual_result_kind") or emitted.get("actual_result_kind") or ""),
            "tablet_contract": tablet_result.get("tablet_contract"),
            "step_count": int(self._step_count),
            "episode_object_count": len(dict((episode_context or {}).get("objects") or {})),
            "strategic_reset": bool(strategic_reset),
            "avatar_centroid": avatar_centroid,
            "stuck_signal": bool(stuck_signal),
            "centroid_drift": float(centroid_drift),
            "direct_action_materialized": True,
            **payload,
        }
        self.action_history.append(record)
        self._last_frame = _clone_grid(normalized_frame)
        if fresh_context:
            self._needs_reperceive = False
        if strategic_reset:
            self._attempt_actions = 0
        else:
            self._attempt_actions += 1
        self._step_count += 1
        return record

    def learn_from_outcome(
        self,
        *,
        levels_completed: int = 0,
        frame: list[list[int]] | None = None,
        action: str = "",
        prev_frame: list[list[int]] | None = None,
        reward: float = 0.0,
        lives_delta: int = 0,
        levels_delta: int = 0,
    ) -> int:
        """Record lightweight outcome metadata; Knowledgeverse owns consolidation."""
        current = max(0, int(levels_completed))
        normalized_frame = _normalize_grid(frame) if frame is not None else None
        normalized_prev = _normalize_grid(prev_frame) if prev_frame is not None else None
        if current > self._last_levels_completed:
            outcome = 1
            self._needs_reperceive = True
            self._attempt_actions = 0
        elif normalized_frame is not None and self._last_frame is not None and not _same_gameplay_state(normalized_frame, self._last_frame):
            outcome = 0
        else:
            outcome = -1
        if normalized_frame is not None and normalized_prev is not None:
            avatar_centroid = _avatar_centroid(normalized_frame) or _focus_centroid(normalized_frame)
            prev_avatar_centroid = _avatar_centroid(normalized_prev) or _focus_centroid(normalized_prev)
            self._record_visited_cell(avatar_centroid)
            for obj in _detect_static_objects(
                normalized_frame,
                avatar_centroid,
                known_objects=self._known_objects,
            ):
                self._merge_visible_objects([obj])
            self._episode_galaxy.seed_outcome(
                step_count=max(0, int(self._step_count) - 1),
                action=str(action or ""),
                prev_grid=normalized_prev,
                next_grid=normalized_frame,
                reward=float(reward),
                lives_delta=int(lives_delta),
                levels_delta=int(levels_delta),
            )
            action_name = str(action or "").strip().upper()
            if (
                action_name == "ACTION5"
                and not _same_gameplay_state(normalized_prev, normalized_frame)
            ):
                reference_centroid = prev_avatar_centroid or avatar_centroid
                if reference_centroid is not None:
                    for obj_key, obj in self._known_objects.items():
                        if bool(obj.get("interaction_tested", False)):
                            continue
                        centroid = obj.get("centroid")
                        if not (isinstance(centroid, (tuple, list)) and len(centroid) == 2):
                            continue
                        distance = (
                            abs(float(centroid[0]) - float(reference_centroid[0]))
                            + abs(float(centroid[1]) - float(reference_centroid[1]))
                        )
                        if distance > 4.0:
                            continue
                        obj["interaction_tested"] = True
                        obj["interaction_result"] = "frame_changed"
                        self._episode_galaxy.seed_object(dict(obj))
                        self._episode_galaxy.seed_rule(
                            state=f"agent_near_color_{int(obj['color'])}_object",
                            action=4,
                            outcome="frame_changed",
                            confidence=0.9,
                            evidence_count=3,
                        )
                        break
            self._episode_galaxy.run_micro_sleeptime()
        if hasattr(self.kv, "record_outcome"):
            self.kv.record_outcome(outcome)
        self._last_levels_completed = current
        if self.action_history:
            self.action_history[-1]["outcome_signal"] = int(outcome)
            self.action_history[-1]["levels_completed"] = current
            self.action_history[-1]["reward"] = float(reward)
            self.action_history[-1]["lives_delta"] = int(lives_delta)
            self.action_history[-1]["levels_delta"] = int(levels_delta)
            self.action_history[-1]["micro_sleeptime_scheduled"] = normalized_frame is not None and normalized_prev is not None
        return int(outcome)

    def close(self) -> None:
        self._episode_galaxy.close()
        if self.log_path and self.action_history:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in self.action_history))
                handle.write("\n")

__all__ = ["ACTION_LABELS", "ACTION_NAMES", "ARC3_ROUTE_GALAXIES", "K3DARC3Agent"]
