"""K3D sovereign ARC-AGI-3 agent — thin I/O adapter over Knowledgeverse."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]
RESET_ACTION_NAME = "RESET"
RESET_ACTION_LABEL = "Reset"
ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]
SPATIAL_WALKABLE_COLORS = {0, 1, 3, 9, 11, 12, 15}


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


def _select_mechanic_target(
    grid: list[list[int]],
    avatar_centroid: tuple[float, float] | None,
    *,
    budget_snapshot: dict[str, Any] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    if avatar_centroid is None:
        return None, ""
    avatar_row, avatar_col = avatar_centroid
    target_specs = [
        ("switch", {11, 15}, 4, 16, 0),
        ("door", {9}, 4, 32, 1),
        ("recharge", {12}, 4, 16, 2),
    ]
    if budget_snapshot and str(budget_snapshot.get("bucket", "")) in {"low", "critical"}:
        target_specs = [
            ("recharge", {12}, 4, 16, 0),
            ("switch", {11, 15}, 4, 16, 1),
            ("door", {9}, 4, 32, 2),
        ]
    candidates: list[tuple[int, float, float, tuple[float, float], str]] = []
    for label, colors, min_size, max_size, priority in target_specs:
        for component in _components_for_colors(grid, colors):
            size = int(component["size"])
            if size < min_size or size > max_size:
                continue
            centroid = (
                float(component["centroid"][0]),
                float(component["centroid"][1]),
            )
            distance = abs(centroid[0] - avatar_row) + abs(centroid[1] - avatar_col)
            ideal_size = (min_size + max_size) / 2.0
            size_penalty = abs(float(size) - ideal_size)
            candidates.append((priority, distance, size_penalty, centroid, label))
    if not candidates:
        return None, ""
    _, _, _, centroid, label = min(candidates)
    return centroid, label


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


def _clamp_click_target(grid: list[list[int]], x: int, y: int) -> dict[str, int]:
    if not grid or not grid[0]:
        return {"x": 0, "y": 0}
    height = len(grid)
    width = len(grid[0])
    return {
        "x": max(0, min(int(x), width - 1)),
        "y": max(0, min(int(y), height - 1)),
    }


def _salient_click_centers(grid: list[list[int]]) -> list[dict[str, int]]:
    if not grid or not grid[0]:
        return [{"x": 0, "y": 0}]

    candidates: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(x: int, y: int) -> None:
        payload = _clamp_click_target(grid, x, y)
        key = (int(payload["x"]), int(payload["y"]))
        if key in seen:
            return
        seen.add(key)
        candidates.append(payload)

    height = len(grid)
    width = len(grid[0])

    # Live ARC3 start screens expose a unique color-6 focus cell inside the Start button cluster.
    focus_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if int(value) == 6
    ]
    if len(focus_cells) == 1:
        row_index, col_index = focus_cells[0]
        _add(col_index, row_index)

    lower_magic_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if int(value) in {6, 15} and row_index >= max(0, height // 2)
    ]
    if lower_magic_cells:
        avg_row = sum(row_index for row_index, _ in lower_magic_cells) / float(len(lower_magic_cells))
        avg_col = sum(col_index for _, col_index in lower_magic_cells) / float(len(lower_magic_cells))
        _add(int(round(avg_col)), int(round(avg_row)))

    background = _background_value(grid)
    non_background_counts: Counter[int] = Counter(
        int(value) for row in grid for value in row if int(value) != int(background)
    )
    if non_background_counts:
        rarest = min(non_background_counts.values())
        rare_colors = {color for color, count in non_background_counts.items() if count == rarest}
        rare_cells = [
            (row_index, col_index)
            for row_index, row in enumerate(grid)
            for col_index, value in enumerate(row)
            if int(value) in rare_colors
        ]
        if rare_cells:
            avg_row = sum(row_index for row_index, _ in rare_cells) / float(len(rare_cells))
            avg_col = sum(col_index for _, col_index in rare_cells) / float(len(rare_cells))
            _add(int(round(avg_col)), int(round(avg_row)))

    preferred_colors = [3, 15, 1, 0]
    for preferred in preferred_colors:
        cells = [
            (row_index, col_index)
            for row_index, row in enumerate(grid)
            for col_index, value in enumerate(row)
            if int(value) == preferred
        ]
        if not cells:
            continue
        avg_row = sum(row_index for row_index, _ in cells) / float(len(cells))
        avg_col = sum(col_index for _, col_index in cells) / float(len(cells))
        _add(int(round(avg_col)), int(round(avg_row)))

    _add(width // 2, height // 2)

    return candidates or [{"x": 0, "y": 0}]


def _tracked_click_target(grid: list[list[int]]) -> dict[str, int]:
    return dict(_salient_click_centers(grid)[0])


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


def _movement_action_indices(valid_action_indices: list[int] | None) -> list[int]:
    if valid_action_indices is None:
        return [0, 1, 2, 3]
    movement = [int(index) for index in list(valid_action_indices) if 0 <= int(index) <= 3]
    return movement


def _exploration_order(last_action_index: int | None) -> list[int]:
    if int(last_action_index or -1) == 0:
        return [2, 3, 1]
    if int(last_action_index or -1) == 1:
        return [2, 3, 0]
    if int(last_action_index or -1) == 2:
        return [0, 1, 3]
    if int(last_action_index or -1) == 3:
        return [0, 1, 2]
    return [0, 1, 2, 3]


def _navigation_state_key(grid: list[list[int]], *, levels_completed: int) -> tuple[int, int, int]:
    centroid = _avatar_centroid(grid) or _focus_centroid(grid)
    if centroid is None:
        return int(levels_completed), -1, -1
    return (
        int(levels_completed),
        int(round(float(centroid[0]))),
        int(round(float(centroid[1]))),
    )


def _walkable_cells(grid: list[list[int]]) -> list[tuple[int, int]]:
    gameplay = _gameplay_grid(grid)
    return [
        (row_index, col_index)
        for row_index, row in enumerate(gameplay)
        for col_index, value in enumerate(row)
        if int(value) in SPATIAL_WALKABLE_COLORS
    ]


def _nearest_cell(
    point: tuple[float, float] | None,
    cells: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if point is None or not cells:
        return None
    row, col = float(point[0]), float(point[1])
    return min(
        cells,
        key=lambda cell: abs(float(cell[0]) - row) + abs(float(cell[1]) - col),
    )


def _component_goal_cells(
    component: dict[str, Any],
    cell_to_index: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    goals: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row_index, col_index in list(component.get("points") or []):
        for candidate in (
            (int(row_index), int(col_index)),
            (int(row_index) - 1, int(col_index)),
            (int(row_index) + 1, int(col_index)),
            (int(row_index), int(col_index) - 1),
            (int(row_index), int(col_index) + 1),
        ):
            if candidate in cell_to_index and candidate not in seen:
                seen.add(candidate)
                goals.append(candidate)
    return goals


def _decode_path_action(
    path_indices: list[int],
    cells_by_index: list[tuple[int, int]],
) -> int | None:
    if len(path_indices) < 2:
        return None
    start = cells_by_index[int(path_indices[0])]
    nxt = cells_by_index[int(path_indices[1])]
    delta_row = int(nxt[0]) - int(start[0])
    delta_col = int(nxt[1]) - int(start[1])
    if delta_row == -1 and delta_col == 0:
        return 0
    if delta_row == 1 and delta_col == 0:
        return 1
    if delta_row == 0 and delta_col == -1:
        return 2
    if delta_row == 0 and delta_col == 1:
        return 3
    return None


def _spatial_target_specs(budget_snapshot: dict[str, Any] | None) -> list[tuple[str, set[int], int, int, int]]:
    if budget_snapshot and str(budget_snapshot.get("bucket", "")) in {"low", "critical"}:
        return [
            ("recharge", {12}, 4, 24, 0),
            ("switch", {11, 15}, 4, 24, 1),
            ("door", {9}, 4, 64, 2),
        ]
    return [
        ("switch", {11, 15}, 4, 24, 0),
        ("door", {9}, 4, 64, 1),
        ("recharge", {12}, 4, 24, 2),
    ]


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
    force_reset: bool = False,
) -> str:
    normalized_goal = _normalize_grid(goal_frame) if goal_frame is not None else [[]]
    rows = len(frame)
    cols = len(frame[0]) if rows and isinstance(frame[0], list) else 0
    goal_state = "goal present" if normalized_goal != [[]] else "goal absent"
    position_tokens: list[str] = []
    guidance_tokens: list[str] = []
    state_tokens: list[str] = []
    current_centroid = None if frame_state == "transition" else (_avatar_centroid(frame) or _focus_centroid(frame))
    goal_centroid = _foreground_centroid(normalized_goal) if normalized_goal != [[]] else None
    derived_target_centroid: tuple[float, float] | None = None
    derived_target_label = ""
    if goal_centroid is None and current_centroid is not None and frame_state != "transition":
        derived_target_centroid, derived_target_label = _select_mechanic_target(
            frame,
            current_centroid,
            budget_snapshot=budget_snapshot,
        )
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
    if budget_snapshot:
        state_tokens.append("movement budget visual bar")
        bucket = str(budget_snapshot.get("bucket", ""))
        if bucket == "critical":
            state_tokens.extend(["movement budget critical", "budget sufficiency check"])
        elif bucket == "low":
            state_tokens.extend(["movement budget low", "movement budget conservation"])
        else:
            state_tokens.append("movement budget healthy")
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
    if derived_target_label:
        state_tokens.append(f"{derived_target_label} target visible")
        if derived_target_label == "door":
            state_tokens.append("target room visible")
    if current_centroid is not None and rows > 0 and cols > 0:
        avg_row, avg_col = current_centroid
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0
        row_margin = max(rows * 0.1, 0.5)
        col_margin = max(cols * 0.1, 0.5)
        primary_action: str | None = None
        secondary_action: str | None = None
        if force_reset:
            state_tokens.extend(
                [
                    "strategic reset",
                    "budget sufficiency check",
                    "preserve life before depletion",
                ]
            )
            position_tokens.append("budget insufficient reset now")
            primary_action = "action reset"
        target_centroid = goal_centroid or derived_target_centroid
        if force_reset:
            secondary_action = None
        elif target_centroid is not None:
            goal_row, goal_col = target_centroid
            row_delta = float(goal_row - avg_row)
            col_delta = float(goal_col - avg_col)
            if row_delta > row_margin:
                position_tokens.append("object above goal move down")
                guidance_tokens.append("action move down")
            elif row_delta < -row_margin:
                position_tokens.append("object below goal move up")
                guidance_tokens.append("action move up")
            if col_delta > col_margin:
                position_tokens.append("object left of goal move right")
                guidance_tokens.append("action move right")
            elif col_delta < -col_margin:
                position_tokens.append("object right of goal move left")
                guidance_tokens.append("action move left")
            if guidance_tokens:
                ordered_actions = list(guidance_tokens)
                if len(ordered_actions) > 1 and abs(row_delta) >= abs(col_delta):
                    if "action move down" in ordered_actions or "action move up" in ordered_actions:
                        primary_action = (
                            "action move down" if row_delta > row_margin else "action move up"
                        )
                elif len(ordered_actions) > 1 and abs(col_delta) > abs(row_delta):
                    if "action move right" in ordered_actions or "action move left" in ordered_actions:
                        primary_action = (
                            "action move right" if col_delta > col_margin else "action move left"
                        )
                if primary_action is None:
                    primary_action = ordered_actions[0]
                for action_token in ordered_actions:
                    if action_token != primary_action:
                        secondary_action = action_token
                        break
            else:
                position_tokens.append("object at goal perform")
                primary_action = "action perform"
        else:
            row_delta = float(center_row - avg_row)
            col_delta = float(center_col - avg_col)
            if row_delta > row_margin:
                position_tokens.append("object above center top north")
                guidance_tokens.append("action move down")
            elif row_delta < -row_margin:
                position_tokens.append("object below center bottom south")
                guidance_tokens.append("action move up")
            if col_delta > col_margin:
                position_tokens.append("object left of center west")
                guidance_tokens.append("action move right")
            elif col_delta < -col_margin:
                position_tokens.append("object right of center east")
                guidance_tokens.append("action move left")
            if guidance_tokens:
                ordered_actions = list(guidance_tokens)
                if len(ordered_actions) > 1 and abs(row_delta) >= abs(col_delta):
                    if "action move down" in ordered_actions or "action move up" in ordered_actions:
                        primary_action = (
                            "action move down" if row_delta > row_margin else "action move up"
                        )
                elif len(ordered_actions) > 1 and abs(col_delta) > abs(row_delta):
                    if "action move right" in ordered_actions or "action move left" in ordered_actions:
                        primary_action = (
                            "action move right" if col_delta > col_margin else "action move left"
                        )
                if primary_action is None:
                    primary_action = ordered_actions[0]
                for action_token in ordered_actions:
                    if action_token != primary_action:
                        secondary_action = action_token
                        break
            else:
                position_tokens.append("object centered balanced")
                primary_action = "action perform"
        if primary_action:
            guidance_tokens.insert(0, f"primary {primary_action}")
        if secondary_action:
            guidance_tokens.append(f"secondary {secondary_action}")
    action_tokens: list[str] = []
    for action_index in _available_action_indices(available_actions):
        action_tokens.append(ACTION_NAMES[int(action_index)].lower())
    actions_text = " ".join(action_tokens) if action_tokens else "actions unknown"
    position_text = " ".join(state_tokens + position_tokens + guidance_tokens)
    return (
        f"arc3 interactive game frame grid {rows}x{cols} "
        f"{position_text} {goal_state} "
        f"available actions {actions_text} "
        "levels navigation visual"
    ).strip()

def _derive_action_from_result(
    frame: list[list[int]],
    result: dict[str, Any],
    *,
    goal_frame: list[list[int]] | None = None,
) -> tuple[int | str, dict[str, int]]:
    raw_action_name = str(result.get("action_name", "")).strip().upper()
    if raw_action_name == RESET_ACTION_NAME:
        return RESET_ACTION_NAME, {}

    raw_answer_index = result.get("answer_index")
    if isinstance(raw_answer_index, (int, float)):
        action_index = max(0, min(int(raw_answer_index), len(ACTION_NAMES) - 1))
        return action_index, {}

    if isinstance(result.get("x"), (int, float)) and isinstance(result.get("y"), (int, float)):
        return 5, {"x": int(result["x"]), "y": int(result["y"])}

    predicted = _normalize_grid(result.get("output_grid"))
    if predicted != [[]]:
        def _first_active_cell(grid: list[list[int]]) -> tuple[int, int] | None:
            for row_index, row in enumerate(grid):
                for col_index, value in enumerate(row):
                    if int(value) != 0:
                        return row_index, col_index
            return None

        current_cell = _first_active_cell(frame)
        predicted_cell = _first_active_cell(predicted)
        if current_cell is not None and predicted_cell is not None:
            delta_row = int(predicted_cell[0]) - int(current_cell[0])
            delta_col = int(predicted_cell[1]) - int(current_cell[1])
            if abs(delta_row) > abs(delta_col):
                return (0 if delta_row < 0 else 1), {}
            if abs(delta_col) > 0:
                return (2 if delta_col < 0 else 3), {}
            if predicted != frame:
                return 4, {}
        if predicted != frame:
            return 4, {}

    return 0, {}


class K3DARC3Agent:
    """ARC-AGI-3 agent — thin I/O wrapper over Knowledgeverse.execute_task()."""

    def __init__(
        self,
        max_actions: int = 500,
        log_path: str | Path | None = None,
        knowledgeverse: Knowledgeverse | None = None,
    ) -> None:
        self.max_actions = int(max_actions)
        self.log_path = Path(log_path) if log_path else None
        self.kv = knowledgeverse or Knowledgeverse()
        self.action_history: list[dict[str, Any]] = []
        self._last_levels_completed = 0
        self._last_frame: list[list[int]] | None = None
        self._last_click_focus: tuple[int, int] | None = None
        self._click_focus_streak = 0
        self._click_probe_index = 1
        self._needs_reperceive = False
        self._attempt_actions = 0
        self._blocked_actions_by_state: dict[tuple[int, int, int], set[int]] = {}
        self._last_blocked_action: int | None = None
        self._blocked_repeat_count = 0
        self._frame_morton = None

    def _next_click_payload(self, grid: list[list[int]]) -> tuple[dict[str, int], str]:
        candidates = _salient_click_centers(grid)
        focus = candidates[0]
        focus_key = (int(focus["x"]), int(focus["y"]))
        if self._last_click_focus == focus_key:
            self._click_focus_streak += 1
        else:
            self._last_click_focus = focus_key
            self._click_focus_streak = 0
            self._click_probe_index = 1

        if self._click_focus_streak <= 1 or len(candidates) == 1:
            self._click_probe_index = 1
            return dict(focus), "tracked_focus"

        candidate_index = min(self._click_probe_index, len(candidates) - 1)
        payload = dict(candidates[candidate_index])
        if candidate_index < len(candidates) - 1:
            self._click_probe_index += 1
        return payload, f"tracked_focus_probe_{candidate_index}"

    def _exploration_fallback(
        self,
        *,
        valid_action_indices: list[int],
        blocked_actions: set[int],
        repeated_action: int | None,
    ) -> int:
        movement_candidates = _movement_action_indices(valid_action_indices)
        recent_actions = [
            int(row.get("action_index"))
            for row in self.action_history[-3:]
            if isinstance(row.get("action_index"), int) and 0 <= int(row.get("action_index")) <= 3
        ]
        ordered: list[int] = []
        for candidate in _exploration_order(repeated_action):
            if candidate in movement_candidates and candidate not in ordered:
                ordered.append(candidate)
        for candidate in movement_candidates:
            if candidate not in ordered:
                ordered.append(candidate)
        for candidate in ordered:
            if candidate in blocked_actions:
                continue
            if recent_actions[-3:] == [candidate, candidate, candidate]:
                continue
            return int(candidate)
        return int(ordered[0]) if ordered else int(valid_action_indices[0] if valid_action_indices else 0)

    def _spatial_path_plan(
        self,
        grid: list[list[int]],
        *,
        avatar_centroid: tuple[float, float] | None,
        budget_snapshot: dict[str, Any] | None,
        valid_action_indices: list[int],
    ) -> dict[str, Any] | None:
        if avatar_centroid is None or not hasattr(self.kv, "get_led_pathfinder"):
            return None
        pathfinder = self.kv.get_led_pathfinder()
        if not pathfinder:
            return None
        walkable_cells = _walkable_cells(grid)
        if not walkable_cells or len(walkable_cells) > 4096:
            return None
        ordered_cells = list(walkable_cells)
        try:
            if self._frame_morton is None:
                from knowledge3d.cranium.spatial_sovereign.morton_octree import MortonOctreeSovereign

                self._frame_morton = MortonOctreeSovereign()
            morton_points = [(float(col), float(row), 0.0) for row, col in walkable_cells]
            morton_codes = self._frame_morton.encode(morton_points)
            _, morton_order = self._frame_morton.sort(morton_codes, return_indices=True)
            ordered_cells = [walkable_cells[int(index)] for index in morton_order]
        except Exception:
            ordered_cells = list(walkable_cells)
        cell_to_index = {cell: index for index, cell in enumerate(ordered_cells)}
        start_cell = _nearest_cell(avatar_centroid, ordered_cells)
        if start_cell is None or start_cell not in cell_to_index:
            return None

        row_offsets = [0]
        col_indices: list[int] = []
        packed_costs: list[int] = []
        for row_index, col_index in ordered_cells:
            for neighbor in (
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ):
                neighbor_index = cell_to_index.get(neighbor)
                if neighbor_index is None:
                    continue
                col_indices.append(int(neighbor_index))
                packed_costs.append(1)
            row_offsets.append(len(col_indices))

        best_plan: dict[str, Any] | None = None
        gameplay = _gameplay_grid(grid)
        for label, colors, min_size, max_size, priority in _spatial_target_specs(budget_snapshot):
            components = _components_for_colors(gameplay, colors)
            for component in components:
                size = int(component["size"])
                if size < min_size or size > max_size:
                    continue
                goal_cells = _component_goal_cells(component, cell_to_index)
                if not goal_cells:
                    continue
                ranked_goals = sorted(
                    goal_cells,
                    key=lambda cell: abs(int(cell[0]) - int(start_cell[0])) + abs(int(cell[1]) - int(start_cell[1])),
                )[:4]
                for goal_cell in ranked_goals:
                    try:
                        path = pathfinder.navigate_csr(
                            row_offsets,
                            col_indices,
                            packed_costs,
                            start=int(cell_to_index[start_cell]),
                            goal=int(cell_to_index[goal_cell]),
                            max_path_length=256,
                        )
                    except Exception:
                        continue
                    path_indices = [int(index) for index in path]
                    action_index = _decode_path_action(path_indices, ordered_cells)
                    if action_index is None:
                        continue
                    if valid_action_indices and action_index not in set(valid_action_indices):
                        continue
                    plan = {
                        "action_index": int(action_index),
                        "confidence": float(1.0 / (1.0 + 0.08 * max(1, len(path_indices) - 1))),
                        "target_label": label,
                        "path_length": max(0, len(path_indices) - 1),
                        "program_type": "spatial_frame_pathfinder",
                        "solver": "spatial_frame_led_pathfinder",
                    }
                    if best_plan is None or (
                        int(priority),
                        int(plan["path_length"]),
                    ) < (
                        int(best_plan["priority"]),
                        int(best_plan["path_length"]),
                    ):
                        plan["priority"] = int(priority)
                        best_plan = plan
        if best_plan is None:
            return None
        best_plan.pop("priority", None)
        return best_plan

    def choose_action(
        self,
        frame: list[list[int]],
        *,
        goal_frame: list[list[int]] | None = None,
        task_data: dict[str, Any] | None = None,
        available_actions: list[Any] | None = None,
        game_id: str | None = None,
        levels_completed: int = 0,
    ) -> dict[str, Any]:
        """Translate frame → ARC_TASK → kv.execute_task() → ARC3 action dict."""
        normalized_frame = _normalize_grid(frame)
        normalized_goal = _normalize_grid(goal_frame) if goal_frame is not None else [[]]
        frame_state = _frame_state(normalized_frame)
        flash_semantics = _flash_semantics(normalized_frame)
        budget_snapshot = _movement_budget_snapshot(normalized_frame)
        lives_remaining = _lives_remaining(normalized_frame)
        reference_box_visible = _reference_box_visible(normalized_frame)
        avatar_centroid = None if frame_state == "transition" else (_avatar_centroid(normalized_frame) or _focus_centroid(normalized_frame))
        goal_centroid = _foreground_centroid(normalized_goal) if normalized_goal != [[]] else None
        derived_target_centroid: tuple[float, float] | None = None
        derived_target_label = ""
        if goal_centroid is None and avatar_centroid is not None and frame_state != "transition":
            derived_target_centroid, derived_target_label = _select_mechanic_target(
                normalized_frame,
                avatar_centroid,
                budget_snapshot=budget_snapshot,
            )
        target_centroid = goal_centroid or derived_target_centroid
        target_label = "goal" if goal_centroid is not None else derived_target_label
        valid_action_indices = _available_action_indices(available_actions)
        state_key = _navigation_state_key(normalized_frame, levels_completed=levels_completed)
        blocked_actions = set(self._blocked_actions_by_state.get(state_key, set()))
        previous_action_index = None
        previous_frame_blocked = False
        if self.action_history and self._last_frame is not None:
            last_action_index = self.action_history[-1].get("action_index")
            if isinstance(last_action_index, int) and 0 <= int(last_action_index) <= 3:
                previous_action_index = int(last_action_index)
                if _same_gameplay_state(normalized_frame, self._last_frame):
                    previous_frame_blocked = True
                    blocked_actions.add(previous_action_index)
                    self._blocked_actions_by_state[state_key] = set(blocked_actions)
                    if self._last_blocked_action == previous_action_index:
                        self._blocked_repeat_count += 1
                    else:
                        self._last_blocked_action = previous_action_index
                        self._blocked_repeat_count = 1
                else:
                    self._last_blocked_action = None
                    self._blocked_repeat_count = 0
            else:
                self._last_blocked_action = None
                self._blocked_repeat_count = 0
        force_reset = bool(valid_action_indices) and _should_force_reset(
            budget_snapshot=budget_snapshot,
            avatar_centroid=avatar_centroid,
            target_centroid=target_centroid,
            target_label=target_label,
        )
        if int(levels_completed) >= 1:
            force_reset = False
        fresh_context = False
        if self._needs_reperceive:
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
            self._blocked_actions_by_state.clear()
            self._last_blocked_action = None
            self._blocked_repeat_count = 0
            if frame_state != "transition":
                fresh_context = True
        task_context = dict(task_data or {}) if isinstance(task_data, dict) else {}
        if self._needs_reperceive and frame_state == "transition":
            action_index = int(valid_action_indices[0]) if valid_action_indices else 0
            record = {
                "action": ACTION_NAMES[action_index],
                "action_index": action_index,
                "label": ACTION_LABELS[action_index],
                "confidence": 0.0,
                "converged": 0,
                "iterations_used": 0,
                "frame_number": len(self.action_history) + 1,
                "gpu_execution": False,
                "solver": "arc3_transition_reperceive_bridge",
                "task_result": {"program_type": "transition_anim_bridge"},
                "available_actions": list(available_actions or []),
                "click_reason": "transition_anim_neutral",
                "frame_state": frame_state,
                "fresh_context": False,
                "game_id": str(game_id or ""),
                "levels_completed": int(levels_completed),
                "movement_budget": dict(budget_snapshot or {}),
                "lives_remaining": lives_remaining,
                "target_label": target_label,
                "attempt_actions": int(self._attempt_actions),
            }
            self.action_history.append(record)
            self._last_frame = _clone_grid(normalized_frame)
            return record
        spatial_plan = None
        if frame_state != "transition" and not force_reset and _movement_action_indices(valid_action_indices):
            spatial_plan = self._spatial_path_plan(
                normalized_frame,
                avatar_centroid=avatar_centroid,
                budget_snapshot=budget_snapshot,
                valid_action_indices=valid_action_indices,
            )
        gpu_task = {
            "type": "ARC_TASK",
            "task_id": str(
                task_context.get("task_id")
                or task_context.get("id")
                or f"arc3_live_{len(self.action_history) + 1:04d}"
            ),
            "query": _frame_to_query_text(
                normalized_frame,
                normalized_goal,
                available_actions=available_actions,
                frame_state=frame_state,
                fresh_context=fresh_context,
                budget_snapshot=budget_snapshot,
                lives_remaining=lives_remaining,
                reference_box_visible=reference_box_visible,
                flash_semantics=flash_semantics,
                force_reset=force_reset,
            ),
            "input_grid": normalized_frame,
            "expected_output": normalized_goal if normalized_goal != [[]] else [],
            "training_examples": list(task_context.get("train") or []),
            "available_actions": list(available_actions or []),
            "action_options": list(ACTION_NAMES),
            "options": list(ACTION_NAMES),
        }
        if spatial_plan is not None:
            result = {
                "answer_index": int(spatial_plan["action_index"]),
                "confidence": float(spatial_plan["confidence"]),
                "convergence_signal": 1,
                "iterations_used": int(spatial_plan["path_length"]),
                "gpu_execution": True,
                "solver": str(spatial_plan["solver"]),
                "program_type": str(spatial_plan["program_type"]),
                "target_label": str(spatial_plan["target_label"]),
            }
        else:
            result = self.kv.execute_task(
                task=gpu_task,
                route={
                    "specialist": "visual",
                    "domain_hint": "arc3_interactive",
                    "galaxy_names": list(ARC3_ROUTE_GALAXIES),
                },
                specialist="visual",
                domain_hint="arc3_interactive",
            )
        click_reason = ""
        action_choice, payload = _derive_action_from_result(
            normalized_frame,
            dict(result or {}),
            goal_frame=normalized_goal,
        )
        click_only_state = bool(valid_action_indices) and set(valid_action_indices) == {5}
        exploration_reason = ""
        if action_choice == RESET_ACTION_NAME and int(levels_completed) >= 1:
            action_choice = self._exploration_fallback(
                valid_action_indices=valid_action_indices,
                blocked_actions=blocked_actions,
                repeated_action=previous_action_index,
            )
            payload = {}
            exploration_reason = "preserve_progress_no_reset"
        elif isinstance(action_choice, int):
            if previous_frame_blocked and previous_action_index is not None and int(action_choice) == previous_action_index:
                action_choice = self._exploration_fallback(
                    valid_action_indices=valid_action_indices,
                    blocked_actions=blocked_actions,
                    repeated_action=previous_action_index,
                )
                payload = {}
                exploration_reason = "blocked_direction_explore"
            elif previous_action_index is not None and self._blocked_repeat_count >= 2 and int(action_choice) == previous_action_index:
                action_choice = self._exploration_fallback(
                    valid_action_indices=valid_action_indices,
                    blocked_actions=blocked_actions,
                    repeated_action=previous_action_index,
                )
                payload = {}
                exploration_reason = "repeat_loop_explore"
            elif int(action_choice) in blocked_actions:
                action_choice = self._exploration_fallback(
                    valid_action_indices=valid_action_indices,
                    blocked_actions=blocked_actions,
                    repeated_action=previous_action_index,
                )
                payload = {}
                exploration_reason = "blocked_direction_avoid"
        if action_choice == RESET_ACTION_NAME:
            payload = {}
            click_reason = click_reason or exploration_reason or "strategic_reset"
            self._needs_reperceive = True
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
            self._blocked_actions_by_state.clear()
            self._last_blocked_action = None
            self._blocked_repeat_count = 0
            action_name = RESET_ACTION_NAME
            action_index = -1
            action_label = RESET_ACTION_LABEL
        else:
            action_index = int(action_choice)
            if valid_action_indices and action_index not in set(valid_action_indices):
                action_index = int(valid_action_indices[0])
            action_name = ACTION_NAMES[action_index]
            action_label = ACTION_LABELS[action_index]
            if exploration_reason:
                click_reason = exploration_reason
        if action_name == "ACTION6" and not {"x", "y"} <= set(payload):
            payload, click_reason = self._next_click_payload(normalized_frame)
        elif action_name != "ACTION6":
            payload = {}
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
        elif not click_only_state:
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
        if not click_reason and spatial_plan is not None:
            click_reason = f"spatial_path:{spatial_plan['target_label']}"
        record = {
            "action": action_name,
            "action_index": action_index,
            "label": action_label,
            "confidence": float((result or {}).get("confidence", (result or {}).get("similarity", 0.0))),
            "converged": int((result or {}).get("convergence_signal", (result or {}).get("converged", 0))),
            "iterations_used": int((result or {}).get("iterations_used", 0)),
            "frame_number": len(self.action_history) + 1,
            "gpu_execution": bool((result or {}).get("gpu_execution", False)),
            "solver": str((result or {}).get("solver", "knowledgeverse_gpu_query")),
            "task_result": dict(result or {}),
            "available_actions": list(available_actions or []),
            "click_reason": click_reason,
            "frame_state": frame_state,
            "fresh_context": fresh_context,
            "game_id": str(game_id or ""),
            "levels_completed": int(levels_completed),
            "movement_budget": dict(budget_snapshot or {}),
            "lives_remaining": lives_remaining,
            "reference_box_visible": bool(reference_box_visible),
            "flash_semantics": flash_semantics,
            "target_label": target_label,
            "attempt_actions": int(self._attempt_actions),
            "frame_unchanged": bool(previous_frame_blocked),
            "blocked_actions": sorted(int(index) for index in blocked_actions),
            **payload,
        }
        self.action_history.append(record)
        self._last_frame = _clone_grid(normalized_frame)
        if fresh_context:
            self._needs_reperceive = False
        if action_name == RESET_ACTION_NAME:
            self._attempt_actions = 0
        else:
            self._attempt_actions += 1
        return record

    def learn_from_outcome(
        self,
        *,
        levels_completed: int = 0,
        frame: list[list[int]] | None = None,
    ) -> int:
        """Record lightweight outcome metadata; Knowledgeverse owns consolidation."""
        current = max(0, int(levels_completed))
        normalized_frame = _normalize_grid(frame) if frame is not None else None
        if current > self._last_levels_completed:
            outcome = 1
            self._needs_reperceive = True
            self._attempt_actions = 0
            self._blocked_actions_by_state.clear()
            self._last_blocked_action = None
            self._blocked_repeat_count = 0
        elif normalized_frame is not None and self._last_frame is not None and not _same_gameplay_state(normalized_frame, self._last_frame):
            outcome = 0
        else:
            outcome = -1
        if hasattr(self.kv, "record_outcome"):
            self.kv.record_outcome(outcome)
        self._last_levels_completed = current
        if self.action_history:
            self.action_history[-1]["outcome_signal"] = int(outcome)
            self.action_history[-1]["levels_completed"] = current
        return int(outcome)

    def close(self) -> None:
        if self.log_path and self.action_history:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in self.action_history))
                handle.write("\n")

__all__ = ["ACTION_LABELS", "ACTION_NAMES", "ARC3_ROUTE_GALAXIES", "K3DARC3Agent"]
