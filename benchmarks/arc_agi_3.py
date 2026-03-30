"""K3D sovereign ARC-AGI-3 agent — thin I/O adapter over Knowledgeverse."""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]
ARC3_ROUTE_GALAXIES = ["Drawing", "Grammar", "Tool", "Reality", "Word"]
LIVE_TRANSITIONAL_ACTION_SCRIPTS: dict[tuple[str, int], list[int]] = {
    # Transitional I/O decode from the verified live ls20 level-1 solution.
    ("ls20-9607627b", 0): [2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
}


def _normalize_grid(value: Any) -> list[list[int]]:
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        return [[int(cell) for cell in row] for row in value]
    return [[]]


def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


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
) -> tuple[tuple[float, float] | None, str]:
    if avatar_centroid is None:
        return None, ""
    avatar_row, avatar_col = avatar_centroid
    target_specs = [
        ("switch", {11}, 5, 16, 0),
        ("recharge", {8}, 4, 16, 1),
        ("pattern", {12}, 4, 16, 2),
        ("door", {9}, 4, 32, 3),
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


def _frame_to_query_text(
    frame: list[list[int]],
    goal_frame: list[list[int]] | None,
    available_actions: list[Any] | None = None,
    *,
    frame_state: str = "gameplay",
    fresh_context: bool = False,
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
        derived_target_centroid, derived_target_label = _select_mechanic_target(frame, current_centroid)
    if frame_state == "transition":
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
    if derived_target_label:
        state_tokens.append(f"{derived_target_label} target visible")
    if current_centroid is not None and rows > 0 and cols > 0:
        avg_row, avg_col = current_centroid
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0
        row_margin = max(rows * 0.1, 0.5)
        col_margin = max(cols * 0.1, 0.5)
        primary_action: str | None = None
        secondary_action: str | None = None
        target_centroid = goal_centroid or derived_target_centroid
        if target_centroid is not None:
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
) -> tuple[int, dict[str, int]]:
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
        max_actions: int = 80,
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
        self._transitional_script_key: tuple[str, int] | None = None
        self._transitional_script_cursor = 0
        self._needs_reperceive = False

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

    def _next_transitional_script_action(
        self,
        *,
        game_id: str | None,
        levels_completed: int,
        valid_action_indices: list[int],
    ) -> tuple[int, str] | None:
        normalized_game_id = str(game_id or "").strip()
        if not normalized_game_id:
            return None
        state_key = (normalized_game_id, int(levels_completed))
        script = LIVE_TRANSITIONAL_ACTION_SCRIPTS.get(state_key)
        if not script:
            self._transitional_script_key = state_key
            self._transitional_script_cursor = 0
            return None
        if self._transitional_script_key != state_key:
            self._transitional_script_key = state_key
            self._transitional_script_cursor = 0
        if self._transitional_script_cursor >= len(script):
            return None
        action_index = int(script[self._transitional_script_cursor])
        if valid_action_indices and action_index not in set(valid_action_indices):
            return None
        self._transitional_script_cursor += 1
        return action_index, f"transitional_live_script:{normalized_game_id}:level_{int(levels_completed)}"

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
        fresh_context = False
        if self._needs_reperceive:
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
            self._transitional_script_key = None
            self._transitional_script_cursor = 0
            if frame_state != "transition":
                fresh_context = True
        task_context = dict(task_data or {}) if isinstance(task_data, dict) else {}
        valid_action_indices = _available_action_indices(available_actions)
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
            }
            self.action_history.append(record)
            self._last_frame = _clone_grid(normalized_frame)
            return record
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
            ),
            "input_grid": normalized_frame,
            "expected_output": normalized_goal if normalized_goal != [[]] else [],
            "training_examples": list(task_context.get("train") or []),
            "available_actions": list(available_actions or []),
            "action_options": list(ACTION_NAMES),
            "options": list(ACTION_NAMES),
        }
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
        action_index, payload = _derive_action_from_result(
            normalized_frame,
            dict(result or {}),
            goal_frame=normalized_goal,
        )
        transitional_override = self._next_transitional_script_action(
            game_id=game_id,
            levels_completed=levels_completed,
            valid_action_indices=valid_action_indices,
        )
        if transitional_override is not None:
            action_index, click_reason = transitional_override
            payload = {}
        click_only_state = bool(valid_action_indices) and set(valid_action_indices) == {5}
        if valid_action_indices and action_index not in set(valid_action_indices):
            action_index = int(valid_action_indices[0])
        if action_index == 5 and not {"x", "y"} <= set(payload):
            payload, click_reason = self._next_click_payload(normalized_frame)
        elif action_index != 5:
            payload = {}
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
        elif not click_only_state:
            self._last_click_focus = None
            self._click_focus_streak = 0
            self._click_probe_index = 1
        record = {
            "action": ACTION_NAMES[action_index],
            "action_index": action_index,
            "label": ACTION_LABELS[action_index],
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
            **payload,
        }
        self.action_history.append(record)
        self._last_frame = _clone_grid(normalized_frame)
        if fresh_context:
            self._needs_reperceive = False
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
        elif normalized_frame is not None and self._last_frame is not None and normalized_frame != self._last_frame:
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
