"""Local ARC3 benchmark with deterministic grid-navigation tasks."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

from benchmarks.arc_agi_3 import ACTION_LABELS, ACTION_NAMES, K3DARC3Agent


TASK_CONFIGS: list[dict[str, Any]] = [
    {"start": (0, 4), "goal": (7, 4), "size": 8},
    {"start": (7, 4), "goal": (0, 4), "size": 8},
    {"start": (3, 4), "goal": (6, 4), "size": 8},
    {"start": (6, 4), "goal": (3, 4), "size": 8},
    {"start": (4, 0), "goal": (4, 7), "size": 8},
    {"start": (4, 7), "goal": (4, 0), "size": 8},
    {"start": (4, 2), "goal": (4, 5), "size": 8},
    {"start": (4, 5), "goal": (4, 2), "size": 8},
    {"start": (1, 1), "goal": (6, 6), "size": 8},
    {"start": (6, 6), "goal": (1, 1), "size": 8},
    {"start": (1, 6), "goal": (6, 1), "size": 8},
    {"start": (6, 1), "goal": (1, 6), "size": 8},
    {"start": (4, 4), "goal": (4, 4), "size": 8},
    {"start": (0, 0), "goal": (0, 0), "size": 8},
    {"start": (3, 4), "goal": (4, 4), "size": 8},
    {"start": (4, 4), "goal": (3, 4), "size": 8},
    {"start": (4, 3), "goal": (4, 4), "size": 8},
    {"start": (4, 4), "goal": (4, 3), "size": 8},
    {"start": (0, 0), "goal": (7, 7), "size": 8},
    {"start": (7, 7), "goal": (0, 0), "size": 8},
]

ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _foreground_cells(grid: list[list[int]]) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            cell = int(value)
            if cell != 0:
                cells.append((row_index, col_index, cell))
    return cells


def _make_grid(size: int, position: tuple[int, int], color: int) -> list[list[int]]:
    grid = [[0] * size for _ in range(size)]
    grid[int(position[0])][int(position[1])] = int(color)
    return grid


def _procedural_task_config(index: int, grid_size: int) -> dict[str, Any]:
    size = max(8, int(grid_size))
    usable = max(2, size - 1)
    start_row = int((index * 3) % usable)
    start_col = int((index * 5) % usable)
    goal_row = int((index * 7 + 2) % usable)
    goal_col = int((index * 11 + 1) % usable)
    return {
        "start": (start_row, start_col),
        "goal": (goal_row, goal_col),
        "size": size,
    }


def make_task(index: int, grid_size: int = 8) -> dict[str, Any]:
    config = dict(TASK_CONFIGS[index]) if 0 <= int(index) < len(TASK_CONFIGS) else _procedural_task_config(index, grid_size)
    start = tuple(int(value) for value in config["start"])
    goal = tuple(int(value) for value in config["goal"])
    size = int(config["size"])
    color = 1 + (int(index) % 9)
    optimal_steps = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    valid_first_actions: list[int] = []
    if goal[0] < start[0]:
        valid_first_actions.append(0)
    elif goal[0] > start[0]:
        valid_first_actions.append(1)
    if goal[1] < start[1]:
        valid_first_actions.append(2)
    elif goal[1] > start[1]:
        valid_first_actions.append(3)
    if not valid_first_actions:
        valid_first_actions.append(4)
    return {
        "task_id": f"arc3_local_{int(index):03d}",
        "start_grid": _make_grid(size, start, color),
        "goal_grid": _make_grid(size, goal, color),
        "start_pos": start,
        "goal_pos": goal,
        "color": color,
        "size": size,
        "optimal_steps": optimal_steps,
        "budget": max(1, optimal_steps * 3),
        "valid_first_actions": valid_first_actions,
    }


def apply_action(
    frame: list[list[int]],
    action_index: int,
    *,
    frame_stack: list[list[list[int]]] | None = None,
) -> tuple[list[list[int]], bool]:
    current = _clone_grid(frame)
    if action_index == 6:
        if frame_stack is not None and len(frame_stack) > 1:
            frame_stack.pop()
            return _clone_grid(frame_stack[-1]), True
        return current, False
    if action_index in {4, 5}:
        return current, False
    if action_index not in ACTION_DELTAS:
        return current, False

    cells = _foreground_cells(current)
    if not cells:
        return current, False
    rows = len(current)
    cols = len(current[0]) if rows else 0
    delta_row, delta_col = ACTION_DELTAS[action_index]
    next_grid = [[0] * cols for _ in range(rows)]
    for row_index, col_index, color in cells:
        next_row = min(rows - 1, max(0, row_index + delta_row))
        next_col = min(cols - 1, max(0, col_index + delta_col))
        next_grid[next_row][next_col] = color
    changed = next_grid != current
    if changed and frame_stack is not None:
        frame_stack.append(_clone_grid(next_grid))
    return next_grid, changed


def grids_equal(left: list[list[int]], right: list[list[int]]) -> bool:
    return list(left) == list(right)


def run_game(
    agent: K3DARC3Agent,
    task: dict[str, Any],
    *,
    max_actions: int | None = None,
) -> dict[str, Any]:
    frame = _clone_grid(task["start_grid"])
    goal = _clone_grid(task["goal_grid"])
    budget = min(int(task["budget"]), int(max_actions)) if max_actions is not None else int(task["budget"])
    budget = max(1, budget)
    frame_stack = [_clone_grid(frame)]
    actions_taken: list[str] = []
    solved = False
    steps_taken = 0
    correct_first_move = False

    for step in range(budget):
        action = agent.choose_action(
            frame,
            goal_frame=goal,
            task_data={
                "task_id": task["task_id"],
                "train": [{"input": _clone_grid(task["start_grid"]), "output": _clone_grid(goal)}],
            },
            available_actions=list(range(len(ACTION_NAMES))),
        )
        action_index = max(0, min(int(action.get("action_index", 0)), len(ACTION_NAMES) - 1))
        if step == 0:
            correct_first_move = action_index in set(int(value) for value in task.get("valid_first_actions") or [])
        frame, _changed = apply_action(frame, action_index, frame_stack=frame_stack)
        steps_taken = step + 1
        solved = grids_equal(frame, goal)
        agent.learn_from_outcome(levels_completed=1 if solved else 0, frame=frame)
        actions_taken.append(ACTION_NAMES[action_index])
        if solved:
            break

    return {
        "suite": "arc3_local",
        "id": str(task["task_id"]),
        "solved": bool(solved),
        "correct": bool(solved),
        "steps_taken": int(steps_taken),
        "optimal_steps": int(task["optimal_steps"]),
        "first_action_correct": bool(correct_first_move),
        "valid_first_actions": [ACTION_NAMES[int(index)] for index in task.get("valid_first_actions") or []],
        "actions_taken": list(actions_taken),
        "start_pos": list(task["start_pos"]),
        "goal_pos": list(task["goal_pos"]),
        "size": int(task["size"]),
    }


def run_local_arc3(
    *,
    count: int = 20,
    grid_size: int = 8,
    max_actions: int = 40,
    knowledgeverse: Any | None = None,
    log_path: str | Path | None = None,
    row_cb: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int | None = None,
) -> dict[str, Any]:
    total = max(1, int(count))
    results: list[dict[str, Any]] = []
    solved = 0
    correct_first_moves = 0
    solved_steps: list[int] = []
    solved_optimality: list[float] = []
    progress_step = max(1, int(progress_every or 1))
    start = time.monotonic()
    agent = K3DARC3Agent(max_actions=max_actions, knowledgeverse=knowledgeverse)
    output_path = Path(log_path) if log_path is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    try:
        for index in range(total):
            task = make_task(index, grid_size=grid_size)
            row = run_game(agent, task, max_actions=max_actions)
            results.append(row)
            solved += int(bool(row["solved"]))
            correct_first_moves += int(bool(row["first_action_correct"]))
            if row["solved"] and int(row["steps_taken"]) > 0:
                solved_steps.append(int(row["steps_taken"]))
                solved_optimality.append(float(row["optimal_steps"]) / float(row["steps_taken"]))
            if output_path is not None:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row_cb is not None:
                row_cb(task, row)
            if progress_cb is not None and ((index + 1) % progress_step == 0 or (index + 1) == total):
                progress_cb(
                    {
                        "completed": index + 1,
                        "total": total,
                        "correct": solved,
                        "correct_first_moves": correct_first_moves,
                        "elapsed_s": time.monotonic() - start,
                        "benchmark": "arc3_local",
                    }
                )
    finally:
        agent.close()

    return {
        "suite": "arc3_local",
        "total": total,
        "solved": solved,
        "correct": solved,
        "accuracy": float(solved / total) if total else 0.0,
        "correct_first_moves": int(correct_first_moves),
        "first_move_accuracy": float(correct_first_moves / total) if total else 0.0,
        "avg_steps": float(sum(solved_steps) / len(solved_steps)) if solved_steps else 0.0,
        "avg_optimality": float(sum(solved_optimality) / len(solved_optimality)) if solved_optimality else 0.0,
        "results": results,
    }


__all__ = [
    "ACTION_LABELS",
    "ACTION_NAMES",
    "TASK_CONFIGS",
    "apply_action",
    "grids_equal",
    "make_task",
    "run_game",
    "run_local_arc3",
]
