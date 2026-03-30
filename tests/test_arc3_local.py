from __future__ import annotations

import json

from benchmarks.arc3_local import ACTION_NAMES, apply_action, make_task, run_local_arc3


class _GoalRelativeFakeKnowledgeverse:
    def execute_task(self, *, task, route=None, specialist=None, domain_hint=None, **kwargs):
        del route, specialist, domain_hint, kwargs
        query = str(task.get("query", "")).lower()
        mapping = {
            "primary action move up": 0,
            "primary action move down": 1,
            "primary action move left": 2,
            "primary action move right": 3,
            "primary action perform": 4,
        }
        for phrase, answer_index in mapping.items():
            if phrase in query:
                return {"answer_index": answer_index, "gpu_execution": True}
        return {"answer_index": 0, "gpu_execution": True}


def test_make_task_matches_expected_known_answer():
    task = make_task(0)

    assert task["start_pos"] == (0, 4)
    assert task["goal_pos"] == (7, 4)
    assert task["optimal_steps"] == 7
    assert task["valid_first_actions"] == [1]


def test_apply_action_moves_foreground_cell_in_correct_direction():
    frame = [[0] * 8 for _ in range(8)]
    frame[1][4] = 2

    moved, changed = apply_action(frame, 1, frame_stack=[frame])

    assert changed is True
    assert moved[2][4] == 2
    assert moved[1][4] == 0


def test_run_local_arc3_solves_single_axis_tasks():
    result = run_local_arc3(
        count=4,
        max_actions=24,
        knowledgeverse=_GoalRelativeFakeKnowledgeverse(),
    )

    assert result["suite"] == "arc3_local"
    assert result["total"] == 4
    assert result["solved"] == 4
    assert result["correct_first_moves"] == 4
    assert result["accuracy"] == 1.0
    assert result["first_move_accuracy"] == 1.0
    assert all(row["actions_taken"][0] in ACTION_NAMES for row in result["results"])


def test_run_local_arc3_streams_rows_to_log(tmp_path):
    log_path = tmp_path / "arc3_local.jsonl"

    result = run_local_arc3(
        count=3,
        max_actions=24,
        knowledgeverse=_GoalRelativeFakeKnowledgeverse(),
        log_path=log_path,
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert result["total"] == 3
    assert len(rows) == 3
    assert [row["id"] for row in rows] == [f"arc3_local_{index:03d}" for index in range(3)]
