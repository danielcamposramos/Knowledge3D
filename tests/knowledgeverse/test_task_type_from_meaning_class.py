from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_numeric_meaning_maps_to_math_task() -> None:
    task_type = Knowledgeverse._task_type_from_meaning_class(
        meaning_class="NUMERIC_COMPUTE",
        task={"query": "Janet had 16 ducks and bought 2 more. How many ducks now?"},
    )
    assert task_type == "MATH_TASK"


def test_spatial_inputs_map_to_arc_task() -> None:
    task_type = Knowledgeverse._task_type_from_meaning_class(
        meaning_class="FACTUAL_RECALL",
        task={"input_grid": [[1, 0], [0, 1]], "training_examples": [{"input": [[1]], "output": [[1]]}]},
    )
    assert task_type == "ARC_TASK"


def test_choice_meaning_maps_to_question_task() -> None:
    task_type = Knowledgeverse._task_type_from_meaning_class(
        meaning_class="COMPARATIVE_CHOICE",
        task={"query": "Which option is best?", "options": ["A", "B", "C", "D"]},
        options=["A", "B", "C", "D"],
    )
    assert task_type == "QUESTION_TASK"
