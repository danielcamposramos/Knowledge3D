from __future__ import annotations

import pytest

from knowledge3d.knowledgeverse.galaxy_vram_table import (
    GalaxyVRAMTable,
    compose_star_embedding,
)
from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch, cpu_reference_dispatch
from knowledge3d.knowledgeverse.persistent_brain import PersistentBrainState
from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer

from tests.foundational_test_utils import build_resolved_foundational_stars


def _dispatch_tasks(count: int = 10):
    tasks = []
    for index in range(count):
        answer_index = index % 4
        query = [0.0] * 32
        query[answer_index] = 1.0
        option_embeddings = []
        for option_index in range(4):
            option = [0.0] * 32
            option[option_index] = 1.0
            option_embeddings.append(option)
        tasks.append(
            {
                "type": "MMLU_TASK",
                "query_embedding": query,
                "option_embeddings": option_embeddings,
                "subject": "synthetic_subject",
                "domain_hint": "synthetic_domain",
            }
        )
    return tasks


def _dispatch_tasks_arc3(count: int = 4):
    tasks = []
    for index in range(count):
        answer_index = index % 7
        query = [0.0] * 32
        query[answer_index] = 1.0
        option_embeddings = []
        for option_index in range(7):
            option = [0.0] * 32
            option[option_index] = 1.0
            option_embeddings.append(option)
        tasks.append(
            {
                "type": "ARC3_TASK",
                "query_embedding": query,
                "option_embeddings": option_embeddings,
                "subject": "arc3_subject",
                "domain_hint": "arc3_domain",
            }
        )
    return tasks


def test_gpu_task_dispatch_matches_cpu_reference():
    tasks = _dispatch_tasks(10)
    reference = cpu_reference_dispatch(tasks)
    task_buffer = VRAMTaskBuffer(max_tasks=16)
    try:
        task_buffer.bulk_load(tasks)
        dispatcher = GPUTaskDispatch()
        dispatcher.launch(task_buffer, len(tasks))
        results = task_buffer.read_results(len(tasks))
    finally:
        task_buffer.close()

    assert [row["answer_index"] for row in results] == [row["answer_index"] for row in reference]
    assert [row["convergence_signal"] for row in results] == [row["convergence_signal"] for row in reference]
    assert [row["iterations_used"] for row in results] == [row["iterations_used"] for row in reference]


@pytest.mark.parametrize(
    "task_type",
    [
        "ARC_TASK",
        "MATH_TASK",
        "GSM8K_TASK",
        "LHE_TASK",
        "MMLU_TASK",
        "CHAT_TASK",
        "GENERAL_TASK",
        "GRAMMAR_TASK",
    ],
)
def test_gpu_task_dispatch_specialist_switch_keeps_one_hot_winner(task_type: str):
    tasks = _dispatch_tasks(4)
    for task in tasks:
        task["type"] = task_type
    reference = cpu_reference_dispatch(tasks)
    task_buffer = VRAMTaskBuffer(max_tasks=8)
    try:
        task_buffer.bulk_load(tasks)
        dispatcher = GPUTaskDispatch()
        dispatcher.launch(task_buffer, len(tasks))
        results = task_buffer.read_results(len(tasks))
    finally:
        task_buffer.close()

    assert [row["answer_index"] for row in results] == [row["answer_index"] for row in reference]


def test_gpu_task_dispatch_arc3_seven_option_support():
    tasks = _dispatch_tasks_arc3(7)
    reference = cpu_reference_dispatch(tasks)
    task_buffer = VRAMTaskBuffer(max_tasks=8)
    try:
        task_buffer.bulk_load(tasks)
        dispatcher = GPUTaskDispatch()
        dispatcher.launch(task_buffer, len(tasks))
        results = task_buffer.read_results(len(tasks))
    finally:
        task_buffer.close()

    assert [row["answer_index"] for row in results] == [row["answer_index"] for row in reference]


def test_gpu_task_dispatch_thinking_budget_respected():
    tasks = _dispatch_tasks(1)
    tasks[0]["thinking_budget"] = 10
    reference = cpu_reference_dispatch(tasks)
    task_buffer = VRAMTaskBuffer(max_tasks=2)
    try:
        task_buffer.bulk_load(tasks)
        dispatcher = GPUTaskDispatch()
        dispatcher.launch(task_buffer, len(tasks))
        results = task_buffer.read_results(len(tasks))
    finally:
        task_buffer.close()

    assert 5 <= results[0]["iterations_used"] <= 10
    assert results[0]["iterations_used"] == reference[0]["iterations_used"]


def test_gpu_task_dispatch_arc3_persistent_brain_tracks_state():
    first = _dispatch_tasks_arc3(1)[0]
    second = _dispatch_tasks_arc3(1)[0]
    first["query_embedding"][10] = 0.2
    second["query_embedding"][10] = 0.8
    reference_state: dict[str, object] = {}
    reference_first = cpu_reference_dispatch([first], brain_state=reference_state)[0]
    reference_second = cpu_reference_dispatch([second], brain_state=reference_state)[0]

    task_buffer = VRAMTaskBuffer(max_tasks=1)
    brain = PersistentBrainState()
    try:
        dispatcher = GPUTaskDispatch()
        task_buffer.bulk_load([first])
        dispatcher.launch(task_buffer, 1, brain_ptr=brain.gpu_ptr)
        result_first = task_buffer.read_results(1)[0]

        task_buffer.bulk_load([second])
        dispatcher.launch(task_buffer, 1, brain_ptr=brain.gpu_ptr)
        result_second = task_buffer.read_results(1)[0]
        brain_state = brain.read_state()
    finally:
        brain.close()
        task_buffer.close()

    assert result_first["answer_index"] == reference_first["answer_index"]
    assert result_second["answer_index"] == reference_second["answer_index"]
    assert brain_state["frame_count"] == 2
    assert len(brain_state["action_ring"]) == 2
    assert brain_state["reasoning_norm"] > 0.0


def test_gpu_task_dispatch_arc3_galaxy_table_matches_composed_reference():
    galaxy_stars = build_resolved_foundational_stars()
    composed_query = compose_star_embedding(galaxy_stars, 4)
    task = {
        "type": "GAME_2D",
        "query_embedding": composed_query,
        "option_embeddings": [[1.0 if i == j else 0.0 for i in range(32)] for j in range(7)],
        "subject": "arc3_subject",
        "domain_hint": "arc3_domain",
    }
    reference_galaxy = cpu_reference_dispatch([task], galaxy_stars=galaxy_stars)[0]
    reference_slot = cpu_reference_dispatch([task])[0]

    task_buffer = VRAMTaskBuffer(max_tasks=1)
    galaxy_table = GalaxyVRAMTable(max_stars=32)
    try:
        galaxy_table.load_stars(galaxy_stars)
        dispatcher = GPUTaskDispatch()

        task_buffer.bulk_load([task])
        dispatcher.launch(
            task_buffer,
            1,
            star_table=galaxy_table,
        )
        result_galaxy = task_buffer.read_results(1)[0]

        task_buffer.bulk_load([task])
        dispatcher.launch(task_buffer, 1)
        result_slot = task_buffer.read_results(1)[0]
    finally:
        galaxy_table.close()
        task_buffer.close()

    assert result_galaxy["answer_index"] == reference_galaxy["answer_index"]
    assert result_slot["answer_index"] == reference_slot["answer_index"]
    assert (
        result_galaxy["answer_index"] != result_slot["answer_index"]
        or abs(result_galaxy["confidence"] - result_slot["confidence"]) > 1.0e-4
    )


def test_gpu_task_dispatch_non_arc3_galaxy_navigation_changes_result_shape():
    galaxy_stars = build_resolved_foundational_stars()
    composed_query = compose_star_embedding(galaxy_stars, 12)
    task = {
        "type": "QUESTION",
        "query_embedding": composed_query,
        "option_embeddings": [[1.0 if i == j else 0.0 for i in range(32)] for j in range(4)],
        "subject": "mmlu_subject",
        "domain_hint": "broad_knowledge",
    }
    reference_galaxy = cpu_reference_dispatch([task], galaxy_stars=galaxy_stars)[0]
    reference_slot = cpu_reference_dispatch([task])[0]

    task_buffer = VRAMTaskBuffer(max_tasks=1)
    galaxy_table = GalaxyVRAMTable(max_stars=128)
    try:
        galaxy_table.load_stars(galaxy_stars)
        dispatcher = GPUTaskDispatch()

        task_buffer.bulk_load([task])
        dispatcher.launch(
            task_buffer,
            1,
            star_table=galaxy_table,
        )
        result_galaxy = task_buffer.read_results(1)[0]
    finally:
        galaxy_table.close()
        task_buffer.close()

    assert result_galaxy["answer_index"] == reference_galaxy["answer_index"]
    assert (
        reference_galaxy["answer_index"] != reference_slot["answer_index"]
        or abs(reference_galaxy["confidence"] - reference_slot["confidence"]) > 1.0e-4
    )


def test_gpu_task_dispatch_arc3_outputs_goal_progress():
    task = _dispatch_tasks_arc3(1)[0]
    task["goal_embedding"] = [0.0] * 32
    task["goal_embedding"][5] = 1.0
    reference = cpu_reference_dispatch([task])[0]
    task_buffer = VRAMTaskBuffer(max_tasks=1)
    try:
        task_buffer.bulk_load([task])
        dispatcher = GPUTaskDispatch()
        dispatcher.launch(task_buffer, 1)
        result = task_buffer.read_results(1)[0]
    finally:
        task_buffer.close()

    assert result["goal_progress"] == pytest.approx(reference["goal_progress"], abs=1.0e-6)
