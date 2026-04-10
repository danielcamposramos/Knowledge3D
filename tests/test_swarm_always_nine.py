from __future__ import annotations

import pytest

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


@pytest.mark.parametrize(
    ("task_type", "query_text", "options", "task"),
    [
        ("ARC_TASK", "solve arc transformation task", None, None),
        ("MATH_TASK", "Mary has 5 apples and eats 2. How many are left?", None, {"type": "MATH_TASK"}),
        ("MMLU_TASK", "Which option best matches the concept?", ["A", "B", "C", "D"], None),
    ],
)
def test_swarm_paths_always_assign_all_nine_workers(tmp_path, task_type, query_text, options, task):
    kv = Knowledgeverse(storage_root=tmp_path / f"kv_swarm_{task_type.lower()}")

    paths = kv._build_gpu_reasoning_paths(
        task=task,
        task_type=task_type,
        primary_program_id=Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID
        if task_type == "ARC_TASK"
        else Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID
        if task_type == "MATH_TASK"
        else "reasoning_elimination_top1",
        query_text=query_text,
        options=options,
    )

    assert len(paths) >= 9
    assert [int(path.get("worker_slot", -1)) for path in paths[:9]] == list(range(9))
    assert [str(path.get("worker_name", "")) for path in paths[:9]] == list(kv.FIXED_GRE_WORKERS)
    assert {str(path.get("worker_name", "")) for path in paths[:9]} == set(kv.FIXED_GRE_WORKERS)


def test_arc_surface_keeps_arc_reasoner_in_fixed_slot_three(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_swarm_arc_slot")
    paths = kv._build_gpu_reasoning_paths(
        task_type="ARC_TASK",
        primary_program_id=Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID,
        query_text="solve arc transformation task",
    )
    assert str(paths[3].get("worker_name", "")) == "gre_arc_reasoner"
