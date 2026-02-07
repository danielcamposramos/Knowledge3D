from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_trm_weights_persist_across_knowledgeverse_restarts(tmp_path):
    storage_root = tmp_path / "kv_weights"
    kv1 = Knowledgeverse(storage_root=storage_root)

    for _ in range(3):
        kv1.log_event(
            "math_problem_success",
            {
                "specialist": "math",
                "query": "Find derivative of x^2 at x=3",
                "confidence": 0.9,
            },
        )

    state_path = storage_root / "checkpoints" / "trm_routing_state.json"
    assert state_path.exists()

    kv2 = Knowledgeverse(storage_root=storage_root)
    bias = kv2.trm_navigator.specialist_router.get_specialist_bias()
    assert bias["math"] > 0.0
    assert kv2.navigator_specialist.routing_topology


def test_sleeptime_stage_b_consolidates_trm_weights(tmp_path):
    storage_root = tmp_path / "kv_sleeptime_weights"
    kv = Knowledgeverse(storage_root=storage_root)

    kv.log_event(
        "arc_task_success",
        {
            "specialist": "visual",
            "query": "Rotate this ARC grid and reflect color mapping",
            "confidence": 0.88,
        },
    )
    kv.log_event(
        "math_problem_failure",
        {
            "specialist": "math",
            "query": "Compute hard olympiad expression",
            "confidence": 0.2,
        },
    )

    result = kv.sleeptime.execute()
    stage_b = result["stage_b"]
    assert stage_b["success"] is True
    assert stage_b["updated_count"] >= 1
    weights_path = Path(stage_b["weights_path"])
    assert weights_path.exists()

    kv_reloaded = Knowledgeverse(storage_root=storage_root)
    bias = kv_reloaded.trm_navigator.specialist_router.get_specialist_bias()
    assert bias["visual"] > 0.0
