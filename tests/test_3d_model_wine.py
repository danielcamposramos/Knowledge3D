from __future__ import annotations

from knowledge3d.tablet.wine import (
    MODEL_TYPE_PROCEDURAL,
    External3DWineBridge,
    build_3d_task,
    create_multimodal_3d_task,
)


def test_procedural_bridge_preserves_task_id_and_route() -> None:
    bridge = External3DWineBridge()
    result = bridge.bridge_external_3d(
        MODEL_TYPE_PROCEDURAL,
        {
            "params": {
                "primitive_type": "cube",
                "dimensions": [2.0, 3.0, 4.0],
                "position": [1.0, 2.0, 3.0],
            }
        },
        task_id="proc_task_001",
    )
    envelope = result["envelope"]
    assert envelope.task_id == "proc_task_001"
    assert envelope.task["task_id"] == "proc_task_001"
    assert envelope.specialist == "visual_3d"
    assert "PROCEDURAL_3D_BEGIN" in result["rpn_program"]


def test_build_3d_task_merges_metadata() -> None:
    task, route = build_3d_task(
        task_id="proc_task_002",
        model_type=MODEL_TYPE_PROCEDURAL,
        external_data={"params": {"primitive_type": "sphere", "dimensions": [1.5, 1.5, 1.5]}},
        metadata={"author": "codex"},
    )
    assert task["task_id"] == "proc_task_002"
    assert task["metadata"]["author"] == "codex"
    assert route["specialist"] == "visual_3d"
    assert "Drawing" in route["galaxy_names"]


def test_multimodal_3d_task_is_deterministic_shape() -> None:
    envelope = create_multimodal_3d_task(
        task_id="multi_001",
        text_prompt="stone tower with spiral stairs",
        generation_params={"temperature": 0.2, "style": "architectural"},
    )
    assert envelope.task_id == "multi_001"
    assert envelope.task["task_id"] == "multi_001"
    assert envelope.metadata["embedding_dims"] == 512
    assert "MULTIMODAL_FUSION" in envelope.task["rpn_program"]
