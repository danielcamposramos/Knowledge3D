from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.bridge.headless_tablet import ActionType, HeadlessTabletMPC, TabletIngest
from knowledge3d.cranium.bridges.trm_step_fused_bridge import TRMStepFusedBridge
from knowledge3d.cranium.sovereign import loader


class _FailingDaemon:
    def handle_command(self, payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("daemon fallback should not run on the bridge query path")


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


@pytest.mark.gpu
def test_tablet_submit_uses_real_bridge_ring_query(tmp_path: Path) -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        tablet = HeadlessTabletMPC(
            command_handler=_FailingDaemon(),
            bridge=bridge,
            storage_root=tmp_path,
        )
        envelope = TabletIngest.math_problem(
            task_id="tablet_ring_cuda_demo",
            question="What is 2 + 2?",
            expected_answer="4",
        )

        result = tablet.submit(envelope)
    finally:
        bridge.cleanup()

    task_result = result["response"]["task_result"]
    assert result["tablet_contract"]["sovereign_path"] == "tablet_bridge_ring"
    assert result["tablet_contract"]["output_action_type"] == ActionType.NO_ACTION.name
    assert task_result["runtime"] == "tablet_bridge_ring_query"
    assert task_result["gpu_execution"] is True
    assert task_result["answer_materialized"] is False
    assert len(task_result["query_embedding_512"]) == 512
    assert len(task_result["y_new_vector_512"]) == 512
    assert len(task_result["action_buffer_words"]) == 72
