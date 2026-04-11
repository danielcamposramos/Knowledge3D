from __future__ import annotations

import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_EVENT_INTERNAL,
    TRM_STATE_IDLE,
    TRM_STATE_REASONING,
)
from knowledge3d.cranium.sovereign import loader


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


@pytest.mark.gpu
def test_trm_state_machine_drains_at_most_8_events_per_tick() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        bridge.reset_runtime(current_state=TRM_STATE_IDLE)
        bridge.enqueue_events(
            [
                {"entity_id": 0, "event_type": TRM_EVENT_INTERNAL, "priority": idx, "payload": idx}
                for idx in range(16)
            ]
        )
        states = bridge.step_state_machine(tick=1)
        remaining = bridge.drain_events(max_events=16)
    finally:
        bridge.cleanup()

    assert states[0]["current_state"] == TRM_STATE_REASONING
    assert len(remaining) == 8
    assert all(int(row["event_type"]) == TRM_EVENT_INTERNAL for row in remaining)
