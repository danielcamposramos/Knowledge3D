from __future__ import annotations

import ctypes

import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    _GPUEventStruct,
)


@pytest.mark.gpu
def test_gpu_event_struct_abi_matches_phase11_contract() -> None:
    assert ctypes.sizeof(_GPUEventStruct) == 16

    bridge = TRMStepFusedBridge()
    try:
        assert int(bridge._d_event_ring.value) % 16 == 0
    finally:
        bridge.cleanup()


@pytest.mark.gpu
def test_gpu_event_queue_enqueue_host_batch_lands_256_events_exactly_once() -> None:
    bridge = TRMStepFusedBridge()
    try:
        bridge.reset_runtime()
        pushed = bridge.enqueue_events(
            [
                {
                    "entity_id": 0,
                    "event_type": 5,
                    "priority": index & 0xFF,
                    "payload": 50_000 + index,
                }
                for index in range(256)
            ]
        )
        drained = bridge.drain_events(max_events=256)
    finally:
        bridge.cleanup()

    assert pushed == [1] * 256
    assert len(drained) == 256
    payloads = sorted(int(row["payload"]) for row in drained)
    assert payloads == list(range(50_000, 50_256))


@pytest.mark.gpu
def test_bridge_blocks_host_injection_when_gpu_producers_flag_is_set() -> None:
    bridge = TRMStepFusedBridge()
    try:
        bridge.set_gpu_producers_active(True)
        with pytest.raises(AssertionError, match="GPU producers are active"):
            bridge.enqueue_event(entity_id=0, event_type=5, payload=1)
    finally:
        bridge.cleanup()
