from __future__ import annotations

import time

import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_STATE_IDLE,
    TRM_STATE_SLEEP,
)
from knowledge3d.cranium.sovereign import loader


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _wait_until(predicate, *, timeout_s: float = 1.0) -> bool:
    deadline = time.perf_counter() + float(timeout_s)
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@pytest.mark.gpu
def test_trm_tick_loop_runs_background_ticks() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        start_count = bridge.tick_count
        bridge.start_tick_loop()
        assert _wait_until(lambda: bridge.tick_count - start_count >= 8, timeout_s=0.5)
        status = bridge.stop_tick_loop()
    finally:
        bridge.cleanup()

    assert status["tick_count"] >= start_count + 8
    assert status["last_error"] == ""


@pytest.mark.gpu
def test_trm_tick_loop_query_preemption_resumes_background() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        bridge.start_tick_loop()
        assert _wait_until(lambda: bridge.tick_count >= 2, timeout_s=0.5)
        before_query_count = bridge.tick_count
        query_result = bridge.run_query_tick(max_steps=1, epsilon=0.0)
        after_query_count = bridge.tick_count
        assert query_result["steps"] == 1
        assert _wait_until(lambda: bridge.tick_count > after_query_count, timeout_s=0.5)
        status = bridge.stop_tick_loop()
    finally:
        bridge.cleanup()

    assert after_query_count >= before_query_count
    assert status["tick_count"] > after_query_count
    assert status["last_error"] == ""


@pytest.mark.gpu
def test_trm_tick_loop_idle_to_sleep_without_waiting_35_seconds() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        bridge.reset_runtime(current_state=TRM_STATE_IDLE)
        bridge.bind_state_machines(
            [
                {
                    "current_state": TRM_STATE_IDLE,
                    "owner_entity_id": 0,
                    "idle_accumulator": 29.99,
                    "stack_depth": 0,
                    "state_entry_tick": 0,
                    "deferred_event_mask": 0,
                    "interrupt_priority_level": 0,
                    "last_tick": 0,
                }
            ]
        )
        bridge.start_tick_loop()
        assert _wait_until(
            lambda: bridge.read_state_machines()[0]["current_state"] == TRM_STATE_SLEEP,
            timeout_s=0.5,
        )
        status = bridge.stop_tick_loop()
        state = bridge.read_state_machines()[0]
    finally:
        bridge.cleanup()

    assert state["current_state"] == TRM_STATE_SLEEP
    assert status["last_error"] == ""
