from __future__ import annotations

import ctypes

import numpy as np
import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_EVENT_INTERNAL,
    TRM_EVENT_WAKEUP,
    TRM_STATE_HANDLING_QUERY,
    TRM_STATE_IDLE,
    TRM_STATE_REASONING,
    TRM_STATE_SLEEP,
)
from knowledge3d.cranium.sovereign import loader


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _alloc_vector(values: np.ndarray):
    ptr = loader.gpu_malloc(values.nbytes)
    loader.memcpy_htod(ptr, values.ctypes.data_as(ctypes.c_void_p), values.nbytes)
    return ptr


@pytest.mark.gpu
def test_gpu_event_ring_buffer_handles_1000_events_across_32_threads() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        bridge.reset_runtime()
        pushed = bridge.stress_enqueue(thread_count=32, total_events=1000, entity_id=0, payload_base=10_000)
        drained = bridge.drain_events(max_events=1000)
    finally:
        bridge.cleanup()

    assert len(pushed) == 1000
    assert all(value == 1 for value in pushed)
    assert len(drained) == 1000
    payloads = sorted(int(row["payload"]) for row in drained)
    assert payloads == list(range(10_000, 11_000))
    assert all(int(row["event_type"]) == TRM_EVENT_INTERNAL for row in drained)


@pytest.mark.gpu
def test_trm_state_machine_runs_scripted_query_preemption_and_idle_sleep() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    try:
        bridge.reset_runtime(current_state=TRM_STATE_SLEEP)

        bridge.enqueue_event(entity_id=0, event_type=TRM_EVENT_WAKEUP)
        state = bridge.step_state_machine(tick=1)[0]
        assert state["current_state"] == TRM_STATE_IDLE

        bridge.enqueue_event(entity_id=0, event_type=TRM_EVENT_INTERNAL)
        state = bridge.step_state_machine(tick=2)[0]
        assert state["current_state"] == TRM_STATE_REASONING

        bridge.enqueue_query(entity_id=0, payload=77)
        state = bridge.step_state_machine(tick=3)[0]
        assert state["current_state"] == TRM_STATE_HANDLING_QUERY
        assert state["stack_depth"] == 1
        assert state["state_stack"][0] == TRM_STATE_REASONING

        state = bridge.step_state_machine(tick=4)[0]
        assert state["current_state"] == TRM_STATE_REASONING
        assert state["stack_depth"] == 0

        bridge.reset_runtime(current_state=TRM_STATE_IDLE)
        bridge.bind_state_machines(
            [
                {
                    "current_state": TRM_STATE_IDLE,
                    "stack_depth": 0,
                    "idle_accumulator": 29.99,
                    "state_entry_tick": 0,
                    "deferred_event_mask": 0,
                    "interrupt_priority_level": 0,
                    "last_tick": 0,
                }
            ]
        )
        state = bridge.step_state_machine(delta_time=0.02, tick=5)[0]
        assert state["current_state"] == TRM_STATE_SLEEP
    finally:
        bridge.cleanup()


@pytest.mark.gpu
def test_trm_step_fused_routes_live_phase1_states() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    rng = np.random.default_rng(7)
    q = rng.standard_normal(512, dtype=np.float32)
    y = np.ones(512, dtype=np.float32)
    z = np.ones(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    allocations = []
    try:
        d_q = _alloc_vector(q)
        d_y = _alloc_vector(y)
        d_z = _alloc_vector(z)
        d_W1 = _alloc_vector(W1)
        d_W2 = _alloc_vector(W2)
        d_W3 = _alloc_vector(W3)
        d_W4 = _alloc_vector(W4)
        d_z_new = loader.gpu_malloc(z.nbytes)
        d_y_new = loader.gpu_malloc(y.nbytes)
        d_workspace = loader.gpu_malloc(4096 * 4)
        allocations.extend([d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, d_workspace])

        bridge.reset_runtime(current_state=TRM_STATE_SLEEP)
        sleep_meta = bridge.launch_tick(
            q_ptr=d_q,
            y_ptr=d_y,
            z_ptr=d_z,
            W1_ptr=d_W1,
            W2_ptr=d_W2,
            W3_ptr=d_W3,
            W4_ptr=d_W4,
            z_new_ptr=d_z_new,
            y_new_ptr=d_y_new,
            workspace_ptr=d_workspace,
            tick=1,
        )
        y_sleep = np.empty(512, dtype=np.float32)
        loader.memcpy_dtoh(y_sleep.ctypes.data_as(ctypes.c_void_p), d_y_new, y_sleep.nbytes)
        assert sleep_meta["current_state"] == TRM_STATE_SLEEP
        assert sleep_meta["steps"] == 0
        assert np.all(y_sleep < 1.0)

        bridge.reset_runtime(current_state=TRM_STATE_IDLE)
        bridge.launch_tick(
            q_ptr=d_q,
            y_ptr=d_y,
            z_ptr=d_z,
            W1_ptr=d_W1,
            W2_ptr=d_W2,
            W3_ptr=d_W3,
            W4_ptr=d_W4,
            z_new_ptr=d_z_new,
            y_new_ptr=d_y_new,
            workspace_ptr=d_workspace,
            tick=2,
        )
        entity_idle = bridge.read_entity_hot_paths()[0]
        assert entity_idle["sleep_state"] == TRM_STATE_IDLE
        assert entity_idle["awareness"] <= 0.0

        bridge.reset_runtime(current_state=TRM_STATE_REASONING)
        reasoning_meta = bridge.launch_tick(
            q_ptr=d_q,
            y_ptr=d_y,
            z_ptr=d_z,
            W1_ptr=d_W1,
            W2_ptr=d_W2,
            W3_ptr=d_W3,
            W4_ptr=d_W4,
            z_new_ptr=d_z_new,
            y_new_ptr=d_y_new,
            workspace_ptr=d_workspace,
            tick=3,
            max_steps=3,
            epsilon=0.0,
        )
        assert reasoning_meta["current_state"] == TRM_STATE_REASONING
        assert reasoning_meta["steps"] == 3

        bridge.reset_runtime(current_state=TRM_STATE_REASONING)
        query_meta = bridge.run_query_tick(
            q_ptr=d_q,
            y_ptr=d_y,
            z_ptr=d_z,
            W1_ptr=d_W1,
            W2_ptr=d_W2,
            W3_ptr=d_W3,
            W4_ptr=d_W4,
            z_new_ptr=d_z_new,
            y_new_ptr=d_y_new,
            workspace_ptr=d_workspace,
            tick=4,
            max_steps=3,
            epsilon=0.0,
        )
        assert query_meta["current_state"] == TRM_STATE_REASONING
        assert query_meta["steps"] == 3
        state_after_query = bridge.read_state_machines()[0]
        assert state_after_query["current_state"] == TRM_STATE_REASONING
        assert state_after_query["stack_depth"] == 0
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)
