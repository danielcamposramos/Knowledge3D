from __future__ import annotations

import ctypes

import numpy as np
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


def _alloc_vector(values: np.ndarray):
    ptr = loader.gpu_malloc(values.nbytes)
    loader.memcpy_htod(ptr, values.ctypes.data_as(ctypes.c_void_p), values.nbytes)
    return ptr


@pytest.mark.gpu
def test_trm_step_fused_dispatches_distinct_entities_per_block() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    rng = np.random.default_rng(11)
    q = rng.standard_normal(512, dtype=np.float32)
    y = np.ones(512, dtype=np.float32)
    z = np.ones(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    allocations = []
    try:
        bridge.bind_entity_hot_paths(
            [
                {"star_table_idx": 0, "sleep_state": TRM_STATE_IDLE, "awareness": 1.0, "house_z": 0.0},
                {"star_table_idx": 1, "sleep_state": TRM_STATE_SLEEP, "awareness": 1.0, "house_z": 2.0},
                {"star_table_idx": 2, "sleep_state": TRM_STATE_IDLE, "awareness": 1.0, "house_z": 4.0},
                {"star_table_idx": 3, "sleep_state": TRM_STATE_SLEEP, "awareness": 1.0, "house_z": 6.0},
            ]
        )
        bridge.bind_state_machines(
            [
                {"current_state": TRM_STATE_IDLE, "owner_entity_id": 0},
                {"current_state": TRM_STATE_SLEEP, "owner_entity_id": 1},
                {"current_state": TRM_STATE_IDLE, "owner_entity_id": 2},
                {"current_state": TRM_STATE_SLEEP, "owner_entity_id": 3},
            ]
        )

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

        tick_meta = bridge.launch_tick(
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
        entities = bridge.read_entity_hot_paths()
        states = bridge.read_state_machines()
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)

    assert len(tick_meta["entity_results"]) == 4
    assert [row["owner_entity_id"] for row in states] == [0, 1, 2, 3]
    assert entities[0]["awareness"] > entities[1]["awareness"]
    assert entities[2]["awareness"] > entities[3]["awareness"]
    assert states[0]["current_state"] == TRM_STATE_IDLE
    assert states[1]["current_state"] == TRM_STATE_SLEEP
