from __future__ import annotations

import ctypes

import numpy as np
import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_STATE_IDLE,
    TRM_STATE_NAVIGATING,
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
def test_trm_navigating_phase_updates_motor_output_toward_goal() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    rng = np.random.default_rng(17)
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
                {
                    "star_table_idx": 0,
                    "sleep_state": TRM_STATE_NAVIGATING,
                    "attention_entity_id": 1,
                    "current_goal_star": 1,
                    "house_x": 0.0,
                    "house_y": 1.75,
                    "house_z": 0.0,
                },
                {
                    "star_table_idx": 1,
                    "sleep_state": TRM_STATE_IDLE,
                    "house_x": 5.0,
                    "house_y": 1.75,
                    "house_z": 0.0,
                },
            ]
        )
        bridge.bind_state_machines(
            [
                {"current_state": TRM_STATE_NAVIGATING, "owner_entity_id": 0},
                {"current_state": TRM_STATE_IDLE, "owner_entity_id": 1},
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
            tick=1,
            grid_x_override=1,
            entity_count_override=2,
        )
        entity = bridge.read_entity_hot_paths()[0]
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)

    assert entity["motor_output"][0] > 0.0
    assert abs(entity["gaze_yaw"]) > 0.01
