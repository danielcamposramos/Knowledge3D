from __future__ import annotations

import ctypes

import numpy as np
import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_STATE_IDLE,
    TRM_STATE_NAVIGATING,
    TRM_STATE_PERCEIVING,
)
from knowledge3d.cranium.sovereign import loader
from knowledge3d.knowledgeverse.galaxy_vram_table import GalaxyVRAMTable


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
def test_trm_perceiving_phase_selects_goal_and_advances_to_navigation() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    rng = np.random.default_rng(13)
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
                    "sleep_state": TRM_STATE_PERCEIVING,
                    "perception_radius": 20.0,
                    "gaze_yaw": 0.0,
                    "gaze_pitch": 0.0,
                    "gaze_fov": 0.9,
                    "house_x": 0.0,
                    "house_y": 1.75,
                    "house_z": 0.0,
                },
                {
                    "star_table_idx": 1,
                    "sleep_state": TRM_STATE_IDLE,
                    "house_x": 0.0,
                    "house_y": 1.75,
                    "house_z": 5.0,
                },
            ]
        )
        bridge.bind_state_machines(
            [
                {"current_state": TRM_STATE_PERCEIVING, "owner_entity_id": 0},
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
        entity_after_perception = bridge.read_entity_hot_paths()[0]
        tick_two = bridge.launch_tick(
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
            grid_x_override=1,
            entity_count_override=2,
        )
        entity_after_nav = bridge.read_entity_hot_paths()[0]
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)

    assert entity_after_perception["attention_entity_id"] == 1
    assert entity_after_perception["current_goal_star"] == 1
    assert tick_two["current_state"] == TRM_STATE_NAVIGATING
    assert entity_after_nav["motor_output"][2] > 0.0


@pytest.mark.gpu
def test_trm_perceiving_phase_selects_visible_house_star() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    table = GalaxyVRAMTable(max_stars=4)
    q = np.zeros(512, dtype=np.float32)
    y = np.ones(512, dtype=np.float32)
    z = np.ones(512, dtype=np.float32)
    w = np.zeros(1, dtype=np.float32)

    allocations = []
    try:
        table.load_stars(
            [
                {
                    "id": "behind_avatar",
                    "embedding": [1.0] + [0.0] * 63,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "semantic_position": [0.0, 1.75, -2.0],
                },
                {
                    "id": "visible_house_star",
                    "embedding": [0.0, 1.0] + [0.0] * 62,
                    "selection_role": "answer",
                    "answer_eligible": True,
                    "semantic_position": [0.0, 1.75, 4.0],
                },
            ]
        )
        bridge.bind_galaxy_table(
            table.gpu_ptr,
            table.star_count,
            host_stars=table.read_stars(),
        )
        bridge.bind_entity_hot_paths(
            [
                {
                    "star_table_idx": 0,
                    "sleep_state": TRM_STATE_PERCEIVING,
                    "perception_radius": 20.0,
                    "gaze_yaw": 0.0,
                    "gaze_pitch": 0.0,
                    "gaze_fov": 0.9,
                    "house_x": 0.0,
                    "house_y": 1.75,
                    "house_z": 0.0,
                },
            ]
        )
        bridge.bind_state_machines(
            [
                {"current_state": TRM_STATE_PERCEIVING, "owner_entity_id": 0},
            ]
        )

        d_q = _alloc_vector(q)
        d_y = _alloc_vector(y)
        d_z = _alloc_vector(z)
        d_W1 = _alloc_vector(w)
        d_W2 = _alloc_vector(w)
        d_W3 = _alloc_vector(w)
        d_W4 = _alloc_vector(w)
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
            entity_count_override=1,
        )
        entity_after_perception = bridge.read_entity_hot_paths()[0]
    finally:
        bridge.cleanup()
        table.close()
        for ptr in allocations:
            loader.gpu_free(ptr)

    assert entity_after_perception["current_goal_star"] == 1
    assert entity_after_perception["blackboard_star_id"] == 1
    assert entity_after_perception["attention_entity_id"] == 0xFFFFFFFF
