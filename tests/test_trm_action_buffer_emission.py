from __future__ import annotations

import ctypes
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    ACTION_BUFFER_BYTES,
    ACTION_BUFFER_WORDS,
    ACTION_DIALOGUE,
    ACTION_NAV_LOOK,
    ACTION_NAV_MOVE,
    ACTION_NO_ACTION,
    ACTION_UPDATE_TABLET,
    ACTION_WRITE_MEM,
    TRMStepFusedBridge,
    TRM_STATE_ACTING,
    TRM_STATE_HANDLING_QUERY,
    TRM_STATE_IDLE,
    TRM_STATE_NAVIGATING,
    TRM_STATE_PERCEIVING,
    TRM_STATE_SLEEP,
)
from knowledge3d.cranium.sovereign import loader


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _float_from_word(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word) & 0xFFFFFFFF))[0]


def _alloc_vector(values: np.ndarray):
    ptr = loader.gpu_malloc(values.nbytes)
    loader.memcpy_htod(ptr, values.ctypes.data_as(ctypes.c_void_p), values.nbytes)
    return ptr


def _launch_and_read_actions(
    entities: list[dict],
    states: list[dict],
    *,
    tick: int = 1,
    grid_x_override: int | None = None,
    entity_count_override: int | None = None,
) -> tuple[list[list[int]], list[dict]]:
    bridge = TRMStepFusedBridge()
    q = np.zeros(512, dtype=np.float32)
    y = np.ones(512, dtype=np.float32)
    z = np.ones(512, dtype=np.float32)
    w = np.zeros(1, dtype=np.float32)

    allocations = []
    try:
        bridge.bind_entity_hot_paths(entities)
        bridge.bind_state_machines(states)
        bridge.reset_event_ring()

        d_q = _alloc_vector(q)
        d_y = _alloc_vector(y)
        d_z = _alloc_vector(z)
        d_w1 = _alloc_vector(w)
        d_w2 = _alloc_vector(w)
        d_w3 = _alloc_vector(w)
        d_w4 = _alloc_vector(w)
        d_z_new = loader.gpu_malloc(z.nbytes)
        d_y_new = loader.gpu_malloc(y.nbytes)
        d_workspace = loader.gpu_malloc(max(1, len(entities)) * 4096 * 4)
        allocations.extend([d_q, d_y, d_z, d_w1, d_w2, d_w3, d_w4, d_z_new, d_y_new, d_workspace])

        bridge.launch_tick(
            q_ptr=d_q,
            y_ptr=d_y,
            z_ptr=d_z,
            W1_ptr=d_w1,
            W2_ptr=d_w2,
            W3_ptr=d_w3,
            W4_ptr=d_w4,
            z_new_ptr=d_z_new,
            y_new_ptr=d_y_new,
            workspace_ptr=d_workspace,
            tick=tick,
            grid_x_override=grid_x_override,
            entity_count_override=entity_count_override,
        )
        count = int(entity_count_override if entity_count_override is not None else len(entities))
        actions = bridge.read_action_buffers_words(count)
        entity_rows = bridge.read_entity_hot_paths()
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)
    return actions, entity_rows


def test_action_buffer_dtype_layout_direct_import() -> None:
    path = Path("knowledge3d/cranium/actions/action_types.py")
    spec = importlib.util.spec_from_file_location("k3d_action_types_direct", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dtype = module.ACTION_BUFFER_DTYPE
    assert dtype.itemsize == ACTION_BUFFER_BYTES == 288
    assert ACTION_BUFFER_WORDS == 72
    assert dtype.fields["action_type"][1] == 0
    assert dtype.fields["confidence"][1] == 4
    assert dtype.fields["nav_position"][1] == 16
    assert dtype.fields["nav_direction"][1] == 28
    assert dtype.fields["nav_velocity"][1] == 40
    assert dtype.fields["dialogue_token_ids"][1] == 76
    assert dtype.fields["dialogue_length"][1] == 140
    assert dtype.fields["dialogue_thinking_score"][1] == 148
    assert dtype.fields["mem_summary_hash"][1] == 176
    assert dtype.fields["mem_embedding"][1] == 192
    assert dtype.fields["tablet_mutation_type"][1] == 240
    assert dtype.fields["tablet_data"][1] == 244


@pytest.mark.gpu
def test_trm_action_buffer_perceiving_emits_nav_look() -> None:
    _ensure_cuda()
    actions, _entities = _launch_and_read_actions(
        [
            {
                "star_table_idx": 0,
                "sleep_state": TRM_STATE_PERCEIVING,
                "awareness": 0.4,
                "perception_radius": 20.0,
                "gaze_yaw": 0.0,
                "gaze_pitch": 0.0,
                "gaze_fov": 0.9,
                "house_x": 1.0,
                "house_y": 2.0,
                "house_z": 3.0,
            },
            {
                "star_table_idx": 77,
                "sleep_state": TRM_STATE_IDLE,
                "house_x": 1.0,
                "house_y": 2.0,
                "house_z": 8.0,
            },
        ],
        [
            {"current_state": TRM_STATE_PERCEIVING, "owner_entity_id": 0},
            {"current_state": TRM_STATE_IDLE, "owner_entity_id": 1},
        ],
        grid_x_override=1,
        entity_count_override=2,
    )

    slot = actions[0]
    assert slot[0] == ACTION_NAV_LOOK
    assert _float_from_word(slot[4]) == pytest.approx(1.0)
    assert _float_from_word(slot[5]) == pytest.approx(2.0)
    assert _float_from_word(slot[6]) == pytest.approx(3.0)
    assert _float_from_word(slot[7]) == pytest.approx(0.0, abs=1e-6)
    assert _float_from_word(slot[8]) == pytest.approx(0.0, abs=1e-6)
    assert _float_from_word(slot[9]) == pytest.approx(1.0, abs=1e-6)
    assert slot[11] == 77
    assert _float_from_word(slot[12]) == pytest.approx(0.125, abs=1e-6)


@pytest.mark.gpu
def test_trm_action_buffer_navigating_emits_nav_move() -> None:
    _ensure_cuda()
    actions, entities = _launch_and_read_actions(
        [
            {
                "star_table_idx": 0,
                "sleep_state": TRM_STATE_NAVIGATING,
                "attention_entity_id": 1,
                "current_goal_star": 1,
                "awareness": 0.6,
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
        ],
        [
            {"current_state": TRM_STATE_NAVIGATING, "owner_entity_id": 0},
            {"current_state": TRM_STATE_IDLE, "owner_entity_id": 1},
        ],
        grid_x_override=1,
        entity_count_override=2,
    )

    slot = actions[0]
    entity = entities[0]
    speed = _float_from_word(slot[10])
    assert slot[0] == ACTION_NAV_MOVE
    assert speed > 0.0
    assert _float_from_word(slot[4]) == pytest.approx(entity["house_x"], abs=1e-6)
    assert _float_from_word(slot[7]) > 0.0
    assert _float_from_word(slot[8]) == pytest.approx(0.0, abs=1e-6)
    assert slot[11] == 1
    assert _float_from_word(slot[12]) == pytest.approx(0.6, abs=1e-6)


@pytest.mark.gpu
def test_trm_action_buffer_sleep_and_idle_emit_no_action() -> None:
    _ensure_cuda()
    actions, _entities = _launch_and_read_actions(
        [
            {"star_table_idx": 0, "sleep_state": TRM_STATE_SLEEP, "awareness": 0.5},
            {"star_table_idx": 1, "sleep_state": TRM_STATE_IDLE, "awareness": 0.5},
        ],
        [
            {"current_state": TRM_STATE_SLEEP, "owner_entity_id": 0},
            {"current_state": TRM_STATE_IDLE, "owner_entity_id": 1},
        ],
    )

    assert actions[0][0] == ACTION_NO_ACTION
    assert actions[1][0] == ACTION_NO_ACTION


@pytest.mark.gpu
def test_trm_action_buffer_handling_query_emits_no_action_before_pop_state() -> None:
    _ensure_cuda()
    actions, _entities = _launch_and_read_actions(
        [
            {"star_table_idx": 0, "sleep_state": TRM_STATE_IDLE, "awareness": 0.2},
            {
                "star_table_idx": 1,
                "sleep_state": TRM_STATE_HANDLING_QUERY,
                "awareness": 1.0,
                "current_goal_star": 123,
                "motor_output": [1.0, 0.0, 0.0],
            },
        ],
        [
            {"current_state": TRM_STATE_IDLE, "owner_entity_id": 0},
            {
                "current_state": TRM_STATE_HANDLING_QUERY,
                "owner_entity_id": 1,
                "stack_depth": 1,
                "state_stack": [TRM_STATE_NAVIGATING],
            },
        ],
    )

    assert actions[1][0] == ACTION_NO_ACTION


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("meta_rule_addr", "expected_action"),
    [
        (4, ACTION_DIALOGUE),
        (5, ACTION_WRITE_MEM),
        (6, ACTION_UPDATE_TABLET),
        (7, ACTION_NAV_MOVE),
    ],
)
def test_trm_action_buffer_acting_dispatches_payloads(meta_rule_addr: int, expected_action: int) -> None:
    _ensure_cuda()
    actions, _entities = _launch_and_read_actions(
        [
            {
                "star_table_idx": 0,
                "sleep_state": TRM_STATE_ACTING,
                "awareness": 0.2,
                "meta_rule_addr": meta_rule_addr,
                "current_goal_star": 77,
                "house_x": 1.0,
                "house_y": 2.0,
                "house_z": 3.0,
                "motor_output": [1.0, 0.0, 0.0],
            }
        ],
        [{"current_state": TRM_STATE_ACTING, "owner_entity_id": 0}],
    )

    slot = actions[0]
    assert slot[0] == expected_action
    assert _float_from_word(slot[1]) > 0.2
    if expected_action == ACTION_DIALOGUE:
        assert (slot[19] & 0xFFFF) == 77
        assert slot[35] == 1
        assert _float_from_word(slot[37]) > 0.2
    elif expected_action == ACTION_WRITE_MEM:
        assert slot[44] != 0 or slot[45] != 0
        assert slot[46] == 77
        assert _float_from_word(slot[47]) > 0.2
        assert _float_from_word(slot[48]) > 1.0
        assert _float_from_word(slot[49]) == pytest.approx(2.0, abs=1e-6)
        assert _float_from_word(slot[51]) > 0.2
    elif expected_action == ACTION_UPDATE_TABLET:
        assert slot[60] == meta_rule_addr >> 2
        assert slot[61] == 77
        assert slot[62] == TRM_STATE_ACTING
        assert _float_from_word(slot[63]) > 0.2
    else:
        assert _float_from_word(slot[10]) > 0.0
        assert slot[11] == 77


@pytest.mark.gpu
def test_trm_action_buffer_multi_entity_slots_are_independent() -> None:
    _ensure_cuda()
    actions, _entities = _launch_and_read_actions(
        [
            {
                "star_table_idx": 0,
                "sleep_state": TRM_STATE_NAVIGATING,
                "attention_entity_id": 1,
                "current_goal_star": 1,
                "awareness": 0.7,
                "house_x": 0.0,
                "house_y": 1.75,
                "house_z": 0.0,
            },
            {
                "star_table_idx": 1,
                "sleep_state": TRM_STATE_PERCEIVING,
                "awareness": 0.4,
                "house_x": 5.0,
                "house_y": 1.75,
                "house_z": 0.0,
            },
            {"star_table_idx": 2, "sleep_state": TRM_STATE_IDLE, "awareness": 0.3},
        ],
        [
            {"current_state": TRM_STATE_NAVIGATING, "owner_entity_id": 0},
            {"current_state": TRM_STATE_PERCEIVING, "owner_entity_id": 1},
            {"current_state": TRM_STATE_IDLE, "owner_entity_id": 2},
        ],
    )

    assert len(actions) == 3
    assert actions[0][0] == ACTION_NAV_MOVE
    assert actions[1][0] == ACTION_NAV_LOOK
    assert actions[2][0] == ACTION_NO_ACTION
    assert _float_from_word(actions[0][12]) == pytest.approx(0.7, abs=1e-6)
    assert _float_from_word(actions[1][4]) == pytest.approx(5.0, abs=1e-6)
    assert _float_from_word(actions[2][1]) < _float_from_word(actions[0][1])
