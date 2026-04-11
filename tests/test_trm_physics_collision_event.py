from __future__ import annotations

import ctypes

import numpy as np
import pytest

from knowledge3d.cranium.bridges.trm_step_fused_bridge import (
    TRMStepFusedBridge,
    TRM_EVENT_COLLISION,
    TRM_STATE_IDLE,
)
from knowledge3d.cranium.sovereign import loader


class _Float4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float), ("w", ctypes.c_float)]


class _Float2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class _UInt2(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32)]


class _PhysicsBodySOA(ctypes.Structure):
    _fields_ = [
        ("pos_inv", ctypes.c_uint64),
        ("vel_sleep", ctypes.c_uint64),
        ("orientation", ctypes.c_uint64),
        ("ang_vel_damp", ctypes.c_uint64),
        ("inv_inertia_rest", ctypes.c_uint64),
        ("galaxy_handles", ctypes.c_uint64),
        ("island_flags", ctypes.c_uint64),
        ("bound_friction", ctypes.c_uint64),
        ("body_count", ctypes.c_uint32),
        ("capacity", ctypes.c_uint32),
    ]


class _ContactManifoldSOA(ctypes.Structure):
    _fields_ = [
        ("body_a_id", ctypes.c_uint64),
        ("body_b_id", ctypes.c_uint64),
        ("contact_x", ctypes.c_uint64),
        ("contact_y", ctypes.c_uint64),
        ("contact_z", ctypes.c_uint64),
        ("normal_x", ctypes.c_uint64),
        ("normal_y", ctypes.c_uint64),
        ("normal_z", ctypes.c_uint64),
        ("penetration_depth", ctypes.c_uint64),
        ("lambda_normal", ctypes.c_uint64),
        ("lambda_tangent0", ctypes.c_uint64),
        ("lambda_tangent1", ctypes.c_uint64),
        ("compliance_normal", ctypes.c_uint64),
        ("persistent_id", ctypes.c_uint64),
        ("frame_stamp", ctypes.c_uint64),
        ("color_id", ctypes.c_uint64),
        ("capacity", ctypes.c_uint32),
        ("write_head", ctypes.c_uint32),
        ("persistent_counter", ctypes.c_uint32),
    ]


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


def _alloc_host_array(host_obj, allocations: list) -> ctypes.c_uint64:
    ptr = loader.gpu_malloc(ctypes.sizeof(host_obj))
    host_ptr = ctypes.cast(host_obj, ctypes.c_void_p) if isinstance(host_obj, ctypes.Array) else ctypes.cast(ctypes.byref(host_obj), ctypes.c_void_p)
    loader.memcpy_htod(ptr, host_ptr, ctypes.sizeof(host_obj))
    allocations.append(ptr)
    return ptr


@pytest.mark.gpu
def test_trm_physics_phase_integrates_motor_output_and_emits_collision_event() -> None:
    _ensure_cuda()
    bridge = TRMStepFusedBridge()
    rng = np.random.default_rng(23)
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
                    "physics_body_id": 0,
                    "sleep_state": TRM_STATE_IDLE,
                    "motor_output": [1.0, 0.0, 0.0],
                }
            ]
        )
        bridge.bind_state_machines([{"current_state": TRM_STATE_IDLE, "owner_entity_id": 0}])

        pos_arr = (_Float4 * 1)(_Float4(0.0, 1.75, 0.0, 1.0))
        vel_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 0.0))
        ori_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 1.0))
        ang_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 0.01))
        inertia_arr = (_Float4 * 1)(_Float4(1.0, 1.0, 1.0, 0.25))
        handles_arr = (_UInt2 * 1)(_UInt2(2, 0))
        flags_arr = (ctypes.c_uint32 * 1)(0)
        bounds_arr = (_Float2 * 1)(_Float2(0.5, 0.30))

        body_soa = _PhysicsBodySOA(
            pos_inv=int(_alloc_host_array(pos_arr, allocations).value),
            vel_sleep=int(_alloc_host_array(vel_arr, allocations).value),
            orientation=int(_alloc_host_array(ori_arr, allocations).value),
            ang_vel_damp=int(_alloc_host_array(ang_arr, allocations).value),
            inv_inertia_rest=int(_alloc_host_array(inertia_arr, allocations).value),
            galaxy_handles=int(_alloc_host_array(handles_arr, allocations).value),
            island_flags=int(_alloc_host_array(flags_arr, allocations).value),
            bound_friction=int(_alloc_host_array(bounds_arr, allocations).value),
            body_count=1,
            capacity=1,
        )
        d_body_soa = _alloc_host_array(body_soa, allocations)

        body_a = (ctypes.c_uint32 * 1)(0)
        body_b = (ctypes.c_uint32 * 1)(1)
        contact = (ctypes.c_float * 1)(0.0)
        normal = (ctypes.c_float * 1)(1.0)
        penetration = (ctypes.c_float * 1)(0.2)
        lambda_normal = (ctypes.c_float * 1)(0.7)
        lambda_tangent = (ctypes.c_float * 1)(0.0)
        compliance = (ctypes.c_float * 1)(0.0)
        persistent = (ctypes.c_uint32 * 1)(0)
        frame_stamp = (ctypes.c_uint8 * 1)(0)
        color_id = (ctypes.c_uint8 * 1)(0)

        manifold = _ContactManifoldSOA(
            body_a_id=int(_alloc_host_array(body_a, allocations).value),
            body_b_id=int(_alloc_host_array(body_b, allocations).value),
            contact_x=int(_alloc_host_array(contact, allocations).value),
            contact_y=int(_alloc_host_array(contact, allocations).value),
            contact_z=int(_alloc_host_array(contact, allocations).value),
            normal_x=int(_alloc_host_array(normal, allocations).value),
            normal_y=int(_alloc_host_array(contact, allocations).value),
            normal_z=int(_alloc_host_array(contact, allocations).value),
            penetration_depth=int(_alloc_host_array(penetration, allocations).value),
            lambda_normal=int(_alloc_host_array(lambda_normal, allocations).value),
            lambda_tangent0=int(_alloc_host_array(lambda_tangent, allocations).value),
            lambda_tangent1=int(_alloc_host_array(lambda_tangent, allocations).value),
            compliance_normal=int(_alloc_host_array(compliance, allocations).value),
            persistent_id=int(_alloc_host_array(persistent, allocations).value),
            frame_stamp=int(_alloc_host_array(frame_stamp, allocations).value),
            color_id=int(_alloc_host_array(color_id, allocations).value),
            capacity=1,
            write_head=1,
            persistent_counter=0,
        )
        d_manifold = _alloc_host_array(manifold, allocations)

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
            body_soa_ptr=d_body_soa,
            contact_soa_ptr=d_manifold,
            body_count=1,
            tick=1,
        )
        entity = bridge.read_entity_hot_paths()[0]
        events = bridge.drain_events(max_events=4)
    finally:
        bridge.cleanup()
        for ptr in allocations:
            loader.gpu_free(ptr)

    assert entity["house_x"] > 0.0
    assert any(int(row["event_type"]) == TRM_EVENT_COLLISION for row in events)
