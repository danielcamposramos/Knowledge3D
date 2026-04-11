from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from knowledge3d.cranium.sovereign.loader import (
    gpu_free,
    gpu_malloc,
    memcpy_dtoh,
    memcpy_htod,
)
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_PH_BODY_DESPAWN,
    OP_PH_BODY_SPAWN,
    OP_PH_BROAD_PHASE,
    OP_PH_COLLISION_QUERY,
    OP_PH_CONSTRAINT_COLOR,
    OP_PH_CONSTRAINT_GENERATE,
    OP_PH_FRICTION_APPLY,
    OP_PH_GALAXY_WRITE,
    OP_PH_GRAVITY_APPLY,
    OP_PH_IMPULSE_PROPAGATE,
    OP_PH_INTEGRATE,
    OP_PH_ISLAND_WAKE,
    OP_PH_MATERIAL_FETCH,
    OP_PH_NARROW_PHASE,
    OP_PH_PREDICT_POS,
    OP_PH_RESTITUTION_APPLY,
    OP_PH_SLEEP_CHECK,
    OP_PH_TERNARY_CLASSIFY,
    OP_PH_XPBD_SOLVE,
)
from knowledge3d.cranium.ptx_runtime.sovereign_physics import (
    PHYSICS_OPCODE_CONTRACTS,
    PHYSICS_PHASE_SEQUENCE,
    RESERVED_PHYSICS_OPCODE_RANGES,
)
from knowledge3d.cranium.sovereign_physics_bootstrap import (
    build_default_gravity_force_law,
    build_default_sleep_policy,
    build_physical_constant_stars,
    build_physics_material_stars,
    serialize_material_table,
)
from knowledge3d.ingestion import ingest_physics_bootstrap
from knowledge3d.ingestion.warp_importer import import_warp_model


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


class _PhysicsPredictedSOA(ctypes.Structure):
    _fields_ = [
        ("predicted_pos_inv", ctypes.c_uint64),
        ("predicted_orientation", ctypes.c_uint64),
        ("capacity", ctypes.c_uint32),
    ]


def _device_copy(host_obj) -> tuple[ctypes.c_uint64, object]:
    ptr = gpu_malloc(ctypes.sizeof(host_obj))
    memcpy_htod(ptr, ctypes.cast(ctypes.byref(host_obj), ctypes.c_void_p), ctypes.sizeof(host_obj))
    return ptr, host_obj


def _read_float4(ptr: ctypes.c_uint64) -> _Float4:
    value = _Float4()
    memcpy_dtoh(ctypes.cast(ctypes.byref(value), ctypes.c_void_p), ptr, ctypes.sizeof(value))
    return value


def test_physics_opcode_contracts_cover_live_surface() -> None:
    live_surface = {
        OP_PH_BROAD_PHASE,
        OP_PH_NARROW_PHASE,
        OP_PH_CONSTRAINT_GENERATE,
        OP_PH_XPBD_SOLVE,
        OP_PH_INTEGRATE,
        OP_PH_SLEEP_CHECK,
        OP_PH_GALAXY_WRITE,
        OP_PH_MATERIAL_FETCH,
        OP_PH_PREDICT_POS,
        OP_PH_CONSTRAINT_COLOR,
        OP_PH_IMPULSE_PROPAGATE,
        OP_PH_RESTITUTION_APPLY,
        OP_PH_FRICTION_APPLY,
        OP_PH_ISLAND_WAKE,
        OP_PH_BODY_SPAWN,
        OP_PH_BODY_DESPAWN,
        OP_PH_GRAVITY_APPLY,
        OP_PH_COLLISION_QUERY,
        OP_PH_TERNARY_CLASSIFY,
    }
    assert live_surface.issubset(PHYSICS_OPCODE_CONTRACTS.keys())
    assert PHYSICS_PHASE_SEQUENCE[0] == OP_PH_BROAD_PHASE
    assert PHYSICS_PHASE_SEQUENCE[-1] == OP_PH_GALAXY_WRITE
    assert RESERVED_PHYSICS_OPCODE_RANGES[-1] == (0x178, 0x17F, "soft_body_xpbd")


def test_modular_rpn_engine_registers_physics_tokens() -> None:
    assert ModularRPNEngine.OPCODES["ph_broad_phase"] == OP_PH_BROAD_PHASE
    assert ModularRPNEngine.OPCODES["ph_constraint_color"] == OP_PH_CONSTRAINT_COLOR
    assert ModularRPNEngine.OPCODES["ph_collision_query"] == OP_PH_COLLISION_QUERY


def test_sovereign_physics_bootstrap_contains_required_constants() -> None:
    constant_ids = {entry["star_id"] for entry in build_physical_constant_stars()}
    assert "physics_constant_gravitational" in constant_ids
    assert "physics_constant_speed_of_light" in constant_ids
    assert len(constant_ids) == 11

    material_ids = {entry["star_id"] for entry in build_physics_material_stars()}
    assert {"physics_material_steel", "physics_material_wood", "physics_material_rubber", "physics_material_ice"} <= material_ids

    gravity_law = build_default_gravity_force_law()
    assert gravity_law["facet"] == "force_law"
    assert "PH_GRAVITY_APPLY" in gravity_law["physics_rpn_addr"]

    sleep_policy = build_default_sleep_policy()
    assert sleep_policy["facet"] == "meta_rule"
    assert sleep_policy["layer"] == 4
    serialized = serialize_material_table()
    assert serialized == [
        {"star_id": 1, "friction": 0.57, "restitution": 0.25, "density": 7850.0, "texture_id": 0xFFFFFFFF},
        {"star_id": 2, "friction": 0.3, "restitution": 0.35, "density": 700.0, "texture_id": 0xFFFFFFFF},
        {"star_id": 3, "friction": 0.9, "restitution": 0.8, "density": 1100.0, "texture_id": 0xFFFFFFFF},
        {"star_id": 4, "friction": 0.03, "restitution": 0.05, "density": 917.0, "texture_id": 0xFFFFFFFF},
    ]


def test_ingestion_bootstrap_writes_expected_physics_rows() -> None:
    class _FakeManager:
        def __init__(self) -> None:
            self.rows: list[tuple[str, dict]] = []

        def upsert_entry(self, galaxy_name: str, entry: dict) -> str:
            self.rows.append((galaxy_name, entry))
            return "inserted"

    manager = _FakeManager()
    count = ingest_physics_bootstrap(manager)
    assert count == 17
    reality_ids = {entry["id"] for galaxy, entry in manager.rows if galaxy == "Reality"}
    grammar_ids = {entry["id"] for galaxy, entry in manager.rows if galaxy == "Grammar"}
    assert "physics_constant_gravitational" in reality_ids
    assert "physics_material_wood" in reality_ids
    assert "physics_law_default_gravity" in grammar_ids
    assert "physics_meta_sleep_policy_default" in grammar_ids


def test_warp_importer_maps_bodies_to_rigid_body_stars() -> None:
    class _Shape:
        geo_type = "box"
        half_extents = (0.5, 1.0, 1.5)

    class _Material:
        density = 700.0
        dynamic_friction = 0.3
        restitution = 0.2

    class _Body:
        position = (1.0, 2.0, 3.0)
        is_sleeping = False

    class _Model:
        bodies = [_Body()]
        shapes = [_Shape()]
        shape_materials = [_Material()]
        body_mass = [2.5]
        body_qd = [(0.1, 0.2, 0.3)]

    imported = import_warp_model(_Model(), source_path="warp_scene.py")
    assert len(imported.stars) == 1
    star = imported.stars[0]
    assert star["facet"] == "rigid_body"
    assert star["physics_rpn_addr"] == "physics_law_default_gravity"
    assert star["material_star_id"] == "physics_material_wood"
    assert star["shape_star_id"].startswith("drawing_box_")
    assert star["source_path"] == "warp_scene.py"


def test_trm_step_fused_source_has_explicit_physics_phase_slot() -> None:
    path = Path("knowledge3d/cranium/ptx/trm_step_fused.cu")
    source = path.read_text(encoding="utf-8")
    assert "PHYSICS_PHASE" in source
    assert "trm_phase2_physics_step" in source
    assert "physics_soa_ptr" in source
    assert "solver_iterations" in source


def test_falling_sphere_smoke() -> None:
    try:
        from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine as BridgeModularRPNEngine

        engine = BridgeModularRPNEngine()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not initialise ModularRPNEngine: {exc}")

    pos_arr = (_Float4 * 1)(_Float4(0.0, 2.0, 0.0, 1.0))
    vel_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 0.0))
    ori_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 1.0))
    ang_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 0.01))
    inertia_arr = (_Float4 * 1)(_Float4(1.0, 1.0, 1.0, 0.25))
    handles_arr = (_UInt2 * 1)(_UInt2(2, 0))
    flags_arr = (ctypes.c_uint32 * 1)(0)
    bounds_arr = (_Float2 * 1)(_Float2(0.5, 0.30))
    pred_pos_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 1.0))
    pred_ori_arr = (_Float4 * 1)(_Float4(0.0, 0.0, 0.0, 1.0))

    allocated: list[ctypes.c_uint64] = []

    def _alloc_array(arr) -> ctypes.c_uint64:
        ptr = gpu_malloc(ctypes.sizeof(arr))
        memcpy_htod(ptr, ctypes.cast(arr, ctypes.c_void_p), ctypes.sizeof(arr))
        allocated.append(ptr)
        return ptr

    try:
        d_pos = _alloc_array(pos_arr)
        d_vel = _alloc_array(vel_arr)
        d_ori = _alloc_array(ori_arr)
        d_ang = _alloc_array(ang_arr)
        d_inertia = _alloc_array(inertia_arr)
        d_handles = _alloc_array(handles_arr)
        d_flags = _alloc_array(flags_arr)
        d_bounds = _alloc_array(bounds_arr)
        d_pred_pos = _alloc_array(pred_pos_arr)
        d_pred_ori = _alloc_array(pred_ori_arr)

        body_soa_host = _PhysicsBodySOA(
            pos_inv=int(d_pos.value),
            vel_sleep=int(d_vel.value),
            orientation=int(d_ori.value),
            ang_vel_damp=int(d_ang.value),
            inv_inertia_rest=int(d_inertia.value),
            galaxy_handles=int(d_handles.value),
            island_flags=int(d_flags.value),
            bound_friction=int(d_bounds.value),
            body_count=1,
            capacity=1,
        )
        predicted_soa_host = _PhysicsPredictedSOA(
            predicted_pos_inv=int(d_pred_pos.value),
            predicted_orientation=int(d_pred_ori.value),
            capacity=1,
        )
        d_body_soa, _ = _device_copy(body_soa_host)
        d_predicted_soa, _ = _device_copy(predicted_soa_host)
        allocated.extend([d_body_soa, d_predicted_soa])

        engine.bind_physics_runtime(body_soa=d_body_soa, predicted_soa=d_predicted_soa)
        bound_count = engine.bind_physics_material_table(serialize_material_table())
        assert bound_count == 4

        material_top = engine.execute_single(
            instance_id=0,
            op_codes=[0x00, OP_PH_MATERIAL_FETCH],
            scalars=[2.0],
            vectors=[],
        )
        assert material_top == pytest.approx(700.0, abs=1e-3)
        assert engine.read_instance_stack_scalars(0) == pytest.approx([0.30, 0.35, 700.0], abs=1e-3)

        dt = 1.0 / 60.0
        for _ in range(60):
            top = engine.execute_single(
                instance_id=0,
                op_codes=[0x00, OP_PH_GRAVITY_APPLY, 0x00, OP_PH_PREDICT_POS, 0x00, OP_PH_INTEGRATE, 0x00],
                scalars=[-9.81, dt, dt, 1.0],
                vectors=[],
            )
            assert top == pytest.approx(1.0, abs=1e-6)

        result = _read_float4(d_pos)
        assert result.y == pytest.approx(-2.905, abs=0.2)
    finally:
        for ptr in reversed(allocated):
            try:
                gpu_free(ptr)
            except Exception:
                pass
