"""Sovereign rigid-body physics contracts for the modular RPN/PTX surface.

This module keeps the physics phase grounded in the same vocabulary used by the
rest of K3D:

- Reality Galaxy Layer-2 stores constants/materials/bodies.
- Grammar Galaxy Layer-3 stores `physics_rpn_addr` programs.
- Meta-rules govern sleep/wake and consolidation.

The contracts here are ingestion/runtime metadata only. Hot-path execution still
occurs inside PTX kernels and the fused TRM loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .rpn_opcodes import (
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


@dataclass(frozen=True)
class PhysicsOpcodeContract:
    opcode: int
    name: str
    stack_in: Tuple[str, ...]
    stack_out: Tuple[str, ...]
    kernel: str
    notes: str


PHYSICS_PHASE_SEQUENCE: Tuple[int, ...] = (
    OP_PH_BROAD_PHASE,
    OP_PH_NARROW_PHASE,
    OP_PH_CONSTRAINT_GENERATE,
    OP_PH_CONSTRAINT_COLOR,
    OP_PH_PREDICT_POS,
    OP_PH_XPBD_SOLVE,
    OP_PH_INTEGRATE,
    OP_PH_SLEEP_CHECK,
    OP_PH_GALAXY_WRITE,
)


PHYSICS_OPCODE_CONTRACTS: Dict[int, PhysicsOpcodeContract] = {
    OP_PH_BROAD_PHASE: PhysicsOpcodeContract(
        opcode=OP_PH_BROAD_PHASE,
        name="PH_BROAD_PHASE",
        stack_in=("dt", "body_count"),
        stack_out=("pair_count",),
        kernel="physics_broad_phase_sap.cu",
        notes="Morton update + SAP pair generation; reuses morton_encode_point semantics.",
    ),
    OP_PH_NARROW_PHASE: PhysicsOpcodeContract(
        opcode=OP_PH_NARROW_PHASE,
        name="PH_NARROW_PHASE",
        stack_in=("pair_count",),
        stack_out=("contact_count",),
        kernel="physics_narrow_phase_gjk.cu",
        notes="Warp-cooperative GJK/EPA over broad-phase pairs.",
    ),
    OP_PH_CONSTRAINT_GENERATE: PhysicsOpcodeContract(
        opcode=OP_PH_CONSTRAINT_GENERATE,
        name="PH_CONSTRAINT_GENERATE",
        stack_in=("contact_count",),
        stack_out=("constraint_count",),
        kernel="physics_constraint_generate.cu",
        notes="Contact manifold to XPBD constraints, including warm-start state.",
    ),
    OP_PH_XPBD_SOLVE: PhysicsOpcodeContract(
        opcode=OP_PH_XPBD_SOLVE,
        name="PH_XPBD_SOLVE",
        stack_in=("iter_count",),
        stack_out=("error",),
        kernel="physics_xpbd_solve.cu",
        notes="One XPBD/Jacobi color pass; adapts defeasible-resolver reduction patterns.",
    ),
    OP_PH_INTEGRATE: PhysicsOpcodeContract(
        opcode=OP_PH_INTEGRATE,
        name="PH_INTEGRATE",
        stack_in=("dt",),
        stack_out=(),
        kernel="physics_integrate.cu",
        notes="Symplectic Euler + quaternion exponential-map update.",
    ),
    OP_PH_SLEEP_CHECK: PhysicsOpcodeContract(
        opcode=OP_PH_SLEEP_CHECK,
        name="PH_SLEEP_CHECK",
        stack_in=("energy_threshold",),
        stack_out=("island_count",),
        kernel="physics_sleep_island.cu",
        notes="Warp ballot sleep detection with island refinement.",
    ),
    OP_PH_GALAXY_WRITE: PhysicsOpcodeContract(
        opcode=OP_PH_GALAXY_WRITE,
        name="PH_GALAXY_WRITE",
        stack_in=(),
        stack_out=("edge_count",),
        kernel="physics_collision_event_write.cu",
        notes="Collision event queue to Galaxy edge update staging.",
    ),
    OP_PH_MATERIAL_FETCH: PhysicsOpcodeContract(
        opcode=OP_PH_MATERIAL_FETCH,
        name="PH_MATERIAL_FETCH",
        stack_in=("star_id",),
        stack_out=("friction", "restitution", "density"),
        kernel="inline",
        notes="Layer-2 material/constant fetch through bound Galaxy metadata.",
    ),
    OP_PH_PREDICT_POS: PhysicsOpcodeContract(
        opcode=OP_PH_PREDICT_POS,
        name="PH_PREDICT_POS",
        stack_in=("dt",),
        stack_out=(),
        kernel="physics_xpbd_predict.cu",
        notes="Predict positions/orientations into dedicated SOA buffers.",
    ),
    OP_PH_CONSTRAINT_COLOR: PhysicsOpcodeContract(
        opcode=OP_PH_CONSTRAINT_COLOR,
        name="PH_CONSTRAINT_COLOR",
        stack_in=("constraint_count",),
        stack_out=("color_count",),
        kernel="physics_constraint_color.cu",
        notes="Constraint graph coloring for Gauss-Seidel/XPBD scheduling.",
    ),
    OP_PH_IMPULSE_PROPAGATE: PhysicsOpcodeContract(
        opcode=OP_PH_IMPULSE_PROPAGATE,
        name="PH_IMPULSE_PROPAGATE",
        stack_in=("island_id",),
        stack_out=(),
        kernel="inline",
        notes="Impulse wave through island-local adjacency; reserved for GRE graph crystallizer reuse.",
    ),
    OP_PH_RESTITUTION_APPLY: PhysicsOpcodeContract(
        opcode=OP_PH_RESTITUTION_APPLY,
        name="PH_RESTITUTION_APPLY",
        stack_in=("contact_count",),
        stack_out=(),
        kernel="inline",
        notes="Post-solve restitution impulse application.",
    ),
    OP_PH_FRICTION_APPLY: PhysicsOpcodeContract(
        opcode=OP_PH_FRICTION_APPLY,
        name="PH_FRICTION_APPLY",
        stack_in=("contact_count",),
        stack_out=(),
        kernel="inline",
        notes="Friction cone solve using clustered tangent axes.",
    ),
    OP_PH_ISLAND_WAKE: PhysicsOpcodeContract(
        opcode=OP_PH_ISLAND_WAKE,
        name="PH_ISLAND_WAKE",
        stack_in=("trigger_star_id",),
        stack_out=("woken_count",),
        kernel="physics_sleep_island.cu",
        notes="Wake sleeping islands on external event/impulse.",
    ),
    OP_PH_BODY_SPAWN: PhysicsOpcodeContract(
        opcode=OP_PH_BODY_SPAWN,
        name="PH_BODY_SPAWN",
        stack_in=("star_id", "pos", "vel"),
        stack_out=("body_idx",),
        kernel="physics_spawn.cu",
        notes="Spawn body from Layer-2 rigid-body star.",
    ),
    OP_PH_BODY_DESPAWN: PhysicsOpcodeContract(
        opcode=OP_PH_BODY_DESPAWN,
        name="PH_BODY_DESPAWN",
        stack_in=("body_idx",),
        stack_out=(),
        kernel="physics_spawn.cu",
        notes="Remove body and mark corresponding reality star dirty.",
    ),
    OP_PH_GRAVITY_APPLY: PhysicsOpcodeContract(
        opcode=OP_PH_GRAVITY_APPLY,
        name="PH_GRAVITY_APPLY",
        stack_in=("gravity_star_id",),
        stack_out=(),
        kernel="inline",
        notes="Fetch gravity constant from Reality Galaxy instead of hardcoding.",
    ),
    OP_PH_COLLISION_QUERY: PhysicsOpcodeContract(
        opcode=OP_PH_COLLISION_QUERY,
        name="PH_COLLISION_QUERY",
        stack_in=("ray_origin", "ray_dir"),
        stack_out=("body_idx", "t"),
        kernel="physics_raycast.cu",
        notes="Ray query against active rigid bodies.",
    ),
    OP_PH_TERNARY_CLASSIFY: PhysicsOpcodeContract(
        opcode=OP_PH_TERNARY_CLASSIFY,
        name="PH_TERNARY_CLASSIFY",
        stack_in=("body_idx",),
        stack_out=("sleep_state",),
        kernel="inline",
        notes="Classify awake/drowsy/sleeping using ternary state output (+1/0/-1).",
    ),
}


RESERVED_PHYSICS_OPCODE_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x163, 0x16F, "cloth_rope_xpbd"),
    (0x170, 0x177, "fluid_sph"),
    (0x178, 0x17F, "soft_body_xpbd"),
)


def physics_contract_by_name(name: str) -> PhysicsOpcodeContract:
    for contract in PHYSICS_OPCODE_CONTRACTS.values():
        if contract.name == name:
            return contract
    raise KeyError(name)


__all__ = [
    "PhysicsOpcodeContract",
    "PHYSICS_PHASE_SEQUENCE",
    "PHYSICS_OPCODE_CONTRACTS",
    "RESERVED_PHYSICS_OPCODE_RANGES",
    "physics_contract_by_name",
]
