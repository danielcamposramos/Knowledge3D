"""Minimal glTF export helpers for Reality systems."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_SHORT,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)
from knowledge3d.cranium.reality_nodes import RealitySystem
from knowledge3d.cranium.reality_physics_export import (
    export_acid_base_reaction,
    export_combustion,
    export_composite_material,
    export_constant_acceleration_1d,
    export_coupled_oscillators,
    export_crystal_lattice,
    export_double_pendulum_2d,
    export_dna_replication,
    export_enzyme_kinetics,
    export_harmonic_oscillator_1d,
    export_heat_1d,
    export_heat_2d,
    export_ideal_gas,
    export_lc_circuit,
    export_metal_melting,
    export_orbital_2d,
    export_phase_transition_water,
    export_population_dynamics,
    export_point_charge_2d,
    export_projectile_2d,
    export_rc_circuit,
    export_rigid_body_2d,
    export_rlc_circuit,
    export_simple_cell,
    export_co2_molecule,
    export_water_molecule,
)

EXPORT_ALL = [
    export_constant_acceleration_1d,
    export_harmonic_oscillator_1d,
    export_projectile_2d,
    export_rigid_body_2d,
    export_heat_1d,
    export_coupled_oscillators,
    export_orbital_2d,
    export_heat_2d,
    export_double_pendulum_2d,
    export_point_charge_2d,
    export_lc_circuit,
    export_rc_circuit,
    export_rlc_circuit,
    export_water_molecule,
    export_ideal_gas,
    export_combustion,
    export_co2_molecule,
    export_acid_base_reaction,
    export_phase_transition_water,
    export_simple_cell,
    export_enzyme_kinetics,
    export_dna_replication,
    export_population_dynamics,
    export_crystal_lattice,
    export_composite_material,
    export_metal_melting,
]


def export_system_to_gltf(system: RealitySystem, output_path: Path) -> Path:
    """Export a RealitySystem to a minimal, valid GLB with metadata.

    Geometry: a small triangle anchored at the system's state-derived position.
    Metadata: encoded in `gltf.extras` for downstream UI/inspection.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive a simple anchor from the system state (x/y/z if present).
    position_offset = _state_anchor(system.state)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    ) + position_offset
    indices = np.array([0, 1, 2], dtype=np.uint16)

    # Build binary buffers with 4-byte alignment.
    position_bytes = vertices.astype("<f4").tobytes()
    pos_pad = (4 - (len(position_bytes) % 4)) % 4
    position_bytes += b"\x00" * pos_pad

    index_bytes = indices.astype("<u2").tobytes()
    idx_pad = (4 - (len(index_bytes) % 4)) % 4
    index_offset = len(position_bytes)
    index_bytes += b"\x00" * idx_pad

    binary_blob = position_bytes + index_bytes

    gltf = GLTF2(asset=Asset(version="2.0"))
    gltf.buffers = [Buffer(byteLength=len(binary_blob))]
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=len(position_bytes), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=index_offset, byteLength=len(index_bytes), target=ELEMENT_ARRAY_BUFFER),
    ]

    min_xyz = vertices.min(axis=0).astype(float).tolist()
    max_xyz = vertices.max(axis=0).astype(float).tolist()
    gltf.accessors = [
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=FLOAT,
            count=vertices.shape[0],
            type="VEC3",
            min=min_xyz,
            max=max_xyz,
        ),
        Accessor(
            bufferView=1,
            byteOffset=0,
            componentType=UNSIGNED_SHORT,
            count=indices.shape[0],
            type="SCALAR",
            min=[int(indices.min())],
            max=[int(indices.max())],
        ),
    ]

    gltf.meshes = [Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1)])]
    gltf.nodes = [Node(mesh=0, name=system.node_id)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0

    gltf.extras = _system_extras(system)

    gltf.set_binary_blob(binary_blob)
    gltf.save_binary(str(output_path))
    return output_path


def generate_all_system_gltfs(output_dir: Path | str = Path("output/gltf")) -> None:
    """Generate GLBs for all baseline systems."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fn in EXPORT_ALL:
        sys_obj = fn()
        out = output_dir / f"{sys_obj.node_id.replace('system:', '')}.glb"
        export_system_to_gltf(sys_obj, out)


def _state_anchor(state: Mapping[str, float]) -> np.ndarray:
    """Extract a 3D anchor from common position-like state keys."""

    def pick(keys: List[str]) -> float:
        for k in keys:
            if k in state:
                return float(state[k])
        return 0.0

    x_val = pick(["x", "position_x"])
    y_val = pick(["y", "position_y"])
    z_val = pick(["z", "position_z"])
    return np.array([x_val, y_val, z_val], dtype=np.float32)


def _system_extras(system: RealitySystem) -> dict:
    """Serialize RealitySystem metadata into a JSON-friendly extras payload."""

    return {
        "node_id": system.node_id,
        "node_type": system.node_type,
        "rpn_tier": system.rpn_tier,
        "rpn_instance": system.rpn_instance,
        "matryoshka_dim": system.matryoshka_dim,
        "component_refs": list(system.component_refs),
        "state": {k: float(v) for k, v in system.state.items()},
        "law_rpn": system.law_rpn,
        "behavior_rpn": system.behavior_rpn,
    }


__all__ = ["export_system_to_gltf", "generate_all_system_gltfs", "EXPORT_ALL"]
