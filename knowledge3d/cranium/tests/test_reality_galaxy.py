"""Tests for stacked Reality Galaxy architecture."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_nodes import RealityAtom, RealityMolecule, RealitySystem
from knowledge3d.cranium.reality_physics_bootstrap import (
    bootstrap_constant_acceleration_system,
    bootstrap_harmonic_oscillator_system,
    bootstrap_physics_atoms,
)


def test_reality_atom_creation() -> None:
    atom = RealityAtom(
        node_id="test_atom",
        visual_rpn="0.5 0.5 0.1 circle",
        behavior_rpn="inert",
        metadata={"mass": 1.0},
    )

    assert atom.node_id == "atom:test_atom"
    assert atom.node_type == "reality_atom"
    assert atom.visual_rpn.startswith("0.5")


def test_reality_system_with_component_refs() -> None:
    galaxy = RealityGalaxy()

    atom = RealityAtom(node_id="point_mass", metadata={"mass": 1.0})
    galaxy.add_node(atom)

    system = RealitySystem(
        node_id="pendulum",
        component_refs=["atom:point_mass"],
        state={"theta": 0.1, "omega": 0.0},
    )
    galaxy.add_node(system)

    components = galaxy.resolve_components(system)
    assert len(components) == 1
    assert components[0].node_id == "atom:point_mass"


def test_constant_acceleration_system_via_reality_galaxy(tmp_path) -> None:
    galaxy = RealityGalaxy(galaxy_path=tmp_path / "test_galaxy")
    bootstrap_constant_acceleration_system(galaxy)

    system_id = "system:constant_accel_1d"
    system = galaxy.get_node(system_id)
    assert isinstance(system, RealitySystem)

    n_steps = 100
    final_state = galaxy.step_system(system_id, n_steps=n_steps)

    dt = system.state["dt"]
    t = n_steps * dt
    x_true = 0.0 + 1.0 * t + 0.5 * (-9.81) * t * t
    v_true = 1.0 + (-9.81) * t

    assert math.isclose(final_state["v"], v_true, rel_tol=1e-6, abs_tol=1e-6)
    assert abs(final_state["x"] - x_true) < 0.05


def test_galaxy_persistence(tmp_path) -> None:
    galaxy_path = tmp_path / "persist_galaxy"
    galaxy1 = RealityGalaxy(galaxy_path=galaxy_path)

    bootstrap_physics_atoms(galaxy1)
    bootstrap_constant_acceleration_system(galaxy1)
    galaxy1.save_galaxy()

    galaxy2 = RealityGalaxy(galaxy_path=galaxy_path)
    galaxy2.load_galaxy()

    assert "atom:point_mass" in galaxy2.nodes
    assert "system:constant_accel_1d" in galaxy2.nodes

    system = galaxy2.get_node("system:constant_accel_1d")
    assert isinstance(system, RealitySystem)
    assert "x" in system.state
    assert system.behavior_rpn.strip() != ""


def test_stacking_pattern_consistency() -> None:
    galaxy = RealityGalaxy()

    atom_h = RealityAtom(node_id="H", metadata={"mass": 1.008})
    galaxy.add_node(atom_h)

    molecule_h2 = RealityMolecule(
        node_id="H2",
        component_refs=["atom:H", "atom:H"],
        behavior_rpn="covalent_bond nonpolar",
    )
    galaxy.add_node(molecule_h2)

    assert molecule_h2.node_type == "reality_molecule"
    assert all(ref == "atom:H" for ref in molecule_h2.component_refs)

    components = galaxy.resolve_components(molecule_h2)
    assert len(components) == 2
    assert all(c.node_id == "atom:H" for c in components)


def test_harmonic_oscillator_steps(tmp_path) -> None:
    galaxy = RealityGalaxy(galaxy_path=tmp_path / "harmonic")
    bootstrap_harmonic_oscillator_system(galaxy)

    system_id = "system:harmonic_osc_1d"
    initial = galaxy.get_node(system_id)
    assert isinstance(initial, RealitySystem)

    final_state = galaxy.step_system(system_id, n_steps=10)
    assert "x" in final_state and "v" in final_state and "a" in final_state
    # Energy should remain bounded for small dt; check values are finite.
    assert math.isfinite(final_state["x"])
    assert math.isfinite(final_state["v"])


def test_rpn_store_recall_semantics() -> None:
    galaxy = RealityGalaxy()
    system = RealitySystem(
        node_id="test_store_recall",
        state={"a": 5.0, "b": 3.0},
        behavior_rpn="a RECALL b RECALL + result STORE",
    )
    galaxy.add_node(system)
    final_state = galaxy.step_system("system:test_store_recall", n_steps=1)
    assert final_state["result"] == pytest.approx(8.0)


def test_law_rpn_validation_passes() -> None:
    galaxy = RealityGalaxy()
    system = RealitySystem(
        node_id="test_law_valid",
        state={"F": 10.0, "m": 2.0, "a": 5.0},
        law_rpn="F RECALL m RECALL / a RECALL - ABS 1e-6 LT",
    )
    galaxy.add_node(system)
    assert galaxy.validate_law("system:test_law_valid", system.state)


def test_law_rpn_validation_fails() -> None:
    galaxy = RealityGalaxy()
    system = RealitySystem(
        node_id="test_law_invalid",
        state={"F": 10.0, "m": 2.0, "a": 3.0},
        law_rpn="F RECALL m RECALL / a RECALL - ABS 1e-6 LT",
    )
    galaxy.add_node(system)
    assert not galaxy.validate_law("system:test_law_invalid", system.state)


def test_sovereign_feature_extraction() -> None:
    galaxy = RealityGalaxy()
    atom = RealityAtom(
        node_id="test_atom_feat",
        metadata={"mass": 1.008, "charge": 1},
        behavior_rpn="VALENCE_1",
    )
    galaxy.add_node(atom, encode_embedding=True)
    assert atom.embedding is not None
    features1 = galaxy._extract_node_features(atom)
    features2 = galaxy._extract_node_features(atom)
    assert np.allclose(features1, features2)


def test_gltf_export_roundtrip(tmp_path) -> None:
    galaxy = RealityGalaxy()
    atom = RealityAtom(node_id="export_atom", metadata={"mass": 1.0})
    system = RealitySystem(
        node_id="export_system",
        component_refs=["atom:export_atom"],
        state={"x": 1.0},
        behavior_rpn="x RECALL 0.1 + x STORE",
    )
    galaxy.add_node(atom)
    galaxy.add_node(system)
    gltf_path = tmp_path / "test_galaxy.glb"
    galaxy.export_to_gltf(gltf_path)
    assert gltf_path.exists()


def test_ternary_sign_and_quant() -> None:
    galaxy = RealityGalaxy()
    system = RealitySystem(
        node_id="test_ternary",
        state={"v": -0.05, "th": 0.1},
        behavior_rpn="v RECALL SIGN v_sign STORE v RECALL th RECALL TQUANT vq STORE",
    )
    galaxy.add_node(system)
    state = galaxy.step_system("system:test_ternary", n_steps=1)
    assert state["v_sign"] == -1.0
    # With threshold 0.1 and value -0.05 -> quantizes to 0.0
    assert state["vq"] == 0.0
