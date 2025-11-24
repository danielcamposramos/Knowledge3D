"""Materials systems tests for Phase 4C Reality Enabler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_composite_material,
    export_crystal_lattice,
    export_metal_melting,
)


def test_crystal_thermal_expansion() -> None:
    galaxy = RealityGalaxy()
    system = export_crystal_lattice({"T": 300.0})
    galaxy.add_node(system)
    base = galaxy.step_system("system:crystal_lattice", n_steps=1)["a"]
    system.state["T"] = 500.0
    warmed = galaxy.step_system("system:crystal_lattice", n_steps=1)["a"]
    assert warmed > base


def test_crystal_lattice_constant() -> None:
    galaxy = RealityGalaxy()
    system = export_crystal_lattice({"T": 300.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:crystal_lattice", n_steps=1)
    assert state["a"] == pytest.approx(3.61e-10, rel=0.05)


def test_crystal_linear_expansion_rate() -> None:
    galaxy = RealityGalaxy()
    system = export_crystal_lattice({"T": 350.0})
    galaxy.add_node(system)
    state1 = galaxy.step_system("system:crystal_lattice", n_steps=1)
    system.state["T"] = 400.0
    state2 = galaxy.step_system("system:crystal_lattice", n_steps=1)
    assert state2["a"] > state1["a"]


def test_composite_rule_of_mixtures() -> None:
    galaxy = RealityGalaxy()
    system = export_composite_material({"stress": 100.0, "volume_fraction": 0.6})
    galaxy.add_node(system)
    state = galaxy.step_system("system:composite_material", n_steps=1)
    expected_E = 0.6 * 200e9 + 0.4 * 3e9
    assert state["E_composite"] == pytest.approx(expected_E, rel=0.05)


def test_composite_stress_strain_relationship() -> None:
    galaxy = RealityGalaxy()
    system = export_composite_material({"stress": 150.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:composite_material", n_steps=1)
    assert state["strain"] > 0.0


def test_metal_melting_temperature() -> None:
    galaxy = RealityGalaxy()
    system = export_metal_melting({"T": 1500.0, "T_melting": 1356.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:metal_melting", n_steps=1)
    assert state["phase"] == pytest.approx(1.0, abs=0.05)


def test_metal_solid_state() -> None:
    galaxy = RealityGalaxy()
    system = export_metal_melting({"T": 300.0, "T_melting": 1356.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:metal_melting", n_steps=1)
    assert state["phase"] == pytest.approx(-1.0, abs=0.05)


def test_metal_transition_midpoint() -> None:
    galaxy = RealityGalaxy()
    system = export_metal_melting({"T": 1356.0, "T_melting": 1356.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:metal_melting", n_steps=1)
    assert state["phase"] == pytest.approx(0.0, abs=0.05)
