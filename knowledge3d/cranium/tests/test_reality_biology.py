"""Biology systems tests for Phase 4C Reality Enabler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_dna_replication,
    export_enzyme_kinetics,
    export_population_dynamics,
    export_simple_cell,
)


def test_cell_diffusion_equilibrium() -> None:
    galaxy = RealityGalaxy()
    system = export_simple_cell({"C_inside": 0.1, "C_outside": 1.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:simple_cell", n_steps=200)
    assert state["C_inside"] == pytest.approx(state["C_outside"], rel=0.05, abs=1e-3)


def test_cell_osmosis_direction() -> None:
    galaxy = RealityGalaxy()
    system = export_simple_cell({"C_inside": 0.2, "C_outside": 1.2})
    galaxy.add_node(system)
    before = system.state["C_inside"]
    state = galaxy.step_system("system:simple_cell", n_steps=5)
    assert state["C_inside"] > before


def test_enzyme_kinetics_rate() -> None:
    galaxy = RealityGalaxy()
    system = export_enzyme_kinetics({"S": 10.0, "Vmax": 2.0, "Km": 5.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:enzyme_kinetics", n_steps=1)
    expected_rate = 2.0 * 10.0 / (5.0 + 10.0)
    assert state["P"] == pytest.approx(expected_rate * 0.01, rel=0.05)


def test_enzyme_saturation() -> None:
    galaxy = RealityGalaxy()
    system = export_enzyme_kinetics({"S": 100.0, "Vmax": 1.5, "Km": 5.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:enzyme_kinetics", n_steps=1)
    assert state["P"] > 0.01  # near Vmax scaled by dt


def test_enzyme_mass_conservation() -> None:
    galaxy = RealityGalaxy()
    system = export_enzyme_kinetics({"S": 8.0, "P": 0.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:enzyme_kinetics", n_steps=50)
    total = state["S"] + state["P"] + state["ES"]
    assert total == pytest.approx(system.state["S_initial"], rel=0.1)


def test_dna_replication_rate() -> None:
    galaxy = RealityGalaxy()
    system = export_dna_replication({"template_length": 1000.0, "speed": 25.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:dna_replication", n_steps=4)
    assert state["bases_replicated"] == pytest.approx(100.0)
    assert state["polymerase_position"] == pytest.approx(100.0)


def test_dna_error_rate() -> None:
    galaxy = RealityGalaxy()
    system = export_dna_replication({"template_length": 200.0, "speed": 50.0, "error_rate": 1e-3})
    galaxy.add_node(system)
    state = galaxy.step_system("system:dna_replication", n_steps=2)
    expected_errors = 2 * 50.0 * 1e-3
    assert state["error_count"] == pytest.approx(expected_errors, rel=0.1)


def test_dna_bounds() -> None:
    galaxy = RealityGalaxy()
    system = export_dna_replication({"template_length": 120.0, "speed": 60.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:dna_replication", n_steps=2)
    assert state["bases_replicated"] <= 120.0
    assert state["polymerase_position"] <= 120.0


def test_population_growth() -> None:
    galaxy = RealityGalaxy()
    system = export_population_dynamics({"N_prey": 30.0, "N_predator": 2.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:population_dynamics", n_steps=5)
    assert state["N_prey"] > 30.0


def test_predator_prey_cycles() -> None:
    galaxy = RealityGalaxy()
    system = export_population_dynamics(
        {"N_prey": 80.0, "N_predator": 10.0, "predation_rate": 0.02, "growth_rate": 0.2}
    )
    galaxy.add_node(system)
    state = galaxy.step_system("system:population_dynamics", n_steps=3)
    assert state["N_predator"] > 10.0
    assert state["N_prey"] > 0.0
