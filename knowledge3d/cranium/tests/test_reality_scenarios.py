"""Integration tests for multi-domain scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_acid_base_reaction,
    export_combustion,
    export_composite_material,
    export_crystal_lattice,
    export_heat_1d,
    export_heat_2d,
    export_ideal_gas,
    export_metal_melting,
    export_population_dynamics,
    export_simple_cell,
    export_enzyme_kinetics,
    export_phase_transition_water,
)
from knowledge3d.cranium.reality_scenarios import (
    export_cell_metabolism_scenario,
    export_material_synthesis_scenario,
    export_ecosystem_scenario,
)


def test_cell_metabolism_integration() -> None:
    galaxy = RealityGalaxy()
    enzyme = export_enzyme_kinetics()
    diffusion = export_simple_cell()
    heat = export_heat_1d()
    buffer = export_acid_base_reaction()

    for sys_obj in [enzyme, diffusion, heat, buffer]:
        galaxy.add_node(sys_obj)

    scenario = export_cell_metabolism_scenario(
        {"component_refs": [enzyme.node_id, diffusion.node_id, heat.node_id, buffer.node_id]}
    )
    galaxy.add_node(scenario)

    state = galaxy.step_system(scenario.node_id, n_steps=20)
    assert state["temp_K"] > 310.0
    assert state["pH"] > 7.0


def test_material_synthesis_integration() -> None:
    galaxy = RealityGalaxy()
    combustion = export_combustion({"T": 1500.0})
    heat = export_heat_2d()
    melting = export_metal_melting({"T": 1400.0})
    lattice = export_crystal_lattice({"T": 300.0})
    for sys_obj in [combustion, heat, melting, lattice]:
        galaxy.add_node(sys_obj)

    scenario = export_material_synthesis_scenario(
        {"component_refs": [combustion.node_id, heat.node_id, melting.node_id, lattice.node_id], "combustion_energy": 1000.0}
    )
    galaxy.add_node(scenario)

    state = galaxy.step_system(scenario.node_id, n_steps=10)
    assert state["core_temp"] > 300.0
    assert state["phase"] >= -1.0


def test_ecosystem_dynamics_integration() -> None:
    galaxy = RealityGalaxy()
    pop = export_population_dynamics()
    gas = export_ideal_gas()
    heat = export_heat_1d()
    water = export_phase_transition_water({"T": 290.0})
    for sys_obj in [pop, gas, heat, water]:
        galaxy.add_node(sys_obj)

    scenario = export_ecosystem_scenario(
        {"component_refs": [pop.node_id, gas.node_id, heat.node_id, water.node_id], "resources": 1.0}
    )
    galaxy.add_node(scenario)
    state = galaxy.step_system(scenario.node_id, n_steps=5)
    assert state["resources"] > 1.0
    assert state["water_phase"] >= -1.0
