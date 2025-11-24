"""Chemistry systems tests for Phase 4C Reality Enabler."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_acid_base_reaction,
    export_combustion,
    export_co2_molecule,
    export_ideal_gas,
    export_phase_transition_water,
    export_water_molecule,
)


def _bond_length(state: dict[str, float], atom: str) -> float:
    dx = state[f"x_{atom}"] - state["x_O"]
    dy = state[f"y_{atom}"] - state["y_O"]
    dz = state[f"z_{atom}"] - state["z_O"]
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def test_water_molecule_bond_vibration() -> None:
    galaxy = RealityGalaxy()
    system = export_water_molecule()
    galaxy.add_node(system)

    initial_r1 = _bond_length(system.state, "H1")
    state = galaxy.step_system("system:water_molecule", n_steps=20)
    r1 = _bond_length(state, "H1")
    r2 = _bond_length(state, "H2")

    assert abs(r1 - 0.96) < 0.1
    assert abs(r2 - 0.96) < 0.1
    assert abs(r1 - initial_r1) < 0.2


def test_water_molecule_angle_preservation() -> None:
    galaxy = RealityGalaxy()
    system = export_water_molecule()
    galaxy.add_node(system)
    state = galaxy.step_system("system:water_molecule", n_steps=10)

    v1 = np.array(
        [state["x_H1"] - state["x_O"], state["y_H1"] - state["y_O"], state["z_H1"] - state["z_O"]]
    )
    v2 = np.array(
        [state["x_H2"] - state["x_O"], state["y_H2"] - state["y_O"], state["z_H2"] - state["z_O"]]
    )
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_ang = float(np.clip(cos_ang, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_ang))

    assert 90.0 < angle_deg < 120.0
    assert abs(angle_deg - 104.5) < 10.0


def test_water_molecule_energy_conservation() -> None:
    galaxy = RealityGalaxy()
    system = export_water_molecule()
    galaxy.add_node(system)

    state = galaxy.step_system("system:water_molecule", n_steps=30)
    drift = abs(state["E_total"] - state["E_initial"]) / (abs(state["E_initial"]) + 1e-9)
    assert drift < 0.2


def test_ideal_gas_law_holds() -> None:
    galaxy = RealityGalaxy()
    system = export_ideal_gas({"P": 101325.0, "V": 0.01, "n": 1.0, "T": 300.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:ideal_gas", n_steps=1)
    ratio = state["P"] * state["V"] / (state["n"] * state["R"] * state["T"])
    assert abs(ratio - 1.0) < 0.02


def test_ideal_gas_isothermal_expansion() -> None:
    galaxy = RealityGalaxy()
    system = export_ideal_gas({"P": 101325.0, "V": 0.002, "n": 1.0, "T": 300.0})
    galaxy.add_node(system)
    state1 = galaxy.step_system("system:ideal_gas", n_steps=1)
    system.state["V"] = 0.004
    state2 = galaxy.step_system("system:ideal_gas", n_steps=1)
    assert state2["P"] < state1["P"]
    assert state2["P"] == pytest.approx(state1["P"] * 0.5, rel=0.2)


def test_ideal_gas_adiabatic_compression() -> None:
    galaxy = RealityGalaxy()
    system = export_ideal_gas({"P": 101325.0, "V": 0.004, "n": 1.0, "T": 300.0})
    galaxy.add_node(system)
    _ = galaxy.step_system("system:ideal_gas", n_steps=1)
    system.state["V"] = 0.002
    state = galaxy.step_system("system:ideal_gas", n_steps=1)
    assert state["P"] > 101325.0


def test_ideal_gas_temperature_scaling() -> None:
    galaxy = RealityGalaxy()
    system = export_ideal_gas({"P": 50000.0, "V": 0.002, "n": 1.0, "T": 250.0})
    galaxy.add_node(system)
    state1 = galaxy.step_system("system:ideal_gas", n_steps=1)
    system.state["T"] = 500.0
    state2 = galaxy.step_system("system:ideal_gas", n_steps=1)
    assert state2["P"] > state1["P"]


def test_combustion_activation_energy() -> None:
    galaxy = RealityGalaxy()
    system = export_combustion({"T": 900.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:combustion_ch4", n_steps=5)
    assert state["n_CH4"] == pytest.approx(1.0, rel=0.05)
    assert state["E_released"] == pytest.approx(0.0, rel=0.05, abs=1e-6)


def test_combustion_stoichiometry() -> None:
    galaxy = RealityGalaxy()
    system = export_combustion({"T": 1500.0, "n_CH4": 1.0, "n_O2": 2.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:combustion_ch4", n_steps=100)
    produced_co2 = 1.0 - state["n_CH4"]
    produced_h2o = state["n_H2O"]
    assert produced_co2 >= 0.0
    assert produced_h2o == pytest.approx(2 * produced_co2, rel=0.2)
    assert state["n_O2"] >= 0.0


def test_combustion_energy_release() -> None:
    galaxy = RealityGalaxy()
    system = export_combustion({"T": 1500.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:combustion_ch4", n_steps=10)
    assert state["E_released"] > 0.0


def test_co2_linear_geometry() -> None:
    galaxy = RealityGalaxy()
    system = export_co2_molecule()
    galaxy.add_node(system)
    state = galaxy.step_system("system:co2_molecule", n_steps=10)

    assert abs(state["y_O1"]) < 1e-6 and abs(state["y_O2"]) < 1e-6
    assert state["x_O1"] < state["x_C"] < state["x_O2"]
    sep_left = abs(state["x_O1"] - state["x_C"])
    sep_right = abs(state["x_O2"] - state["x_C"])
    assert sep_left == pytest.approx(sep_right, rel=0.1)


def test_acid_base_neutralization() -> None:
    galaxy = RealityGalaxy()
    system = export_acid_base_reaction({"n_HCl": 1.0, "n_NaOH": 1.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:acid_base", n_steps=40)
    assert state["n_HCl"] < 0.2
    assert state["n_NaOH"] < 0.2
    assert state["n_NaCl"] > 0.5
    assert abs(state["pH"] - 7.0) < 1.0


def test_acid_base_ph_direction() -> None:
    galaxy = RealityGalaxy()
    acidic = export_acid_base_reaction({"n_HCl": 2.0, "n_NaOH": 0.5})
    basic = export_acid_base_reaction({"n_HCl": 0.5, "n_NaOH": 2.0})
    acidic.node_id = "system:acid_base_acidic"
    basic.node_id = "system:acid_base_basic"
    galaxy.add_node(acidic)
    galaxy.add_node(basic)

    acid_state = galaxy.step_system(acidic.node_id, n_steps=10)
    base_state = galaxy.step_system(basic.node_id, n_steps=10)

    assert acid_state["pH"] < 7.0
    assert base_state["pH"] > 7.0


def test_acid_base_charge_balance() -> None:
    galaxy = RealityGalaxy()
    system = export_acid_base_reaction({"n_HCl": 1.5, "n_NaOH": 1.0})
    galaxy.add_node(system)
    state = galaxy.step_system("system:acid_base", n_steps=20)
    total_in = 1.5 + 1.0
    total_out = state["n_H2O"] + state["n_NaCl"] + state["n_HCl"] + state["n_NaOH"]
    assert total_out == pytest.approx(total_in, rel=0.1)


def test_water_phase_transition_states() -> None:
    galaxy = RealityGalaxy()
    system = export_phase_transition_water({"T": 250.0})
    galaxy.add_node(system)

    ice_state = galaxy.step_system("system:water_phase", n_steps=1)
    assert ice_state["phase"] == pytest.approx(-1.0, abs=0.05)

    system.state["T"] = 300.0
    liquid_state = galaxy.step_system("system:water_phase", n_steps=1)
    assert liquid_state["phase"] == pytest.approx(0.0, abs=0.1)

    system.state["T"] = 400.0
    vapor_state = galaxy.step_system("system:water_phase", n_steps=1)
    assert vapor_state["phase"] == pytest.approx(1.0, abs=0.05)
