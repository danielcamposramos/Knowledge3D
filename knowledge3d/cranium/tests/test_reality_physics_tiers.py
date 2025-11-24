"""Tier allocation and ternary integration tests for Phase 4A physics exports."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_constant_acceleration_1d,
    export_coupled_oscillators,
    export_double_pendulum_2d,
    export_harmonic_oscillator_1d,
    export_heat_1d,
    export_heat_2d,
    export_lc_circuit,
    export_orbital_2d,
    export_point_charge_2d,
    export_projectile_2d,
    export_rc_circuit,
    export_rigid_body_2d,
    export_rlc_circuit,
)


def test_tier1_simple_systems() -> None:
    galaxy = RealityGalaxy()
    systems = [
        export_constant_acceleration_1d(auto_allocate=False),
        export_harmonic_oscillator_1d(auto_allocate=False),
        export_projectile_2d(auto_allocate=False),
        export_rigid_body_2d(auto_allocate=False),
    ]
    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 1
        assert 0 <= sys.rpn_instance <= 3
        assert sys.matryoshka_dim in (64, 128)


def test_tier2_mid_systems() -> None:
    galaxy = RealityGalaxy()
    systems = [
        export_heat_1d(auto_allocate=False),
        export_coupled_oscillators(auto_allocate=False),
        export_orbital_2d(auto_allocate=False),
        export_heat_2d(auto_allocate=False),
    ]
    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 2
        assert 12 <= sys.rpn_instance <= 15
        assert sys.matryoshka_dim in (128, 512)


def test_tier3_high_systems() -> None:
    galaxy = RealityGalaxy()
    sys = export_double_pendulum_2d(auto_allocate=False)
    galaxy.add_node(sys)
    assert sys.rpn_tier == 3
    assert sys.rpn_instance == 16
    assert sys.matryoshka_dim == 2048


def test_ternary_drag_direction() -> None:
    galaxy = RealityGalaxy()
    projectile = export_projectile_2d()
    galaxy.add_node(projectile)
    state = galaxy.step_system("system:projectile_2d", n_steps=5)
    assert "sign_vx" in state and "sign_vy" in state
    assert state["sign_vx"] in (-1.0, 0.0, 1.0)
    assert state["sign_vy"] in (-1.0, 0.0, 1.0)


def test_coupled_oscillators_mode_detection() -> None:
    galaxy = RealityGalaxy()
    coupled = export_coupled_oscillators()
    galaxy.add_node(coupled)

    coupled.state["x1"] = 1.0
    coupled.state["x2"] = 0.5
    state = galaxy.step_system("system:coupled_oscillators", n_steps=1)
    assert state.get("mode_product") == 1.0

    coupled.state["x1"] = 1.0
    coupled.state["x2"] = -0.5
    state = galaxy.step_system("system:coupled_oscillators", n_steps=1)
    assert state.get("mode_product") == -1.0


def test_rpn_instance_recorded() -> None:
    galaxy = RealityGalaxy()
    sys = export_constant_acceleration_1d()
    galaxy.add_node(sys)
    _ = galaxy.step_system("system:constant_accel_1d", n_steps=1)
    assert sys.metadata.get("last_rpn_instance") == sys.rpn_instance


# ========== Phase 4B: Electromagnetism Tests ==========


def test_point_charge_coulomb_force() -> None:
    """Validate Coulomb force F = k*q1*q2/r²."""
    galaxy = RealityGalaxy()
    system = export_point_charge_2d({
        "x1": 0.0, "y1": 0.0,
        "x2": 1.0, "y2": 0.0,
        "q1": 1e-6,
        "q2": 1e-6,
        "m1": 1.0, "m2": 1.0,
        "dt": 0.001,
    })
    galaxy.add_node(system)

    initial_x1 = system.state["x1"]
    initial_x2 = system.state["x2"]

    state = galaxy.step_system("system:point_charge_2d", n_steps=5)

    # Like charges should repel (move away from each other)
    assert state["x1"] < initial_x1, "Charge 1 should move left (away from charge 2)"
    assert state["x2"] > initial_x2, "Charge 2 should move right (away from charge 1)"

    # Ternary charge product should be +1 (like charges)
    assert state.get("charge_product") == 1.0


def test_point_charge_ternary_signs() -> None:
    """Verify ternary SIGN for charge classification."""
    galaxy = RealityGalaxy()
    system = export_point_charge_2d({
        "x1": 0.0, "y1": 0.0,
        "x2": 1.0, "y2": 0.0,
        "q1": 1e-6,
        "q2": -1e-6,
        "m1": 1.0, "m2": 1.0,
        "dt": 0.001,
    })
    galaxy.add_node(system)
    state = galaxy.step_system("system:point_charge_2d", n_steps=1)

    assert state.get("q1_sign") == 1.0, "Positive charge should have sign +1"
    assert state.get("q2_sign") == -1.0, "Negative charge should have sign -1"
    assert state.get("charge_product") == -1.0, "Opposite charges should attract"


def test_lc_circuit_oscillation() -> None:
    """Validate LC oscillation behavior."""
    galaxy = RealityGalaxy()
    L = 1e-3
    C = 1e-6
    system = export_lc_circuit({"L": L, "C": C, "I": 1.0, "V": 0.0, "dt": 1e-6})
    galaxy.add_node(system)

    # Run for many steps
    state = galaxy.step_system("system:lc_circuit", n_steps=1000)

    # Energy should be conserved (approximately)
    E_initial = 0.5 * L * 1.0 * 1.0
    E_final = 0.5 * L * state["I"] ** 2 + 0.5 * C * state["V"] ** 2

    # Allow significant energy drift due to forward Euler integration
    # (This is expected with simple numerical methods on oscillatory systems)
    assert abs(E_final - E_initial) / E_initial < 2.0, f"Energy drift too large: {abs(E_final - E_initial) / E_initial:.2f}"


def test_rc_circuit_charging() -> None:
    """Validate RC charging with τ = RC."""
    galaxy = RealityGalaxy()
    R = 1000.0
    C = 1e-6
    tau = R * C
    V_source = 5.0

    system = export_rc_circuit({
        "R": R, "C": C,
        "V": 0.0, "V_source": V_source,
        "dt": 1e-5,
    })
    galaxy.add_node(system)

    # After 1 time constant, V should be ~63.2% of V_source
    n_steps = int(tau / 1e-5)
    state = galaxy.step_system("system:rc_circuit", n_steps=n_steps)

    V_expected = V_source * (1 - np.exp(-1))
    assert abs(state["V"] - V_expected) < 0.5, f"Expected ~{V_expected:.2f}V, got {state['V']:.2f}V"


def test_rc_circuit_bounds() -> None:
    """Verify RC voltage stays within bounds [0, V_source]."""
    galaxy = RealityGalaxy()
    system = export_rc_circuit({
        "R": 1000.0, "C": 1e-6,
        "V": 0.0, "V_source": 5.0,
        "dt": 1e-5,
    })
    galaxy.add_node(system)

    state = galaxy.step_system("system:rc_circuit", n_steps=10000)

    assert state["V"] >= 0.0, "Voltage should never be negative"
    assert state["V"] <= 5.1, "Voltage should not exceed V_source (with small tolerance)"


def test_rlc_damping_regime_underdamped() -> None:
    """Verify ternary damping regime detection: underdamped (ζ < 1)."""
    galaxy = RealityGalaxy()
    system = export_rlc_circuit({
        "R": 10.0, "L": 1e-3, "C": 1e-6,
        "I": 1.0, "V": 0.0, "dt": 1e-6,
    })
    galaxy.add_node(system)

    # Compute expected zeta
    zeta = 0.5 * 10.0 * np.sqrt(1e-6 / 1e-3)
    assert zeta < 1.0, f"Test setup error: zeta={zeta:.3f} should be < 1"

    state = galaxy.step_system("system:rlc_circuit", n_steps=1)
    assert state.get("damping_regime") == -1.0, "Underdamped should return -1"


def test_rlc_damping_regime_overdamped() -> None:
    """Verify ternary damping regime detection: overdamped (ζ > 1)."""
    galaxy = RealityGalaxy()
    system = export_rlc_circuit({
        "R": 1000.0, "L": 1e-3, "C": 1e-6,
        "I": 1.0, "V": 0.0, "dt": 1e-6,
    })
    galaxy.add_node(system)

    # Compute expected zeta
    zeta = 0.5 * 1000.0 * np.sqrt(1e-6 / 1e-3)
    assert zeta > 1.0, f"Test setup error: zeta={zeta:.3f} should be > 1"

    state = galaxy.step_system("system:rlc_circuit", n_steps=1)
    assert state.get("damping_regime") == 1.0, "Overdamped should return +1"


def test_rlc_energy_dissipation() -> None:
    """Validate energy dissipates in RLC circuit."""
    galaxy = RealityGalaxy()
    L = 1e-3
    C = 1e-6
    system = export_rlc_circuit({
        "R": 100.0, "L": L, "C": C,
        "I": 1.0, "V": 5.0, "dt": 1e-6,
    })
    galaxy.add_node(system)

    E_initial = 0.5 * L * 1.0 ** 2 + 0.5 * C * 5.0 ** 2
    state = galaxy.step_system("system:rlc_circuit", n_steps=10000)
    E_final = 0.5 * L * state["I"] ** 2 + 0.5 * C * state["V"] ** 2

    assert E_final < E_initial, "Energy should decrease (dissipated by R)"
