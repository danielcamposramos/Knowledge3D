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
    export_orbital_2d,
    export_projectile_2d,
    export_rigid_body_2d,
)


def test_tier1_simple_systems() -> None:
    galaxy = RealityGalaxy()
    systems = [
        export_constant_acceleration_1d(),
        export_harmonic_oscillator_1d(),
        export_projectile_2d(),
        export_rigid_body_2d(),
    ]
    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 1
        assert 0 <= sys.rpn_instance <= 3
        assert sys.matryoshka_dim in (64, 128)


def test_tier2_mid_systems() -> None:
    galaxy = RealityGalaxy()
    systems = [
        export_heat_1d(),
        export_coupled_oscillators(),
        export_orbital_2d(),
        export_heat_2d(),
    ]
    for sys in systems:
        galaxy.add_node(sys)
        assert sys.rpn_tier == 2
        assert 12 <= sys.rpn_instance <= 15
        assert sys.matryoshka_dim in (128, 512)


def test_tier3_high_systems() -> None:
    galaxy = RealityGalaxy()
    sys = export_double_pendulum_2d()
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
