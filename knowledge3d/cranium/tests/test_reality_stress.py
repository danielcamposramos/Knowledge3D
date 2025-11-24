"""Stress tests for large numbers of concurrent Reality systems."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
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

EXPORTERS: List[Callable[[Dict | None], object]] = [
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


def _variant_params(idx: int, i: int) -> Dict[str, float]:
    """Lightweight parameter variation to avoid identical systems."""
    name = EXPORTERS[idx].__name__
    mod = (i % 5) + 1
    if "ideal_gas" in name:
        return {"T": 280.0 + 5 * mod, "V": 0.001 * (1 + 0.1 * mod)}
    if "combustion" in name:
        return {"T": 900.0 + 50 * mod}
    if "acid_base" in name:
        return {"n_HCl": 0.5 * mod, "n_NaOH": 0.4 * mod}
    if "water_phase" in name:
        return {"T": 250.0 + 30 * mod}
    if "simple_cell" in name:
        return {"C_inside": 0.05 * mod, "C_outside": 0.5 + 0.1 * mod}
    if "enzyme_kinetics" in name:
        return {"S": 5.0 * mod, "Vmax": 1.0 + 0.2 * mod}
    if "population_dynamics" in name:
        return {"N_prey": 20.0 + 5 * mod, "N_predator": 2.0 + mod}
    if "crystal_lattice" in name:
        return {"T": 250.0 + 20 * mod}
    if "composite_material" in name:
        return {"volume_fraction": 0.4 + 0.05 * (mod % 3)}
    if "metal_melting" in name:
        return {"T": 1200.0 + 20 * mod}
    return {}


def _spawn_systems(count: int, steps: int) -> float:
    pool = MathCorePool()
    pool.max_cores = max(pool.max_cores, count + 10)
    galaxy = RealityGalaxy(math_core_pool=pool)

    systems = []
    for i in range(count):
        idx = i % len(EXPORTERS)
        params = _variant_params(idx, i)
        sys_obj = EXPORTERS[idx](params=params, auto_allocate=True)
        sys_obj.node_id = f"{sys_obj.node_id}:{i}"
        galaxy.add_node(sys_obj)
        systems.append(sys_obj)

    core_ids = [s.rpn_instance for s in systems]
    assert len(set(core_ids)) == len(core_ids), "Core collision detected"

    start = time.perf_counter()
    for sys_obj in systems:
        galaxy.step_system(sys_obj.node_id, n_steps=steps)
    elapsed = time.perf_counter() - start
    return (count * steps) / max(elapsed, 1e-9)


@pytest.mark.slow
def test_100_systems_concurrent() -> None:
    throughput = _spawn_systems(count=100, steps=10)
    assert throughput > 1_000, f"Throughput too low for 100 systems: {throughput:.1f} steps/sec"


@pytest.mark.slow
def test_500_systems_concurrent() -> None:
    throughput = _spawn_systems(count=500, steps=5)
    assert throughput > 500, f"Throughput too low for 500 systems: {throughput:.1f} steps/sec"


@pytest.mark.slow
def test_1000_systems_concurrent() -> None:
    throughput = _spawn_systems(count=1000, steps=2)
    assert throughput > 200, f"Throughput too low for 1000 systems: {throughput:.1f} steps/sec"
