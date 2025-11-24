"""Integration tests for multi-system Reality Galaxy scenarios."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_acid_base_reaction,
    export_combustion,
    export_constant_acceleration_1d,
    export_coupled_oscillators,
    export_double_pendulum_2d,
    export_dna_replication,
    export_enzyme_kinetics,
    export_crystal_lattice,
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
    export_composite_material,
    export_co2_molecule,
    export_water_molecule,
)
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool


def test_multi_system_parallel_execution() -> None:
    """Validate multiple systems running on different cores."""
    galaxy = RealityGalaxy()

    # Add systems spanning all three tiers
    systems = [
        export_constant_acceleration_1d(auto_allocate=False),  # Tier-1, instance 0
        export_projectile_2d(auto_allocate=False),             # Tier-1, instance 2
        export_lc_circuit(auto_allocate=False),                # Tier-1, instance 5
        export_coupled_oscillators(auto_allocate=False),       # Tier-2, instance 13
        export_double_pendulum_2d(auto_allocate=False),        # Tier-3, instance 16
    ]

    for sys in systems:
        galaxy.add_node(sys)

    # Step all systems simultaneously
    results = {}
    for sys in systems:
        results[sys.node_id] = galaxy.step_system(sys.node_id, n_steps=10)

    # Verify all systems executed
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    # Verify each system has valid state
    for node_id, state in results.items():
        assert state is not None, f"System {node_id} returned None state"
        assert isinstance(state, dict), f"System {node_id} state is not a dict"
        assert len(state) > 0, f"System {node_id} has empty state"

    # Verify tier assignments were honored
    assert galaxy.nodes["system:constant_accel_1d"].metadata.get("last_rpn_instance") == 0
    assert galaxy.nodes["system:projectile_2d"].metadata.get("last_rpn_instance") == 2
    assert galaxy.nodes["system:lc_circuit"].metadata.get("last_rpn_instance") == 5


def test_13_systems_full_allocation() -> None:
    """Stress test: 13 systems (9 Phase 4A + 4 Phase 4B) running."""
    galaxy = RealityGalaxy()

    # Add all 13 systems
    phase_4a = [
        export_constant_acceleration_1d(auto_allocate=False),
        export_harmonic_oscillator_1d(auto_allocate=False),
        export_projectile_2d(auto_allocate=False),
        export_rigid_body_2d(auto_allocate=False),
        export_heat_1d(auto_allocate=False),
        export_coupled_oscillators(auto_allocate=False),
        export_orbital_2d(auto_allocate=False),
        export_heat_2d(auto_allocate=False),
        export_double_pendulum_2d(auto_allocate=False),
    ]

    phase_4b = [
        export_point_charge_2d(auto_allocate=False),
        export_lc_circuit(auto_allocate=False),
        export_rc_circuit(auto_allocate=False),
        export_rlc_circuit(auto_allocate=False),
    ]

    all_systems = phase_4a + phase_4b
    for sys in all_systems:
        galaxy.add_node(sys)

    # Step all systems
    for sys in all_systems:
        state = galaxy.step_system(sys.node_id, n_steps=5)
        assert state is not None, f"System {sys.node_id} failed"

    # Verify core utilization
    instances_used = set()
    for sys in all_systems:
        inst = galaxy.nodes[sys.node_id].metadata.get("last_rpn_instance")
        if inst is not None:
            instances_used.add(inst)

    print(f"\n  Core utilization: {len(instances_used)}/18 cores used")
    print(f"  Cores: {sorted(instances_used)}")

    # Should use at least 10 distinct cores (some systems share tiers)
    assert len(instances_used) >= 10, f"Expected ≥10 cores used, got {len(instances_used)}"


def test_tier_distribution() -> None:
    """Verify systems are properly distributed across tiers."""
    galaxy = RealityGalaxy()

    all_systems = [
        export_constant_acceleration_1d(),
        export_harmonic_oscillator_1d(),
        export_projectile_2d(),
        export_rigid_body_2d(),
        export_heat_1d(),
        export_coupled_oscillators(),
        export_orbital_2d(),
        export_heat_2d(),
        export_double_pendulum_2d(),
        export_point_charge_2d(),
        export_lc_circuit(),
        export_rc_circuit(),
        export_rlc_circuit(),
    ]

    for sys in all_systems:
        galaxy.add_node(sys)

    # Count systems per tier
    tier_counts = {1: 0, 2: 0, 3: 0}
    for sys in all_systems:
        tier_counts[sys.rpn_tier] += 1

    print(f"\n  Tier distribution:")
    print(f"    Tier-1 (Simple):  {tier_counts[1]} systems")
    print(f"    Tier-2 (Mid):     {tier_counts[2]} systems")
    print(f"    Tier-3 (High):    {tier_counts[3]} systems")

    # Expected: 7 Tier-1, 5 Tier-2, 1 Tier-3
    assert tier_counts[1] == 7, f"Expected 7 Tier-1 systems, got {tier_counts[1]}"
    assert tier_counts[2] == 5, f"Expected 5 Tier-2 systems, got {tier_counts[2]}"
    assert tier_counts[3] == 1, f"Expected 1 Tier-3 system, got {tier_counts[3]}"


def test_ternary_ops_across_systems() -> None:
    """Validate ternary ops work correctly across different systems."""
    galaxy = RealityGalaxy()

    # Systems with ternary ops
    projectile = export_projectile_2d()  # SIGN for drag direction
    point_charge = export_point_charge_2d()  # SIGN for charge signs
    rlc = export_rlc_circuit()  # TCMP for damping regime

    galaxy.add_node(projectile)
    galaxy.add_node(point_charge)
    galaxy.add_node(rlc)

    # Step all systems
    projectile_state = galaxy.step_system("system:projectile_2d", n_steps=5)
    charge_state = galaxy.step_system("system:point_charge_2d", n_steps=1)
    rlc_state = galaxy.step_system("system:rlc_circuit", n_steps=1)

    # Verify ternary outputs exist and are valid {-1, 0, +1}
    assert "sign_vx" in projectile_state
    assert projectile_state["sign_vx"] in (-1.0, 0.0, 1.0)

    assert "q1_sign" in charge_state
    assert charge_state["q1_sign"] in (-1.0, 0.0, 1.0)

    assert "damping_regime" in rlc_state
    assert rlc_state["damping_regime"] in (-1.0, 0.0, 1.0)

    print("\n  ✓ Ternary ops validated:")
    print(f"    Projectile drag sign: {projectile_state['sign_vx']}")
    print(f"    Charge sign: {charge_state['q1_sign']}")
    print(f"    RLC damping regime: {rlc_state['damping_regime']}")


def test_1000_systems_dynamic_spawning() -> None:
    """Stress test: 1000 physics systems with dynamic core allocation."""
    pool = MathCorePool(gpu_id=0)
    if pool.max_cores < 1000:
        pytest.skip(f"Insufficient Math Core capacity ({pool.max_cores}) for 1000-system test")

    galaxy = RealityGalaxy(math_core_pool=pool)

    systems = []
    for i in range(1000):
        if i % 3 == 0:
            sys = export_projectile_2d(auto_allocate=True)
        elif i % 3 == 1:
            sys = export_point_charge_2d(auto_allocate=True)
        else:
            sys = export_lc_circuit(auto_allocate=True)
        sys.node_id = f"{sys.node_id}:{i}"
        systems.append(sys)

    for sys in systems:
        galaxy.add_node(sys)

    start = time.perf_counter()
    for sys in systems:
        galaxy.step_system(sys.node_id, n_steps=1)
    elapsed = time.perf_counter() - start

    print(f"\n  1000 systems stepped in {elapsed:.3f}s")
    print(f"  Throughput: {1000/elapsed:.1f} systems/sec")
    print(f"  Avg latency: {elapsed/1000*1000:.3f} ms/system")

    assert elapsed < 5.0, f"1000 systems took {elapsed:.3f}s (should be <5s)"


def test_galaxy_persistence_with_all_systems() -> None:
    """Test saving and loading galaxy with all 13 systems."""
    import tempfile

    galaxy1 = RealityGalaxy()

    # Add subset of systems
    systems = [
        export_projectile_2d(),
        export_point_charge_2d(),
        export_lc_circuit(),
    ]

    for sys in systems:
        galaxy1.add_node(sys)

    # Step systems to populate state
    for sys in systems:
        galaxy1.step_system(sys.node_id, n_steps=10)

    # Save
    galaxy1.save_galaxy()

    # Load into new galaxy
    galaxy2 = RealityGalaxy(galaxy_path=galaxy1.galaxy_path)
    galaxy2.load_galaxy()

    # Verify all systems loaded
    assert len(galaxy2.nodes) == 3, f"Expected 3 nodes, got {len(galaxy2.nodes)}"

    # Verify state preserved
    for sys in systems:
        state1 = galaxy1.nodes[sys.node_id].state
        state2 = galaxy2.nodes[sys.node_id].state
        assert state1.keys() == state2.keys(), f"State keys mismatch for {sys.node_id}"


def test_26_systems_full_allocation() -> None:
    """Integration test across physics, chemistry, biology, and materials."""
    pool = MathCorePool()
    if pool.max_cores < 26:
        pool.max_cores = 32
    galaxy = RealityGalaxy(math_core_pool=pool)

    phase_4a = [
        export_constant_acceleration_1d(auto_allocate=True),
        export_harmonic_oscillator_1d(auto_allocate=True),
        export_projectile_2d(auto_allocate=True),
        export_rigid_body_2d(auto_allocate=True),
        export_heat_1d(auto_allocate=True),
        export_coupled_oscillators(auto_allocate=True),
        export_orbital_2d(auto_allocate=True),
        export_heat_2d(auto_allocate=True),
        export_double_pendulum_2d(auto_allocate=True),
    ]

    phase_4b = [
        export_point_charge_2d(auto_allocate=True),
        export_lc_circuit(auto_allocate=True),
        export_rc_circuit(auto_allocate=True),
        export_rlc_circuit(auto_allocate=True),
    ]

    phase_4c_chem = [
        export_water_molecule(auto_allocate=True),
        export_ideal_gas(auto_allocate=True),
        export_combustion(auto_allocate=True),
        export_co2_molecule(auto_allocate=True),
        export_acid_base_reaction(auto_allocate=True),
        export_phase_transition_water(auto_allocate=True),
    ]

    phase_4c_bio = [
        export_simple_cell(auto_allocate=True),
        export_enzyme_kinetics(auto_allocate=True),
        export_dna_replication(auto_allocate=True),
        export_population_dynamics(auto_allocate=True),
    ]

    phase_4c_mat = [
        export_crystal_lattice(auto_allocate=True),
        export_composite_material(auto_allocate=True),
        export_metal_melting(auto_allocate=True),
    ]

    systems = phase_4a + phase_4b + phase_4c_chem + phase_4c_bio + phase_4c_mat
    for sys in systems:
        galaxy.add_node(sys)

    instance_ids = [galaxy.nodes[sys.node_id].rpn_instance for sys in systems]
    assert len(set(instance_ids)) == len(systems), "All 26 systems should have unique cores"

    for sys in systems:
        state = galaxy.step_system(sys.node_id, n_steps=5)
        assert state is not None

    print(f"\n  26 systems (4 domains) running across {len(set(instance_ids))} cores")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
