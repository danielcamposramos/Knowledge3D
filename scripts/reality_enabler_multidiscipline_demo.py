"""Demo: run all 26 Reality Enabler systems across four scientific domains."""

from __future__ import annotations

import time

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


def main() -> None:
    """Run Reality Enabler multi-discipline showcase."""
    pool = MathCorePool()
    if pool.max_cores < 32:
        pool.max_cores = 32
    galaxy = RealityGalaxy(math_core_pool=pool)

    systems = [
        # Physics (13)
        ("Constant Acceleration 1D", export_constant_acceleration_1d()),
        ("Harmonic Oscillator 1D", export_harmonic_oscillator_1d()),
        ("Projectile 2D", export_projectile_2d()),
        ("Rigid Body 2D", export_rigid_body_2d()),
        ("Heat 1D", export_heat_1d()),
        ("Coupled Oscillators", export_coupled_oscillators()),
        ("Orbital 2D", export_orbital_2d()),
        ("Heat 2D", export_heat_2d()),
        ("Double Pendulum 2D", export_double_pendulum_2d()),
        ("Point Charge 2D", export_point_charge_2d()),
        ("LC Circuit", export_lc_circuit()),
        ("RC Circuit", export_rc_circuit()),
        ("RLC Circuit", export_rlc_circuit()),
        # Chemistry (6)
        ("Water Molecule (H2O)", export_water_molecule()),
        ("Ideal Gas (PV=nRT)", export_ideal_gas()),
        ("Combustion (CH4)", export_combustion()),
        ("CO2 Molecule", export_co2_molecule()),
        ("Acid-Base Reaction", export_acid_base_reaction()),
        ("Phase Transition (H2O)", export_phase_transition_water()),
        # Biology (4)
        ("Cell Diffusion", export_simple_cell()),
        ("Enzyme Kinetics", export_enzyme_kinetics()),
        ("DNA Replication", export_dna_replication()),
        ("Population Dynamics", export_population_dynamics()),
        # Materials (3)
        ("Crystal Lattice (Cu)", export_crystal_lattice()),
        ("Composite Material", export_composite_material()),
        ("Metal Melting", export_metal_melting()),
    ]

    for name, system in systems:
        galaxy.add_node(system)
        print(f"✓ {name:<45} T{system.rpn_tier} C{system.rpn_instance:2d}")

    start = time.perf_counter()
    for _, system in systems:
        galaxy.step_system(system.node_id, n_steps=10)
    elapsed = time.perf_counter() - start

    total_steps = len(systems) * 10
    print(f"\n26 systems (4 domains) stepped 10× in {elapsed:.3f}s")
    print(f"Throughput: {total_steps/elapsed:.1f} steps/sec")


if __name__ == "__main__":
    main()
