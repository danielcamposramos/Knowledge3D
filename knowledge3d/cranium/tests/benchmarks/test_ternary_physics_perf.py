"""Ternary operation performance benchmarks for Phase 4B."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_point_charge_2d,
    export_projectile_2d,
    export_rlc_circuit,
)


def test_benchmark_sign_vs_float_multiply() -> None:
    """Benchmark ternary SIGN vs float multiply for direction extraction."""
    print("\n" + "=" * 70)
    print("Benchmark: Ternary SIGN vs Float Multiply")
    print("=" * 70)

    # Setup
    velocities = np.random.randn(10000, 2).astype(np.float32)
    drag_factor = 0.1
    n_iterations = 1000

    # Binary approach (float multiply)
    start = time.perf_counter()
    for _ in range(n_iterations):
        ax_binary = -drag_factor * velocities[:, 0]
        ay_binary = -drag_factor * velocities[:, 1]
    t_binary = time.perf_counter() - start

    # Ternary approach (SIGN + multiply)
    start = time.perf_counter()
    for _ in range(n_iterations):
        sign_vx = np.sign(velocities[:, 0])
        sign_vy = np.sign(velocities[:, 1])
        ax_ternary = -sign_vx * drag_factor
        ay_ternary = -sign_vy * drag_factor
    t_ternary = time.perf_counter() - start

    speedup = t_binary / t_ternary

    print(f"\n  Binary (float multiply):  {t_binary:.4f}s")
    print(f"  Ternary (SIGN + multiply): {t_ternary:.4f}s")
    print(f"  Speedup: {speedup:.2f}×")

    if speedup > 1.0:
        print(f"  ✓ Ternary is {speedup:.2f}× faster!")
    elif speedup > 0.95:
        print(f"  ≈ Ternary is comparable (within 5%)")
    else:
        print(f"  ⚠ Ternary is slower by {1/speedup:.2f}×")

    # Verify correctness
    sign_vx = np.sign(velocities[:, 0])
    ax_binary_check = -drag_factor * velocities[:, 0]
    ax_ternary_check = -sign_vx * drag_factor

    # For non-zero velocities, directions should match
    nonzero_mask = velocities[:, 0] != 0
    binary_signs = np.sign(ax_binary_check[nonzero_mask])
    ternary_signs = np.sign(ax_ternary_check[nonzero_mask])
    assert np.all(binary_signs == ternary_signs), "Direction signs should match"

    print(f"\n  ✓ Correctness verified: directions match for all {nonzero_mask.sum()} non-zero velocities")


@pytest.mark.benchmark
def test_benchmark_reality_galaxy_ternary() -> None:
    """Benchmark full Reality Galaxy execution with ternary ops."""
    print("\n" + "=" * 70)
    print("Benchmark: Reality Galaxy Systems with Ternary Ops")
    print("=" * 70)

    systems = [
        ("Projectile2D (Ternary Drag)", export_projectile_2d()),
        ("PointCharge2D (Ternary Signs)", export_point_charge_2d()),
        ("RLCCircuit (Ternary Damping)", export_rlc_circuit()),
    ]

    for name, system in systems:
        galaxy = RealityGalaxy()
        galaxy.add_node(system)

        # Warm-up
        galaxy.step_system(system.node_id, n_steps=10)

        # Benchmark
        n_steps = 100
        n_iterations = 10
        start = time.perf_counter()
        for _ in range(n_iterations):
            galaxy.step_system(system.node_id, n_steps=1)
        t_elapsed = time.perf_counter() - start

        total_steps = n_iterations
        steps_per_sec = total_steps / t_elapsed
        latency_ms = t_elapsed / total_steps * 1000

        print(f"\n  {name}:")
        print(f"    Throughput: {steps_per_sec:.1f} steps/sec")
        print(f"    Latency:    {latency_ms:.3f} ms/step")
        print(f"    Tier:       {system.rpn_tier}, Instance: {system.rpn_instance}")

        # Target: sub-10ms for simple (Tier-1) systems
        if system.rpn_tier == 1:
            if latency_ms < 10:
                print(f"    ✓ Meets sub-10ms target for Tier-1")
            else:
                print(f"    ⚠ Exceeds 10ms target (Tier-1 should be faster)")


@pytest.mark.benchmark
def test_benchmark_multi_system_throughput() -> None:
    """Benchmark throughput with multiple systems running in parallel."""
    print("\n" + "=" * 70)
    print("Benchmark: Multi-System Parallel Throughput")
    print("=" * 70)

    galaxy = RealityGalaxy()

    # Add 13 systems (9 Phase 4A + 4 Phase 4B)
    systems = [
        export_projectile_2d(),
        export_point_charge_2d(),
        export_rlc_circuit(),
    ]

    for sys in systems:
        galaxy.add_node(sys)

    # Benchmark stepping all systems
    n_steps = 10
    start = time.perf_counter()

    for sys in systems:
        galaxy.step_system(sys.node_id, n_steps=n_steps)

    t_elapsed = time.perf_counter() - start

    total_steps = len(systems) * n_steps
    throughput = total_steps / t_elapsed
    avg_latency = t_elapsed / total_steps * 1000

    print(f"\n  Systems:          {len(systems)}")
    print(f"  Steps per system: {n_steps}")
    print(f"  Total steps:      {total_steps}")
    print(f"  Total time:       {t_elapsed:.3f}s")
    print(f"  Throughput:       {throughput:.1f} steps/sec")
    print(f"  Avg latency:      {avg_latency:.3f} ms/step")

    # Target: < 100ms total for stepping all 15 systems once
    if t_elapsed < 0.1:
        print(f"  ✓ Meets <100ms target for multi-system execution")
    else:
        print(f"  ⚠ Exceeds 100ms target")


if __name__ == "__main__":
    # Run benchmarks directly
    test_benchmark_sign_vs_float_multiply()
    test_benchmark_reality_galaxy_ternary()
    test_benchmark_multi_system_throughput()

    print("\n" + "=" * 70)
    print("Ternary Benchmarks Complete")
    print("=" * 70)
