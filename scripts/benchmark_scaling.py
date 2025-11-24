"""Benchmark scaling of RealityGalaxy across varying system counts.

Outputs CSV with throughput and (optional) plots if matplotlib is available.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

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

EXPORTERS = [
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


def query_gpu_memory_mb() -> float:
    if pynvml is None:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024**2)
    except Exception:
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_sweep(system_counts: List[int]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for n in system_counts:
        pool = MathCorePool()
        pool.max_cores = max(pool.max_cores, n + 16)
        galaxy = RealityGalaxy(math_core_pool=pool)

        systems = []
        for i in range(n):
            fn = EXPORTERS[i % len(EXPORTERS)]
            sys_obj = fn()
            sys_obj.node_id = f"{sys_obj.node_id}:{i}"
            galaxy.add_node(sys_obj)
            systems.append(sys_obj)

        start = time.perf_counter()
        for sys_obj in systems:
            galaxy.step_system(sys_obj.node_id, n_steps=10)
        elapsed = time.perf_counter() - start

        throughput = (n * 10) / max(elapsed, 1e-9)
        latency_ms = (elapsed / (n * 10)) * 1000.0
        gpu_mb = query_gpu_memory_mb()

        results.append(
            {
                "system_count": n,
                "throughput_steps_per_sec": throughput,
                "latency_ms": latency_ms,
                "gpu_memory_mb": gpu_mb,
                "active_cores": len(pool.active_cores),
                "core_reuse_pct": 0.0,  # reuse tracking not yet exposed
            }
        )

        print(f"N={n:4d} | {throughput:10.1f} steps/sec | {latency_ms:6.3f} ms/step | {gpu_mb:6.1f} MB | cores={len(pool.active_cores)}")
    return results


def save_results_csv(results: List[Dict[str, float]], path: Path) -> None:
    fieldnames = [
        "system_count",
        "throughput_steps_per_sec",
        "latency_ms",
        "gpu_memory_mb",
        "active_cores",
        "core_reuse_pct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def maybe_plot(results: List[Dict[str, float]], out_path: Path) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot")
        return
    system_counts = [r["system_count"] for r in results]
    throughput = [r["throughput_steps_per_sec"] for r in results]
    gpu_mem = [r["gpu_memory_mb"] for r in results]
    cores = [r["active_cores"] for r in results]

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(system_counts, throughput, marker="o")
    axs[0].set_xlabel("System Count")
    axs[0].set_ylabel("Throughput (steps/sec)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    axs[1].plot(system_counts, gpu_mem, marker="s", color="orange")
    axs[1].set_xlabel("System Count")
    axs[1].set_ylabel("GPU Memory (MB)")

    axs[2].plot(system_counts, cores, marker="^", color="green")
    axs[2].set_xlabel("System Count")
    axs[2].set_ylabel("Active Cores")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main() -> None:
    system_counts = [1, 5, 10, 26, 50, 100, 250, 500, 1000]
    results = run_sweep(system_counts)
    save_results_csv(results, Path("benchmark_scaling.csv"))
    maybe_plot(results, Path("benchmark_scaling.png"))


if __name__ == "__main__":
    main()
