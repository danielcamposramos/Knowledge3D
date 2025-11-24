"""Tests for dynamic Math Core pool management."""

from __future__ import annotations

import pytest

pytest.importorskip("cupy", reason="CuPy required for MathCorePool tests")

from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import export_projectile_2d


def test_math_core_pool_query_capacity() -> None:
    pool = MathCorePool(gpu_id=0)
    if pool.max_cores <= MathCorePool.FALLBACK_MAX_CORES:
        pytest.skip("MathCorePool using fallback capacity; GPU query unavailable")
    assert pool.max_cores > MathCorePool.FALLBACK_MAX_CORES
    assert pool.max_cores < 10000


def test_math_core_pool_spawn_100_cores() -> None:
    pool = MathCorePool(gpu_id=0)
    if pool.max_cores < 100:
        pytest.skip(f"Insufficient capacity ({pool.max_cores}) for 100 cores")

    core_ids = [pool.spawn_core(tier=1) for _ in range(100)]

    assert len(set(core_ids)) == 100
    assert len(pool.active_cores) == 100


def test_math_core_pool_release_and_reuse() -> None:
    pool = MathCorePool(gpu_id=0)

    core_id = pool.spawn_core(tier=1)
    pool.release_core(core_id, pool=True)

    core_id2 = pool.spawn_core(tier=1, reuse=True)
    assert core_id == core_id2


def test_math_core_pool_capacity_limit() -> None:
    pool = MathCorePool(gpu_id=0)
    pool.max_cores = 3

    for _ in range(pool.max_cores):
        pool.spawn_core(tier=1)

    with pytest.raises(RuntimeError, match="at capacity"):
        pool.spawn_core(tier=1)


def test_reality_galaxy_dynamic_allocation() -> None:
    pool = MathCorePool(gpu_id=0)
    if pool.max_cores < 50:
        pytest.skip(f"Insufficient capacity ({pool.max_cores}) for 50 systems")

    galaxy = RealityGalaxy(math_core_pool=pool)

    systems = []
    for i in range(50):
        sys = export_projectile_2d(auto_allocate=True)
        sys.node_id = f"{sys.node_id}:{i}"
        systems.append(sys)

    for sys in systems:
        galaxy.add_node(sys)

    instance_ids = [galaxy.nodes[sys.node_id].rpn_instance for sys in systems]
    assert len(set(instance_ids)) == len(systems)

    for sys in systems:
        state = galaxy.step_system(sys.node_id, n_steps=1)
        assert state is not None
