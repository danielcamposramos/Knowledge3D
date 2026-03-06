from __future__ import annotations

from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool


def test_math_core_pool_snapshot_and_retier():
    pool = MathCorePool(idle_timeout=0.0)
    core_id = pool.spawn_core(tier=1, reuse=False)

    first = pool.snapshot()
    assert first["active"] >= 1
    assert first["active_tiers"][1] >= 1
    assert first["spawn_policy"] == "adaptive_reuse"
    assert pool.describe_tier(1) == "worker_worker"
    assert pool.describe_tier(2) == "worker"
    assert pool.describe_tier(3) == "master"

    pool.retier_core(core_id, tier=3)
    second = pool.snapshot()
    assert second["active_tiers"][3] >= 1

    pool.release_core(core_id)
    third = pool.snapshot()
    assert third["active"] == 0
