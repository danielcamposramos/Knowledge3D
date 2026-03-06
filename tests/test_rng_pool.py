from __future__ import annotations

import numpy as np

from knowledge3d.gpu.rng_pool import RNGPool


def test_rng_pool_initializes_lazily_on_cpu() -> None:
    pool = RNGPool(seed=1234, use_cupy=False)

    assert pool._rng is None

    cpu, gpu = pool.uniform((4,), dtype="float32")

    assert gpu is None
    assert pool._rng is not None
    assert isinstance(cpu, np.ndarray)
    assert cpu.shape == (4,)


def test_rng_pool_seed_resets_lazy_state() -> None:
    pool = RNGPool(seed=7, use_cupy=False)
    first, _ = pool.uniform((3,), dtype="float32")

    pool.seed(7)
    assert pool._rng is None

    second, _ = pool.uniform((3,), dtype="float32")

    assert np.allclose(first, second)
