from __future__ import annotations

"""
Deterministic GPU-friendly RNG pool.

The Step7.2 plan calls for a seedable warp-local RNG pool that can be shared
by kernels such as ``dialogue_sampler``.  This Python implementation provides
the same behaviour without requiring a compiled CUDA extension yet; it keeps
state in CuPy/NumPy RNGs so we can rely on deterministic sequences during
development and unit tests.  When a low-level cubin version lands it can slot
behind the same interface.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional CuPy dependency
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _HAS_CUPY = False


DEFAULT_SEED = 0x1234_5678_90AB_CDEF


@dataclass
class RNGState:
    seed: int
    position: int = 0


class RNGPool:
    """
    Deterministic RNG pool shared across PTX helpers.

    The generator exposes a small API used by ``ptx_ops``.  CuPy is preferred
    when available so we keep all samples on device; otherwise NumPy is used,
    which keeps the tests runnable on CPU-only environments.
    """

    def __init__(self, seed: int = DEFAULT_SEED, *, use_cupy: Optional[bool] = None) -> None:
        self._state = RNGState(seed=seed)
        self._lock = Lock()
        if use_cupy is None:
            use_cupy = _HAS_CUPY
        self._use_cupy = bool(use_cupy and _HAS_CUPY)
        if self._use_cupy:
            self._rng = cp.random.RandomState(seed)
        else:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Reset the generator to a given seed."""
        with self._lock:
            self._state = RNGState(seed=seed)
            if self._use_cupy:
                self._rng = cp.random.RandomState(seed)
            else:
                self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def uniform(self, shape: Iterable[int], dtype: str = "float32") -> Tuple[np.ndarray, Optional["cp.ndarray"]]:
        """
        Generate uniform samples in ``[0, 1)``.

        Returns a tuple ``(cpu_array, gpu_array)``; one of them will be ``None``
        depending on whether CuPy is available.  This keeps call sites simple
        without forcing data transfers.
        """
        with self._lock:
            if self._use_cupy:
                gpu = self._rng.random_sample(shape, dtype=dtype)  # type: ignore[arg-type]
                cpu = None
            else:
                gpu = None
                cpu = self._rng.random(shape, dtype=dtype)  # type: ignore[arg-type]
        if gpu is not None:
            return gpu.get(), gpu
        assert cpu is not None
        return cpu, None

    # ------------------------------------------------------------------
    def integers(self, low: int, high: int, size: Iterable[int]) -> Tuple[np.ndarray, Optional["cp.ndarray"]]:
        """Sample integers in ``[low, high)`` deterministically."""
        with self._lock:
            if self._use_cupy:
                gpu = self._rng.randint(low, high, size=size, dtype=cp.int32)
                cpu = None
            else:
                gpu = None
                cpu = self._rng.integers(low, high, size, dtype=np.int32)
        if gpu is not None:
            return gpu.get(), gpu
        assert cpu is not None
        return cpu, None


# Global pool shared by PTX helpers
global_rng_pool = RNGPool()

