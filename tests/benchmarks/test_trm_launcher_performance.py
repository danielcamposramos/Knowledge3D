from __future__ import annotations

import time

import numpy as np
import pytest

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher


def _ensure_cuda() -> None:
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA context unavailable: {exc}")


def _build_inputs(rng: np.random.Generator):
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)
    return q, y, z, W1, W2, W3, W4


def _time_launcher(use_rpn: bool, iterations: int) -> float:
    launcher = TRMLauncher(use_rpn=use_rpn)
    rng = np.random.default_rng(123)
    inputs = _build_inputs(rng)

    try:
        for _ in range(5):
            launcher.refine(*inputs)
        loader.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            launcher.refine(*inputs)
        loader.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed / iterations
    finally:
        launcher.cleanup()


def test_trm_launcher_rpn_vs_ptx_benchmark():
    _ensure_cuda()

    iterations = 10
    avg_ptx = _time_launcher(use_rpn=False, iterations=iterations)
    avg_rpn = _time_launcher(use_rpn=True, iterations=iterations)

    print(f"\nTRM PTX average latency: {avg_ptx * 1e3:.3f} ms")
    print(f"TRM RPN average latency: {avg_rpn * 1e3:.3f} ms")
    if avg_ptx > 0:
        print(f"Relative speedup (PTX/RPN): {avg_ptx / avg_rpn:.3f}×")

    assert avg_rpn > 0.0
    assert avg_ptx > 0.0
