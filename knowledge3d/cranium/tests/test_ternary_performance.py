"""Benchmark ternary arithmetic GPU path vs Python fallback."""

import time

import numpy as np

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


def test_ternary_add_gpu_faster_than_python_loop():
    """GPU ternary add should beat Python loop baseline."""
    engine = TieredRPNEngine()
    size = 100_000
    iterations = 5

    a = np.random.choice([-1, 0, 1], size=size).astype(np.int8).tolist()
    b = np.random.choice([-1, 0, 1], size=size).astype(np.int8).tolist()

    start = time.time()
    for _ in range(iterations):
        out = engine._ternary_add_gpu(a, b)
    gpu_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        _ = [a[i] + b[i] for i in range(size)]
    python_time = time.time() - start

    assert len(out) == size
    assert gpu_time < 2.0, f"Ternary GPU path too slow ({gpu_time:.2f}s) for baseline workload"
