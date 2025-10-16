import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_op_matvec_f32_benchmark() -> None:
    bridge = ThinkingTagRPNBridge()

    rows, cols = 256, 512
    matrix = np.random.randn(rows, cols).astype(np.float32)
    vector = np.random.randn(cols).astype(np.float32)

    for _ in range(50):
        bridge._test_matvec(matrix, vector)

    runs = 200
    start = time.perf_counter()
    for _ in range(runs):
        bridge._test_matvec(matrix, vector)
    elapsed = (time.perf_counter() - start) * 1e6 / runs

    assert elapsed < 50.0, f"Expected < 50µs, got {elapsed:.2f}µs"

    bridge.cleanup()


@pytest.mark.gpu
def test_thinkingtag_fuse_benchmark() -> None:
    bridge = ThinkingTagRPNBridge()

    input_vec = np.random.randn(512).astype(np.float32)
    context = np.random.randn(64, 256).astype(np.float32)
    weights = {
        "W1": np.random.randn(256, 512).astype(np.float32),
        "W2": np.random.randn(256, 256).astype(np.float32),
        "W3": np.random.randn(100, 256).astype(np.float32),
    }

    for _ in range(50):
        mask, _, _ = bridge.compute_temporal_mask(context, threshold=0.5)
        bridge.execute_temporal(input_vec, weights, mask)

    runs = 200
    start = time.perf_counter()
    for _ in range(runs):
        mask, _, _ = bridge.compute_temporal_mask(context, threshold=0.5)
        bridge.execute_temporal(input_vec, weights, mask)
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs

    assert elapsed_ms < 0.20, f"Expected <0.20ms, got {elapsed_ms:.3f}ms"

    bridge.cleanup()
