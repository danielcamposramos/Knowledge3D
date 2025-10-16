from __future__ import annotations

import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_thinking_tag_parallel_rpn_benchmark() -> None:
    """Benchmark ThinkingTag RPN bridge against a naïve legacy loop."""
    bridge = ThinkingTagRPNBridge()
    rng = np.random.default_rng(321)

    input_dim = 512
    hidden1 = 256
    hidden2 = 256
    output_dim = 100

    input_vec = rng.standard_normal(input_dim).astype(np.float32)
    weights = {
        "W1": rng.standard_normal((hidden1, input_dim)).astype(np.float32),
        "W2": rng.standard_normal((hidden2, hidden1)).astype(np.float32),
        "W3": rng.standard_normal((output_dim, hidden2)).astype(np.float32),
    }
    mask = np.clip(rng.random(hidden2, dtype=np.float32), 0.0, 1.0)

    # GPU warm-up
    for _ in range(5):
        bridge.execute_temporal(input_vec, weights, mask=mask)

    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        bridge.execute_temporal(input_vec, weights, mask=mask)
    gpu_ms = (time.perf_counter() - start) / runs * 1000

    def legacy_forward(x: np.ndarray, w: dict[str, np.ndarray], m: np.ndarray) -> np.ndarray:
        h1 = np.zeros(w["W1"].shape[0], dtype=np.float32)
        for i in range(w["W1"].shape[0]):
            acc = 0.0
            row = w["W1"][i]
            for j in range(row.shape[0]):
                acc += row[j] * x[j]
            h1[i] = acc if acc > 0 else 0.0

        h2 = np.zeros(w["W2"].shape[0], dtype=np.float32)
        for i in range(w["W2"].shape[0]):
            acc = 0.0
            row = w["W2"][i]
            for j in range(row.shape[0]):
                acc += row[j] * h1[j]
            acc *= m[i]
            h2[i] = acc if acc > 0 else 0.0

        out = np.zeros(w["W3"].shape[0], dtype=np.float32)
        for i in range(w["W3"].shape[0]):
            acc = 0.0
            row = w["W3"][i]
            for j in range(row.shape[0]):
                acc += row[j] * h2[j]
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out

    legacy_forward(input_vec, weights, mask)
    start = time.perf_counter()
    for _ in range(3):
        legacy_forward(input_vec, weights, mask)
    legacy_ms = (time.perf_counter() - start) / 3 * 1000

    speedup = legacy_ms / gpu_ms if gpu_ms > 0 else float("inf")
    print(
        "\nThinkingTag temporal stage benchmark:\n"
        f"  GPU bridge:  {gpu_ms:.3f} ms\n"
        f"  Legacy loop: {legacy_ms:.3f} ms\n"
        f"  Speedup:     {speedup:.1f}×"
    )

    bridge.cleanup()
