import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_swarm_bridge import NineChainSwarmBridge


@pytest.mark.gpu
def test_swarm_latency_budget() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)

    for _ in range(100):
        bridge.execute_swarm(input_vec, num_iterations=3)

    runs = 500
    start = time.perf_counter()
    for _ in range(runs):
        bridge.execute_swarm(input_vec, num_iterations=3)
    elapsed_us = (time.perf_counter() - start) * 1e6 / runs

    print(f"\n9-chain swarm latency: {elapsed_us:.2f} µs")
    assert elapsed_us < 95.0

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_iteration_scaling() -> None:
    bridge = NineChainSwarmBridge()
    input_vec = np.random.randn(64).astype(np.float32)

    timings = {}
    for iters in (1, 3, 5):
        for _ in range(50):
            bridge.execute_swarm(input_vec, num_iterations=iters)
        runs = 300
        start = time.perf_counter()
        for _ in range(runs):
            bridge.execute_swarm(input_vec, num_iterations=iters)
        timings[iters] = (time.perf_counter() - start) * 1e6 / runs
        print(f"{iters} iterations -> {timings[iters]:.2f} µs")

    assert timings[3] > timings[1]
    assert timings[5] > timings[3]

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_parallel_efficiency_estimate() -> None:
    bridge = NineChainSwarmBridge()
    input_vec = np.random.randn(64).astype(np.float32)

    for _ in range(100):
        bridge.execute_swarm(input_vec, num_iterations=3)

    runs = 500
    start = time.perf_counter()
    for _ in range(runs):
        bridge.execute_swarm(input_vec, num_iterations=3)
    swarm_latency = (time.perf_counter() - start) * 1e6 / runs

    estimated_single = swarm_latency / bridge.NUM_CHAINS
    print(f"\nSwarm latency: {swarm_latency:.2f} µs")
    print(f"Estimated single-chain: {estimated_single:.2f} µs")

    bridge.cleanup()
