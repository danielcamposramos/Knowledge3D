import time

import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_specialized_bridge import NineChainSpecializedBridge


CHAIN_DIM = NineChainSpecializedBridge.CHAIN_DIM


@pytest.mark.gpu
def test_specialized_swarm_latency_budget() -> None:
    bridge = NineChainSpecializedBridge()
    input_vec = np.random.randn(CHAIN_DIM).astype(np.float32)

    bridge.execute_swarm(input_vec, num_iterations=2)
    for _ in range(50):
        bridge.execute_swarm(input_vec, num_iterations=2, readback_mode="output")

    runs = 200
    start = time.perf_counter()
    for _ in range(runs):
        bridge.execute_swarm(input_vec, num_iterations=2, readback_mode="output")
    elapsed_us = (time.perf_counter() - start) * 1e6 / runs

    print(f"\nSpecialised 9-chain latency: {elapsed_us:.2f} µs")
    assert elapsed_us < 95.0

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_iteration_scaling() -> None:
    bridge = NineChainSpecializedBridge()
    input_vec = np.random.randn(CHAIN_DIM).astype(np.float32)

    timings = {}
    for iters in (1, 2, 4):
        bridge.reset_states()
        for _ in range(25):
            bridge.execute_swarm(input_vec, num_iterations=iters, readback_mode="output")
        runs = 150
        start = time.perf_counter()
        for _ in range(runs):
            bridge.execute_swarm(input_vec, num_iterations=iters, readback_mode="output")
        timings[iters] = (time.perf_counter() - start) * 1e6 / runs
        print(f"{iters} iteration(s) -> {timings[iters]:.2f} µs")

    assert timings[2] > timings[1]
    assert timings[4] > timings[2]

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_resonance_weight_update_cost() -> None:
    bridge = NineChainSpecializedBridge()
    input_vec = np.random.randn(CHAIN_DIM).astype(np.float32)

    bridge.reset_states()
    _, _, weights = bridge.execute_swarm(input_vec, num_iterations=2)
    assert np.isclose(weights.sum(), 1.0, atol=1e-4)

    start = time.perf_counter()
    for _ in range(100):
        bridge.reset_states()
        bridge.execute_swarm(input_vec, num_iterations=2)
    total_time = (time.perf_counter() - start) * 1e6 / 100

    per_chain = total_time / NineChainSpecializedBridge.NUM_ACTIVE_CHAINS
    print(f"\nSwarm total latency: {total_time:.2f} µs -> per active chain: {per_chain:.2f} µs")

    bridge.cleanup()
