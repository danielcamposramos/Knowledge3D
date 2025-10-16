import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_swarm_bridge import NineChainSwarmBridge


@pytest.mark.gpu
def test_swarm_basic_execution() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)
    output, chain_states, resonance = bridge.execute_swarm(input_vec, num_iterations=3)

    assert output.shape == (64,)
    assert chain_states.shape == (9, 64)
    assert resonance.shape == (9,)
    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(chain_states))
    assert np.all(np.isfinite(resonance))

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_resonance_behavior() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)
    _, _, resonance = bridge.execute_swarm(input_vec, num_iterations=3)

    assert np.all(resonance > -10.0)
    assert np.all(resonance < 10.0)

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_adaptation_changes_state() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)
    _, states_one, _ = bridge.execute_swarm(input_vec, num_iterations=1)
    _, states_five, _ = bridge.execute_swarm(input_vec, num_iterations=5)

    diff = np.linalg.norm(states_five - states_one)
    assert diff > 0.01

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_synthesis_not_identical_to_any_chain() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)
    output, chain_states, _ = bridge.execute_swarm(input_vec, num_iterations=3)

    for chain_id in range(9):
        diff = np.linalg.norm(output - chain_states[chain_id])
        if chain_id != 8:
            assert diff > 1e-3

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_diagnostics_structure() -> None:
    bridge = NineChainSwarmBridge()

    input_vec = np.random.randn(64).astype(np.float32)
    bridge.execute_swarm(input_vec, num_iterations=3)

    diagnostics = bridge.get_chain_diagnostics()
    for key in (
        "chain_states",
        "resonance_scores",
        "chain_norms",
        "mean_resonance",
        "resonance_variance",
    ):
        assert key in diagnostics

    bridge.cleanup()
