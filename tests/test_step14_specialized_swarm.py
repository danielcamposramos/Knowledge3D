import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_specialized_bridge import (
    NineChainSpecializedBridge,
    SwarmDiagnostics,
)


CHAIN_DIM = NineChainSpecializedBridge.CHAIN_DIM


@pytest.mark.gpu
def test_specialized_swarm_produces_expected_shapes() -> None:
    bridge = NineChainSpecializedBridge()

    input_vec = np.random.randn(CHAIN_DIM).astype(np.float32)
    output, chain_states, weights = bridge.execute_swarm(input_vec, num_iterations=2)
    output_arr = np.asarray(output, dtype=np.float32)
    chain_states_arr = np.asarray(chain_states, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32)

    assert output_arr.shape == (CHAIN_DIM,)
    assert chain_states_arr.shape == (bridge.NUM_CHAINS, CHAIN_DIM)
    assert weights_arr.shape == (bridge.NUM_ACTIVE_CHAINS,)

    assert np.isfinite(output_arr).all()
    assert np.isfinite(chain_states_arr).all()
    assert np.isfinite(weights_arr).all()
    assert np.isclose(weights_arr.sum(), 1.0, atol=1e-4)

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_output_mode_skips_readback() -> None:
    bridge = NineChainSpecializedBridge()

    vec = np.random.randn(CHAIN_DIM).astype(np.float32)
    output, chain_states, weights = bridge.execute_swarm(
        vec,
        num_iterations=2,
        readback_mode="output",
    )

    assert np.asarray(output, dtype=np.float32).shape == (CHAIN_DIM,)
    assert chain_states is None
    assert weights is None

    diagnostics = bridge.get_chain_diagnostics()
    assert isinstance(diagnostics, SwarmDiagnostics)
    assert np.isclose(np.asarray(diagnostics.resonance_weights, dtype=np.float32).sum(), 1.0, atol=1e-4)

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_persistent_state_effect() -> None:
    bridge = NineChainSpecializedBridge()

    input_vec = np.random.randn(CHAIN_DIM).astype(np.float32)
    first, _, _ = bridge.execute_swarm(input_vec, num_iterations=1)
    second, _, _ = bridge.execute_swarm(input_vec, num_iterations=1)

    assert np.linalg.norm(np.asarray(second, dtype=np.float32) - np.asarray(first, dtype=np.float32)) > 1e-3

    bridge.reset_states()
    reset, _, _ = bridge.execute_swarm(input_vec, num_iterations=1)
    assert np.linalg.norm(np.asarray(reset, dtype=np.float32) - np.asarray(second, dtype=np.float32)) > 1e-3

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_diagnostics_snapshot() -> None:
    bridge = NineChainSpecializedBridge()

    input_vec = np.ones(CHAIN_DIM, dtype=np.float32)
    bridge.execute_swarm(input_vec, num_iterations=3)

    diagnostics = bridge.get_chain_diagnostics()
    assert isinstance(diagnostics, SwarmDiagnostics)

    assert diagnostics.resonance_matrix.shape == (
        bridge.NUM_ACTIVE_CHAINS,
        bridge.NUM_ACTIVE_CHAINS,
    )
    assert np.asarray(diagnostics.resonance_weights, dtype=np.float32).shape == (bridge.NUM_ACTIVE_CHAINS,)
    assert diagnostics.chain_states.shape == (bridge.NUM_CHAINS, CHAIN_DIM)
    assert np.asarray(diagnostics.chain_norms, dtype=np.float32).shape == (bridge.NUM_CHAINS,)

    assert np.all(np.asarray(diagnostics.resonance_weights, dtype=np.float32) >= 0.0)
    assert np.isclose(np.asarray(diagnostics.resonance_weights, dtype=np.float32).sum(), 1.0, atol=1e-4)

    bridge.cleanup()


@pytest.mark.gpu
def test_specialized_swarm_iterations_increase_activity() -> None:
    bridge = NineChainSpecializedBridge()

    input_vec = np.random.default_rng(42).standard_normal(CHAIN_DIM).astype(np.float32)
    bridge.reset_states()
    _, chain_states_one, weights_one = bridge.execute_swarm(input_vec, num_iterations=1)
    bridge.reset_states()
    _, chain_states_three, weights_three = bridge.execute_swarm(input_vec, num_iterations=3)

    activity_delta = np.linalg.norm(
        np.asarray(chain_states_three, dtype=np.float32) - np.asarray(chain_states_one, dtype=np.float32)
    )
    weight_delta = np.linalg.norm(
        np.asarray(weights_three, dtype=np.float32) - np.asarray(weights_one, dtype=np.float32)
    )

    assert activity_delta > 1e-2
    assert weight_delta >= 0.0

    bridge.cleanup()
