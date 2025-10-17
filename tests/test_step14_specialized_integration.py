import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_specialized_bridge import NineChainSpecializedBridge
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_thinking_tag_bridge_with_specialized_swarm() -> None:
    bridge = ThinkingTagRPNBridge(
        use_specialized_swarm=True,
        swarm_iterations=2,
    )

    input_vec = np.random.randn(NineChainSpecializedBridge.CHAIN_DIM).astype(np.float32)
    weights = {
        "W1": np.random.randn(256, NineChainSpecializedBridge.CHAIN_DIM).astype(np.float32),
        "W2": np.random.randn(192, 256).astype(np.float32),
        "W3": np.random.randn(NineChainSpecializedBridge.CHAIN_DIM, 192).astype(np.float32),
    }

    fused, entropy = bridge.execute_temporal(input_vec, weights)

    assert fused.shape == (NineChainSpecializedBridge.CHAIN_DIM,)
    assert np.isfinite(fused).all()
    assert np.isfinite(entropy)

    diagnostics = bridge.get_swarm_diagnostics()
    assert diagnostics is not None
    assert diagnostics.resonance_weights.shape == (NineChainSpecializedBridge.NUM_ACTIVE_CHAINS,)

    bridge.cleanup()


@pytest.mark.gpu
def test_swarm_bridge_direct_execution_consistency() -> None:
    swarm = NineChainSpecializedBridge()

    rng = np.random.default_rng(7)
    vec = rng.standard_normal(NineChainSpecializedBridge.CHAIN_DIM).astype(np.float32)
    weights = []
    for _ in range(3):
        _, _, resonance = swarm.execute_swarm(vec, num_iterations=2, reset_state=True)
        weights.append(resonance)

    weights = np.asarray(weights)
    assert weights.shape == (3, NineChainSpecializedBridge.NUM_ACTIVE_CHAINS)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)

    swarm.cleanup()
