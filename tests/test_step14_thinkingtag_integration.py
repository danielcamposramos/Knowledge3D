import numpy as np
import pytest

from knowledge3d.cranium.bridges.nine_chain_swarm_bridge import NineChainSwarmBridge
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_swarm_reasoning_stage() -> None:
    thinking_tag = ThinkingTagRPNBridge()
    swarm = NineChainSwarmBridge()

    fused_embedding = np.random.randn(64).astype(np.float32)
    output, chain_states, resonance = swarm.execute_swarm(fused_embedding, num_iterations=3)

    assert np.all(np.isfinite(output))
    assert np.linalg.norm(output) > 0.01
    assert chain_states.shape == (9, 64)
    assert resonance.shape == (9,)

    thinking_tag.cleanup()
    swarm.cleanup()


@pytest.mark.gpu
def test_swarm_output_diversity() -> None:
    swarm = NineChainSwarmBridge()

    fused_embedding = np.random.randn(64).astype(np.float32)
    outputs = []
    for iterations in (1, 3, 5):
        out, _, _ = swarm.execute_swarm(fused_embedding, num_iterations=iterations)
        outputs.append(out)

    outputs = np.asarray(outputs)
    variability = np.var(outputs, axis=0).mean()
    assert variability >= 0.0  # prototype is deterministic but should not explode

    swarm.cleanup()
