import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc


@pytest.mark.gpu
def test_temporal_pipeline_end_to_end() -> None:
    bridge = ThinkingTagRPNBridge()

    input_vec = np.random.randn(512).astype(np.float32)
    context = np.random.randn(64, 256).astype(np.float32)
    weights = {
        "W1": np.random.randn(256, 512).astype(np.float32),
        "W2": np.random.randn(256, 256).astype(np.float32),
        "W3": np.random.randn(100, 256).astype(np.float32),
    }

    mask, coherence, activity = bridge.compute_temporal_mask(context, threshold=0.5)
    output, entropy = bridge.execute_temporal(input_vec, weights, mask)

    assert mask.shape == coherence.shape == activity.shape == (256,)
    assert output.shape == (100,)
    assert np.isfinite(entropy)
    assert np.all(np.isfinite(output))

    bridge.cleanup()


@pytest.mark.gpu
def test_step14_foundation_surface() -> None:
    bridge = ThinkingTagRPNBridge()

    chain_states = np.random.randn(9, 64).astype(np.float32)
    fusion_matrix = np.random.randn(64, 64).astype(np.float32)
    fused = bridge._test_matmul_small(chain_states, fusion_matrix)
    assert fused.shape == (9, 64)

    program = [
        5.0,
        0.0,
        ropc.OP_STORE,
        0.0,
        ropc.OP_RECALL,
    ]
    stack = bridge._execute_rpn_program(program)
    assert np.isclose(stack[-1], 5.0)

    bridge.cleanup()
