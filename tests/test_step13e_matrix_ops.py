import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_op_matmul_small_matches_numpy() -> None:
    bridge = ThinkingTagRPNBridge()

    a = np.random.randn(9, 64).astype(np.float32)
    b = np.random.randn(64, 64).astype(np.float32)

    result_gpu = bridge._test_matmul_small(a, b)
    result_ref = a @ b

    np.testing.assert_allclose(result_gpu, result_ref, rtol=1e-5, atol=1e-5)
    bridge.cleanup()


@pytest.mark.gpu
def test_op_dot_batch_matches_reference() -> None:
    bridge = ThinkingTagRPNBridge()

    vectors = np.random.randn(9, 64).astype(np.float32)
    query = np.random.randn(64).astype(np.float32)

    result_gpu = bridge._test_dot_batch(query, vectors)
    result_ref = vectors @ query

    np.testing.assert_allclose(result_gpu, result_ref, rtol=1e-5, atol=1e-5)
    bridge.cleanup()
