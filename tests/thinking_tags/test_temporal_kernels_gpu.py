import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_temporal_mask_shapes_and_bounds() -> None:
    bridge = ThinkingTagRPNBridge()
    try:
        context = np.random.randn(48, 256).astype(np.float32)
        mask, coherence, activity = bridge.compute_temporal_mask(context)

        assert mask.shape == (256,)
        assert coherence.shape == (256,)
        assert activity.shape == (256,)
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)
        assert np.all(coherence >= 0.0)
    finally:
        bridge.cleanup()


@pytest.mark.gpu
def test_temporal_mask_matches_cpu_reference() -> None:
    bridge = ThinkingTagRPNBridge()
    try:
        context = np.random.randn(32, 128).astype(np.float32)
        threshold = float(np.mean(np.abs(context)))

        mask_gpu, coherence_gpu, activity_gpu = bridge.compute_temporal_mask(context, threshold=threshold)

        mean = np.mean(context, axis=0)
        variance = np.mean(context * context, axis=0) - mean * mean
        variance = np.clip(variance, 0.0, None)
        coherence_cpu = 1.0 / (1.0 + np.sqrt(variance + 1e-8))
        activity_cpu = np.mean(np.abs(context), axis=0)
        mask_cpu = 1.0 / (1.0 + np.exp(-(coherence_cpu - threshold) * 4.0))

        assert np.allclose(coherence_gpu, coherence_cpu, rtol=1e-5, atol=1e-5)
        assert np.allclose(activity_gpu, activity_cpu, rtol=1e-5, atol=1e-5)
        assert np.allclose(mask_gpu, mask_cpu, rtol=1e-5, atol=1e-5)
    finally:
        bridge.cleanup()
