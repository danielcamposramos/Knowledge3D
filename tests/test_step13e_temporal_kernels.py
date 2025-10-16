import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge


@pytest.mark.gpu
def test_temporal_coherence_shapes_and_ranges() -> None:
    bridge = ThinkingTagRPNBridge()

    time_steps, feature_dim = 64, 512
    context = np.random.randn(time_steps, feature_dim).astype(np.float32)

    mask, coherence, activity = bridge.compute_temporal_mask(context, threshold=0.5)

    assert mask.shape == (feature_dim,)
    assert coherence.shape == (feature_dim,)
    assert activity.shape == (feature_dim,)

    assert np.all(coherence >= 0.0)
    assert np.all((mask >= 0.0) & (mask <= 1.0))
    assert np.all(activity >= 0.0)

    bridge.cleanup()


@pytest.mark.gpu
def test_temporal_mask_threshold_influence() -> None:
    bridge = ThinkingTagRPNBridge()

    context = np.random.randn(48, 256).astype(np.float32)

    mask_low, _, _ = bridge.compute_temporal_mask(context, threshold=0.1)
    mask_high, _, _ = bridge.compute_temporal_mask(context, threshold=0.9)

    assert np.mean(mask_low) > np.mean(mask_high)

    bridge.cleanup()


@pytest.mark.gpu
def test_temporal_activity_matches_numpy_reference() -> None:
    bridge = ThinkingTagRPNBridge()

    context = np.random.randn(32, 128).astype(np.float32)
    _, _, activity = bridge.compute_temporal_mask(context, threshold=0.5)

    activity_ref = np.mean(np.abs(context), axis=0)
    np.testing.assert_allclose(activity, activity_ref, rtol=1e-5)

    bridge.cleanup()
