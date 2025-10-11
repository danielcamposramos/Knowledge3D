"""Test sovereign galaxy memory updater with PTX backend."""

import numpy as np
import pytest

from knowledge3d.cranium.ptx_runtime.galaxy_memory_updater import GalaxyMemoryUpdater


def test_blend_basic():
    """Test basic blending operation."""
    updater = GalaxyMemoryUpdater()

    old = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    teacher = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    # blend_factor = 0.3 means 70% old + 30% teacher
    result = updater.blend(old, teacher, blend_factor=0.3)

    expected = old * 0.7 + teacher * 0.3  # [1.9, 2.9, 3.9]

    assert result.shape == old.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_blend_identity():
    """Test blend with factor=0 returns old embedding unchanged."""
    updater = GalaxyMemoryUpdater()

    old = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    teacher = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    result = updater.blend(old, teacher, blend_factor=0.0)

    assert np.allclose(result, old, atol=1e-5)


def test_blend_full_teacher():
    """Test blend with factor=1 returns teacher embedding."""
    updater = GalaxyMemoryUpdater()

    old = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    teacher = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    result = updater.blend(old, teacher, blend_factor=1.0)

    assert np.allclose(result, teacher, atol=1e-5)


def test_blend_large_embedding():
    """Test blending with large embeddings (1024 dimensions)."""
    updater = GalaxyMemoryUpdater()

    old = np.random.rand(1024).astype(np.float32)
    teacher = np.random.rand(1024).astype(np.float32)

    result = updater.blend(old, teacher, blend_factor=0.25)

    expected = old * 0.75 + teacher * 0.25

    assert result.shape == old.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_blend_2d_array():
    """Test blending with 2D arrays (automatically flattened)."""
    updater = GalaxyMemoryUpdater()

    old = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    teacher = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    result = updater.blend(old, teacher, blend_factor=0.5)

    expected = old * 0.5 + teacher * 0.5  # [[3, 4], [5, 6]]

    # Result is flattened
    assert result.shape == (4,)
    assert np.allclose(result, expected.flatten(), atol=1e-5)


def test_blend_sequence_single():
    """Test blend_sequence with single teacher."""
    updater = GalaxyMemoryUpdater()

    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    teachers = [np.array([4.0, 5.0, 6.0], dtype=np.float32)]

    result = updater.blend_sequence(base, teachers, blend_factor=0.3)

    # Should be same as single blend
    expected = base * 0.7 + teachers[0] * 0.3

    assert np.allclose(result, expected, atol=1e-5)


def test_blend_sequence_multiple():
    """Test blend_sequence with multiple teachers."""
    updater = GalaxyMemoryUpdater()

    base = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    teachers = [
        np.array([2.0, 2.0, 2.0], dtype=np.float32),
        np.array([3.0, 3.0, 3.0], dtype=np.float32),
    ]

    result = updater.blend_sequence(base, teachers, blend_factor=0.5)

    # Step 1: [1, 1, 1] * 0.5 + [2, 2, 2] * 0.5 = [1.5, 1.5, 1.5]
    # Step 2: [1.5, 1.5, 1.5] * 0.5 + [3, 3, 3] * 0.5 = [2.25, 2.25, 2.25]
    expected = np.array([2.25, 2.25, 2.25], dtype=np.float32)

    assert np.allclose(result, expected, atol=1e-5)


def test_blend_sequence_empty():
    """Test blend_sequence with empty teacher list returns base."""
    updater = GalaxyMemoryUpdater()

    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    teachers = []

    result = updater.blend_sequence(base, teachers, blend_factor=0.3)

    assert np.allclose(result, base, atol=1e-5)


def test_blend_different_blend_factors():
    """Test blending with various blend factors."""
    updater = GalaxyMemoryUpdater()

    old = np.array([10.0], dtype=np.float32)
    teacher = np.array([20.0], dtype=np.float32)

    # Test various blend factors
    factors = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    for factor in factors:
        result = updater.blend(old, teacher, blend_factor=factor)
        expected = old * (1 - factor) + teacher * factor
        assert np.allclose(result, expected, atol=1e-5), \
            f"Failed for blend_factor={factor}: got {result}, expected {expected}"


def test_exponential_moving_average():
    """Test EMA behavior with sequence of updates."""
    updater = GalaxyMemoryUpdater()

    # Simulate EMA with alpha=0.3 (blend_factor)
    base = np.array([100.0], dtype=np.float32)
    teachers = [np.array([110.0], dtype=np.float32) for _ in range(10)]

    result = updater.blend_sequence(base, teachers, blend_factor=0.3)

    # EMA should converge towards teacher value
    # After many iterations: base + (teacher - base) * (1 - (1-alpha)^n)
    # For alpha=0.3, n=10: converges ~97% to teacher
    assert result[0] > 105.0  # Should be close to 110
    assert result[0] < 110.0  # But not quite there yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
