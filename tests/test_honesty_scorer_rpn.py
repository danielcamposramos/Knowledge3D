"""
Tests for RPN-Powered Honesty Scoring

Validates RLWHF honesty scoring using modular RPN kernel.
"""

import numpy as np
import pytest


def _require_gpu():
    """Skip test if GPU not available."""
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_honesty_scorer_loads():
    """Test that honesty scorer module loads successfully."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn

    # Should not raise
    assert compute_honesty_score_rpn is not None


@pytest.mark.cuda
def test_single_honesty_score():
    """Test single honesty score calculation."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn

    # Perfect scores
    score = compute_honesty_score_rpn(
        correctness=1.0,
        reasoning=1.0,
        uncertainty=1.0,
        alignment=1.0
    )

    # Should be 1.0 (0.4 + 0.2 + 0.2 + 0.2 = 1.0)
    assert 0.99 <= score <= 1.01, f"Expected ~1.0, got {score}"


@pytest.mark.cuda
def test_partial_honesty_score():
    """Test honesty score with partial components."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn

    # High correctness, low others
    score = compute_honesty_score_rpn(
        correctness=1.0,  # 0.4 × 1.0 = 0.4
        reasoning=0.5,    # 0.2 × 0.5 = 0.1
        uncertainty=0.5,  # 0.2 × 0.5 = 0.1
        alignment=0.5     # 0.2 × 0.5 = 0.1
    )

    # Should be ~0.7 (0.4 + 0.1 + 0.1 + 0.1)
    assert 0.68 <= score <= 0.72, f"Expected ~0.7, got {score}"


@pytest.mark.cuda
def test_zero_honesty_score():
    """Test honesty score with all zeros."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn

    score = compute_honesty_score_rpn(
        correctness=0.0,
        reasoning=0.0,
        uncertainty=0.0,
        alignment=0.0
    )

    assert score == 0.0, f"Expected 0.0, got {score}"


@pytest.mark.cuda
def test_honesty_batch_scoring():
    """Test batch honesty scoring."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_batch_rpn

    # Create batch of component sets
    components = [
        {'correctness': 1.0, 'reasoning': 1.0, 'uncertainty': 1.0, 'alignment': 1.0},
        {'correctness': 0.8, 'reasoning': 0.7, 'uncertainty': 0.6, 'alignment': 0.9},
        {'correctness': 0.5, 'reasoning': 0.5, 'uncertainty': 0.5, 'alignment': 0.5},
        {'correctness': 0.0, 'reasoning': 0.0, 'uncertainty': 0.0, 'alignment': 0.0},
    ]

    scores = compute_honesty_batch_rpn(components)

    # Validate
    assert len(scores) == 4
    assert 0.99 <= scores[0] <= 1.01, "First should be ~1.0"
    assert scores[1] < scores[0], "Second should be lower than first"
    assert 0.49 <= scores[2] <= 0.51, "Third should be ~0.5"
    assert scores[3] == 0.0, "Fourth should be 0.0"


@pytest.mark.cuda
def test_honesty_weighted_custom():
    """Test custom-weighted honesty scoring."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_weighted_rpn

    # Custom weights favoring correctness
    score = compute_honesty_weighted_rpn(
        correctness=1.0,
        reasoning=0.0,
        uncertainty=0.0,
        alignment=0.0,
        weights={
            'correctness': 1.0,  # Only correctness matters
            'reasoning': 0.0,
            'uncertainty': 0.0,
            'alignment': 0.0
        }
    )

    assert 0.99 <= score <= 1.01, f"Expected ~1.0, got {score}"


@pytest.mark.cuda
def test_honesty_clamping():
    """Test that honesty scores are clamped to [0, 1]."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn

    # Try extreme values (should clamp)
    score = compute_honesty_score_rpn(
        correctness=2.0,  # Out of range
        reasoning=2.0,
        uncertainty=2.0,
        alignment=2.0
    )

    # Should clamp to 1.0
    assert score <= 1.0, f"Should clamp to ≤1.0, got {score}"


@pytest.mark.cuda
def test_honesty_batch_performance():
    """Test batch honesty scoring performance."""
    _require_gpu()

    from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_batch_rpn
    import time

    # Create many component sets
    np.random.seed(42)
    components = [
        {
            'correctness': np.random.rand(),
            'reasoning': np.random.rand(),
            'uncertainty': np.random.rand(),
            'alignment': np.random.rand()
        }
        for _ in range(30)
    ]

    # Time batch execution
    start = time.time()
    scores = compute_honesty_batch_rpn(components)
    elapsed = time.time() - start

    # Should complete quickly (target: <0.1s for 30 scores)
    assert elapsed < 0.5, f"Batch too slow: {elapsed:.3f}s for 30 scores"
    assert len(scores) == 30
    assert all(0.0 <= s <= 1.0 for s in scores), "All scores should be in [0, 1]"

    print(f"\n✓ RPN honesty batch: {len(components)} scores in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed*1000/len(components):.2f}ms per score")
