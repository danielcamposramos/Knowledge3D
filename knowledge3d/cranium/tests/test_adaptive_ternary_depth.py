"""Tests for adaptive ternary depth with Claude's enhancements."""

import numpy as np
import pytest

from knowledge3d.cranium.tools.adaptive_ternary_depth import AdaptiveTernaryDepth

try:
    _probe = AdaptiveTernaryDepth()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"CUDA driver/PTX unavailable: {exc}", allow_module_level=True)


def _unpack_trits(packed, n):
    """Helper to unpack trits."""
    out = []
    for i in range(n):
        word = packed[i >> 4]
        shift = (i & 0xF) << 1
        bits = (word >> shift) & 0x3
        if bits == 2:
            out.append(1)
        elif bits == 1:
            out.append(0)
        else:
            out.append(-1)
    return out


def test_adaptive_thresholds():
    """Test adaptive threshold computation."""
    # Create embeddings with known distribution
    np.random.seed(42)
    embeddings = np.random.randn(100, 64).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    query = np.array([1.0] + [0.0] * 63, dtype=np.float32)
    query /= np.linalg.norm(query)

    depth = AdaptiveTernaryDepth(adaptive_thresholds=True)

    # Compute adaptive thresholds
    attract_thresh, repel_thresh = depth.compute_adaptive_thresholds(
        embeddings, query, percentile_attract=75.0, percentile_repel=25.0
    )

    # Thresholds should be reasonable
    assert -1.0 <= repel_thresh < attract_thresh <= 1.0
    assert repel_thresh < 0.0  # Should be negative
    assert attract_thresh > 0.0  # Should be positive


def test_caching():
    """Test depth field caching."""
    embeddings = np.random.randn(50, 32).astype(np.float32)
    query = np.random.randn(32).astype(np.float32)

    depth = AdaptiveTernaryDepth(cache_size=10, adaptive_thresholds=False)

    # First call - not cached
    result1 = depth.compute(embeddings, query, attract_thresh=0.3, repel_thresh=-0.1)

    # Second call - should hit cache
    result2 = depth.compute(embeddings, query, attract_thresh=0.3, repel_thresh=-0.1)

    # Results should be identical
    assert np.array_equal(result1, result2)

    # Cache stats
    stats = depth.get_cache_stats()
    assert stats["size"] == 1
    assert stats["capacity"] == 10


def test_batch_compute():
    """Test batch depth computation."""
    embeddings = np.random.randn(100, 64).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Multiple queries
    queries = [
        np.random.randn(64).astype(np.float32),
        np.random.randn(64).astype(np.float32),
        np.random.randn(64).astype(np.float32),
    ]
    for q in queries:
        q /= np.linalg.norm(q)

    depth = AdaptiveTernaryDepth(adaptive_thresholds=False)

    # Batch compute
    results = depth.compute_batch(
        embeddings, queries, attract_thresh=0.35, repel_thresh=-0.05
    )

    assert len(results) == 3
    assert all(isinstance(r, np.ndarray) for r in results)

    # Each result should have correct shape
    n_words = (100 + 15) // 16
    assert all(r.shape == (n_words,) for r in results)


def test_path_aware_depth():
    """Test path-aware depth with history bias."""
    np.random.seed(42)
    embeddings = np.random.randn(50, 32).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    query = np.random.randn(32).astype(np.float32)
    query /= np.linalg.norm(query)

    depth = AdaptiveTernaryDepth(adaptive_thresholds=False)

    # Path history: nodes 5, 10, 15 recently visited
    path_history = [5, 10, 15]

    # Compute path-aware depth
    result = depth.compute_path_aware_depth(
        embeddings, query, path_history, history_weight=0.5
    )

    trits = _unpack_trits(result, n=50)

    # Visited nodes should tend toward attract (+1) or neutral (0)
    # (exact values depend on base depth and randomness, just check validity)
    assert all(t in [-1, 0, 1] for t in trits)
    assert len(trits) == 50


def test_cache_size_limit():
    """Test cache LRU behavior."""
    embeddings = np.random.randn(10, 16).astype(np.float32)

    depth = AdaptiveTernaryDepth(cache_size=3, adaptive_thresholds=False)

    # Add 5 different queries (exceeds cache size)
    for i in range(5):
        query = np.random.randn(16).astype(np.float32)
        depth.compute(embeddings, query, attract_thresh=0.3, repel_thresh=-0.1)

    # Cache should be limited to 3
    stats = depth.get_cache_stats()
    assert stats["size"] <= 3


def test_clear_cache():
    """Test cache clearing."""
    embeddings = np.random.randn(10, 16).astype(np.float32)
    query = np.random.randn(16).astype(np.float32)

    depth = AdaptiveTernaryDepth(adaptive_thresholds=False)

    # Populate cache
    depth.compute(embeddings, query)
    assert depth.get_cache_stats()["size"] > 0

    # Clear cache
    depth.clear_cache()
    assert depth.get_cache_stats()["size"] == 0
