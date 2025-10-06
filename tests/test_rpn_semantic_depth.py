"""
Tests for RPN-powered Semantic Depth Calculation

Validates GLM's Semantic Depth Allocation using modular RPN kernel.
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
def test_rpn_executor_loads():
    """Test that RPN executor initializes successfully."""
    _require_gpu()

    from knowledge3d.cranium.rpn_executor import get_rpn_executor

    executor = get_rpn_executor()
    assert executor is not None
    assert executor.MAX_INSTANCES == 15
    assert executor.STACK_DEPTH == 64


@pytest.mark.cuda
def test_semantic_depth_single_cluster():
    """Test semantic depth calculation for a single cluster."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depth_rpn

    # Create test cluster embeddings (10 nodes, 512-dim)
    np.random.seed(42)
    cluster_embs = np.random.randn(10, 512).astype(np.float32)
    cluster_embs /= np.linalg.norm(cluster_embs, axis=1, keepdims=True)  # Normalize

    # Compute semantic depth
    depth = compute_semantic_depth_rpn(
        cluster_embeddings=cluster_embs,
        cluster_size=10,
        min_depth=2,
        max_depth=12
    )

    # Validate output
    assert isinstance(depth, int)
    assert 2 <= depth <= 12, f"Depth {depth} out of range [2, 12]"


@pytest.mark.cuda
def test_semantic_depth_batch():
    """Test batch semantic depth calculation."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depths_batch_rpn

    # Create test clusters (5 clusters, varying sizes)
    np.random.seed(42)
    clusters = [
        np.random.randn(5, 512).astype(np.float32),
        np.random.randn(10, 512).astype(np.float32),
        np.random.randn(15, 512).astype(np.float32),
        np.random.randn(20, 512).astype(np.float32),
        np.random.randn(8, 512).astype(np.float32),
    ]

    # Normalize embeddings
    for cluster_embs in clusters:
        cluster_embs /= np.linalg.norm(cluster_embs, axis=1, keepdims=True)

    # Compute depths in batch
    depths = compute_semantic_depths_batch_rpn(
        clusters=clusters,
        min_depth=2,
        max_depth=12
    )

    # Validate output
    assert len(depths) == 5
    assert all(2 <= d <= 12 for d in depths), f"Depths {depths} out of range"

    # Larger clusters should tend to have higher depths (due to log term)
    # But entropy also matters, so this is a soft check
    assert depths[3] >= depths[0], "Larger cluster should have >= depth"


@pytest.mark.cuda
def test_entropy_estimation():
    """Test information entropy estimation."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import estimate_information_entropy

    # Uniform distribution (high entropy)
    np.random.seed(42)
    uniform_embs = np.random.randn(20, 512).astype(np.float32)
    uniform_embs /= np.linalg.norm(uniform_embs, axis=1, keepdims=True)

    # Concentrated distribution (low entropy)
    concentrated_embs = np.tile(np.random.randn(1, 512), (20, 1)).astype(np.float32)
    concentrated_embs += np.random.randn(20, 512) * 0.01  # Small noise
    concentrated_embs /= np.linalg.norm(concentrated_embs, axis=1, keepdims=True)

    # Compute entropies
    entropy_uniform = estimate_information_entropy(uniform_embs)
    entropy_concentrated = estimate_information_entropy(concentrated_embs)

    # Validate
    assert entropy_uniform > 0, "Uniform entropy should be positive"
    assert entropy_concentrated >= 0, "Entropy should be non-negative"

    # Uniform should have higher entropy than concentrated
    assert entropy_uniform > entropy_concentrated, \
        f"Uniform entropy ({entropy_uniform}) should exceed concentrated ({entropy_concentrated})"


@pytest.mark.cuda
def test_depth_scales_with_cluster_size():
    """Test that depth increases with cluster size (due to log term)."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depth_rpn

    # Fixed embeddings, varying cluster size
    np.random.seed(42)
    base_embs = np.random.randn(100, 512).astype(np.float32)
    base_embs /= np.linalg.norm(base_embs, axis=1, keepdims=True)

    # Test with different sizes
    sizes = [5, 10, 20, 40, 80]
    depths = []

    for size in sizes:
        cluster_embs = base_embs[:size]
        depth = compute_semantic_depth_rpn(
            cluster_embeddings=cluster_embs,
            cluster_size=size
        )
        depths.append(depth)

    # Depths should generally increase with size
    # (log term dominates for similar entropy)
    assert depths[-1] >= depths[0], \
        f"Depth should increase with size: {sizes} → {depths}"


@pytest.mark.cuda
def test_depth_clamping():
    """Test that depth respects min/max bounds."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depth_rpn

    # Very small cluster (should clamp to min)
    small_cluster = np.random.randn(2, 512).astype(np.float32)
    small_cluster /= np.linalg.norm(small_cluster, axis=1, keepdims=True)

    depth_small = compute_semantic_depth_rpn(
        cluster_embeddings=small_cluster,
        cluster_size=2,
        min_depth=3,
        max_depth=10
    )

    assert depth_small >= 3, f"Should respect min_depth: got {depth_small}"

    # Very large cluster with high entropy (should clamp to max)
    large_cluster = np.random.randn(200, 512).astype(np.float32)
    large_cluster /= np.linalg.norm(large_cluster, axis=1, keepdims=True)

    depth_large = compute_semantic_depth_rpn(
        cluster_embeddings=large_cluster,
        cluster_size=200,
        min_depth=2,
        max_depth=8
    )

    assert depth_large <= 8, f"Should respect max_depth: got {depth_large}"


@pytest.mark.cuda
def test_rpn_executor_batch_performance():
    """Test RPN batch execution performance."""
    _require_gpu()

    from knowledge3d.cranium.semantic_depth_rpn import compute_semantic_depths_batch_rpn
    import time

    # Create many clusters for performance test
    np.random.seed(42)
    clusters = [
        np.random.randn(10, 512).astype(np.float32)
        for _ in range(30)
    ]

    # Normalize
    for cluster_embs in clusters:
        cluster_embs /= np.linalg.norm(cluster_embs, axis=1, keepdims=True)

    # Time batch execution
    start = time.time()
    depths = compute_semantic_depths_batch_rpn(clusters)
    elapsed = time.time() - start

    # Should complete quickly (target: <0.1s for 30 clusters)
    assert elapsed < 0.5, f"Batch execution too slow: {elapsed:.3f}s for 30 clusters"
    assert len(depths) == 30
    assert all(isinstance(d, (int, np.integer)) for d in depths)

    print(f"\n✓ RPN batch performance: {len(clusters)} clusters in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed*1000/len(clusters):.2f}ms per cluster")
