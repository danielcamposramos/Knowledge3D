"""
Tests for RPN-Powered Clustering and Similarity Calculations

Validates cosine similarity and clustering operations using modular RPN kernel.
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
def test_clustering_module_loads():
    """Test that clustering module loads successfully."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn

    # Should not raise
    assert compute_cosine_similarity_rpn is not None


@pytest.mark.cuda
def test_cosine_similarity_identical():
    """Test cosine similarity of identical vectors."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn

    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    vec /= np.linalg.norm(vec)  # Normalize

    similarity = compute_cosine_similarity_rpn(vec, vec)

    # Identical vectors should have similarity = 1.0
    assert 0.99 <= similarity <= 1.01, \
        f"Expected ~1.0 for identical vectors, got {similarity}"


@pytest.mark.cuda
def test_cosine_similarity_orthogonal():
    """Test cosine similarity of orthogonal vectors."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn

    vec_u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec_v = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    similarity = compute_cosine_similarity_rpn(vec_u, vec_v)

    # Orthogonal vectors should have similarity ≈ 0.0
    assert np.abs(similarity) < 0.01, \
        f"Expected ~0.0 for orthogonal vectors, got {similarity}"


@pytest.mark.cuda
def test_cosine_similarity_opposite():
    """Test cosine similarity of opposite vectors."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_cosine_similarity_rpn

    vec_u = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vec_v = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

    vec_u /= np.linalg.norm(vec_u)
    vec_v /= np.linalg.norm(vec_v)

    similarity = compute_cosine_similarity_rpn(vec_u, vec_v)

    # Opposite vectors should have similarity ≈ -1.0
    assert -1.01 <= similarity <= -0.99, \
        f"Expected ~-1.0 for opposite vectors, got {similarity}"


@pytest.mark.cuda
def test_pairwise_similarities():
    """Test pairwise similarity matrix computation."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_pairwise_similarities_rpn

    # Create test embeddings (5 vectors, 4-dim)
    np.random.seed(42)
    embeddings = np.random.randn(5, 4).astype(np.float32)

    # Normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise similarities
    similarities = compute_pairwise_similarities_rpn(embeddings)

    # Validate
    assert similarities.shape == (5, 5), "Should be 5×5 matrix"

    # Diagonal should be 1.0 (self-similarity)
    np.testing.assert_allclose(np.diag(similarities), 1.0, atol=0.01)

    # Should be symmetric
    np.testing.assert_allclose(similarities, similarities.T, atol=0.01)

    # All values in [-1, 1]
    assert np.all(similarities >= -1.01) and np.all(similarities <= 1.01), \
        "All similarities should be in [-1, 1]"


@pytest.mark.cuda
def test_similarity_matrix_rectangular_matches_numpy_cosine():
    """Test rectangular source-target similarity matrix against NumPy baseline."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_similarity_matrix_rpn

    rng = np.random.default_rng(42)
    sources = rng.normal(size=(3, 8)).astype(np.float32)
    targets = rng.normal(size=(2, 8)).astype(np.float32)

    gpu_sims = compute_similarity_matrix_rpn(sources, targets)

    src_norm = sources / (np.linalg.norm(sources, axis=1, keepdims=True) + 1e-8)
    tgt_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    expected = src_norm @ tgt_norm.T

    assert gpu_sims.shape == (3, 2)
    np.testing.assert_allclose(gpu_sims, expected.astype(np.float32), atol=1e-4)


@pytest.mark.cuda
def test_nearest_neighbors():
    """Test k-nearest neighbor search."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_nearest_neighbors_rpn

    # Create embeddings with one very similar to query
    np.random.seed(42)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    embeddings = np.random.randn(10, 4).astype(np.float32)
    embeddings[5] = np.array([0.99, 0.01, 0.0, 0.0])  # Very similar to query

    # Normalize
    query /= np.linalg.norm(query)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Find 3 nearest neighbors
    indices, similarities = compute_nearest_neighbors_rpn(query, embeddings, k=3)

    # Validate
    assert len(indices) == 3
    assert len(similarities) == 3

    # Index 5 should be the most similar
    assert indices[0] == 5, f"Expected index 5 first, got {indices[0]}"

    # Similarities should be in descending order
    assert similarities[0] >= similarities[1] >= similarities[2], \
        f"Similarities not sorted: {similarities}"


@pytest.mark.cuda
def test_clustering_by_similarity():
    """Test simple similarity-based clustering."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import cluster_by_similarity_rpn

    # Create embeddings with 2 clear clusters
    np.random.seed(42)

    # Cluster 1: around [1, 0, 0, 0]
    cluster1 = np.random.randn(5, 4).astype(np.float32) * 0.1
    cluster1[:, 0] += 1.0

    # Cluster 2: around [0, 1, 0, 0]
    cluster2 = np.random.randn(5, 4).astype(np.float32) * 0.1
    cluster2[:, 1] += 1.0

    embeddings = np.vstack([cluster1, cluster2])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Cluster with threshold
    clusters = cluster_by_similarity_rpn(embeddings, threshold=0.7, min_cluster_size=2)

    # Should find at least 2 clusters
    assert len(clusters) >= 2, f"Expected ≥2 clusters, got {len(clusters)}"

    # Each cluster should have reasonable size
    for cluster in clusters:
        assert len(cluster) >= 2, f"Cluster too small: {len(cluster)}"


@pytest.mark.cuda
def test_cluster_centroid():
    """Test cluster centroid computation."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_cluster_centroid_rpn

    # Create cluster of similar vectors
    np.random.seed(42)
    embeddings = np.random.randn(10, 4).astype(np.float32)

    # Cluster is indices [2, 3, 4]
    cluster_indices = [2, 3, 4]

    centroid = compute_cluster_centroid_rpn(embeddings, cluster_indices)

    # Validate
    assert centroid.shape == (4,), "Centroid should be 4-dim"

    # Should be normalized
    norm = np.linalg.norm(centroid)
    assert np.abs(norm - 1.0) < 0.01, f"Centroid should be normalized, got norm {norm}"

    # Centroid should be mean of cluster embeddings
    expected = np.mean(embeddings[cluster_indices], axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(centroid, expected, atol=0.01)


@pytest.mark.cuda
def test_cluster_refinement():
    """Test iterative cluster refinement."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import refine_clusters_rpn

    # Create embeddings with 2 clear clusters
    np.random.seed(42)

    # Cluster 1: around [1, 0, 0, 0]
    cluster1 = np.random.randn(4, 4).astype(np.float32) * 0.1
    cluster1[:, 0] += 1.0

    # Cluster 2: around [0, 1, 0, 0]
    cluster2 = np.random.randn(4, 4).astype(np.float32) * 0.1
    cluster2[:, 1] += 1.0

    embeddings = np.vstack([cluster1, cluster2])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Initial rough clustering (intentionally bad)
    initial_clusters = [[0, 1, 4, 5], [2, 3, 6, 7]]

    # Refine
    refined_clusters = refine_clusters_rpn(embeddings, initial_clusters, max_iterations=5)

    # Should improve clustering
    assert len(refined_clusters) == 2, "Should maintain 2 clusters"

    # Each cluster should have 4 members
    assert all(len(c) == 4 for c in refined_clusters), \
        "Each cluster should have 4 members"


@pytest.mark.cuda
def test_pairwise_performance():
    """Test pairwise similarity performance."""
    _require_gpu()

    from knowledge3d.cranium.clustering_rpn import compute_pairwise_similarities_rpn
    import time

    # Create medium-sized embedding set
    np.random.seed(42)
    embeddings = np.random.randn(30, 4).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Time computation
    start = time.time()
    similarities = compute_pairwise_similarities_rpn(embeddings)
    elapsed = time.time() - start

    # Should complete quickly (target: <0.5s for 30×30)
    assert elapsed < 1.0, f"Pairwise too slow: {elapsed:.3f}s for 30×30"
    assert similarities.shape == (30, 30)

    print(f"\n✓ RPN pairwise similarities: 30×30 matrix in {elapsed*1000:.1f}ms")
    print(f"  {30*29/2} pairs computed")
