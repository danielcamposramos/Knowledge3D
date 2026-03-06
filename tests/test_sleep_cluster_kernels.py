from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.ptx_runtime.sleep_cluster_kernels import SleepClusterKernels
from knowledge3d.cranium.ptx_runtime.sleep_glyph_kernels import SleepGlyphKernels


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return (arr / norms).astype(np.float32)


@pytest.mark.cuda
def test_sleep_cluster_refiner_moves_vectors_toward_centroids():
    _require_gpu()
    kernels = SleepClusterKernels()
    embeddings = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    centroids = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    assignments = np.array([0, 0, 1, 1], dtype=np.int32)

    before = float(np.dot(embeddings[1], centroids[0]))
    updated = kernels.refine_embeddings(embeddings, centroids, assignments, learning_rate=0.5)
    after = float(np.dot(updated[1], centroids[0]))

    assert updated.shape == embeddings.shape
    assert after > before
    assert np.allclose(np.linalg.norm(updated, axis=1), 1.0, atol=1e-4)


@pytest.mark.cuda
def test_sleep_cluster_assignment_returns_best_centroid_indices():
    _require_gpu()
    kernels = SleepClusterKernels()
    similarities = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.3, 0.7, 0.1],
            [0.2, 0.4, 0.8],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    assignments = kernels.assign_to_best_centroid(similarities)

    assert assignments.tolist() == [0, 1, 2, 0]


@pytest.mark.cuda
def test_sleep_cluster_centroid_accumulation_matches_expected_means():
    _require_gpu()
    kernels = SleepClusterKernels()
    embeddings = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    assignments = np.array([0, 0, 1, 1], dtype=np.int32)

    centroids, counts = kernels.accumulate_centroids(embeddings, assignments, n_clusters=2)

    expected = _normalize_rows(
        np.array(
            [
                embeddings[0] + embeddings[1],
                embeddings[2] + embeddings[3],
            ],
            dtype=np.float32,
        )
    )

    assert counts.tolist() == [2, 2]
    assert np.allclose(centroids, expected, atol=1e-5)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-4)


@pytest.mark.cuda
def test_sleep_cluster_silhouette_scores_are_positive_for_separated_clusters():
    _require_gpu()
    kernels = SleepClusterKernels()
    embeddings = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    assignments = np.array([0, 0, 1, 1], dtype=np.int32)
    scores = kernels.compute_silhouette_scores(embeddings, assignments, n_clusters=2)

    assert scores.shape == (4,)
    assert float(np.mean(scores)) > 0.5


@pytest.mark.cuda
def test_sleep_glyph_kernel_merges_near_identical_rows():
    _require_gpu()
    kernels = SleepGlyphKernels()
    embeddings = _normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    representatives = kernels.cluster_by_similarity(embeddings, similarity_threshold=0.999)

    assert representatives.tolist() == [0, 0, 2]
