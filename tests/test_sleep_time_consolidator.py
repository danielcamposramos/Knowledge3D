import time

import numpy as np
import pytest

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def _make_cluster(center: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    samples = []
    for _ in range(count):
        perturb = rng.normal(scale=0.05, size=center.shape)
        vec = center + perturb
        vec /= np.linalg.norm(vec) + 1e-8
        samples.append(vec.astype(np.float32))
    return np.vstack(samples)


def build_engine() -> RPNEmbeddingEngine:
    rng = np.random.default_rng(seed=1234)
    engine = RPNEmbeddingEngine()
    centers = rng.normal(size=(3, engine.embedding_dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8

    embeddings = []
    for center in centers:
        embeddings.append(_make_cluster(center, 12, rng))

    matrix = np.vstack(embeddings)
    for idx, vec in enumerate(matrix):
        trigram_hash = idx + 1  # deterministic
        engine.embeddings[trigram_hash] = vec.astype(np.float32)

    engine.vocab_size = len(engine.embeddings)
    engine.mark_unconsolidated()
    return engine


@pytest.mark.cuda
def test_sleep_time_consolidation_improves_clusters(tmp_path):
    _require_gpu()
    engine = build_engine()
    metrics_path = tmp_path / "sleep_metrics.jsonl"
    consolidator = SleepTimeConsolidator(
        engine,
        cluster_count=3,
        learning_rate=0.3,
        max_assignment_iterations=4,
        redundancy_threshold=0.99,
        metrics_path=metrics_path,
    )

    metrics = consolidator.consolidate()

    cluster_metrics = metrics["cluster_refinement"]
    assert "silhouette_before" in cluster_metrics
    assert "silhouette_after" in cluster_metrics
    assert "cohesion_before" in cluster_metrics
    assert "cohesion_after" in cluster_metrics
    assert cluster_metrics["silhouette_after"] >= cluster_metrics["silhouette_before"]
    assert not engine.pending_consolidation
    assert engine.last_consolidated_at is not None
    assert metrics_path.exists()

    # Subsequent run on already consolidated embeddings should be idempotent
    time.sleep(0.01)
    second_metrics = consolidator.consolidate()
    assert second_metrics["status"] == "skipped"
    assert second_metrics["reason"] == "already_consolidated"


@pytest.mark.cuda
def test_sleep_time_outlier_removal_prunes_clear_cluster_outlier():
    _require_gpu()
    engine = RPNEmbeddingEngine()
    dim = engine.embedding_dim

    center = np.zeros(dim, dtype=np.float32)
    center[0] = 1.0
    near_members = []
    for delta in (0.0, 0.03, -0.02, 0.01):
        vec = center.copy()
        vec[1] = delta
        vec /= np.linalg.norm(vec) + 1e-8
        near_members.append(vec.astype(np.float32))

    outlier = np.zeros(dim, dtype=np.float32)
    outlier[1] = 1.0
    outlier /= np.linalg.norm(outlier) + 1e-8

    all_vectors = near_members + [outlier.astype(np.float32)]
    for idx, vec in enumerate(all_vectors, start=1):
        engine.embeddings[idx] = vec
    engine.vocab_size = len(engine.embeddings)

    consolidator = SleepTimeConsolidator(
        engine,
        cluster_count=1,
        outlier_similarity_threshold=0.8,
        outlier_min_cluster_size=4,
        outlier_std_factor=1.5,
    )
    consolidator._last_keys = [1, 2, 3, 4, 5]
    consolidator._last_assignments = np.zeros(5, dtype=np.int32)

    result = consolidator._remove_outliers()

    assert result["status"] == "completed"
    assert result["removed_outliers"] == 1
    assert 5 not in engine.embeddings
    assert consolidator._last_keys == [1, 2, 3, 4]
    assert consolidator._last_assignments.tolist() == [0, 0, 0, 0]


@pytest.mark.cuda
def test_sleep_time_redundancy_pruning_merges_duplicate_cluster_members():
    _require_gpu()
    engine = RPNEmbeddingEngine()
    dim = engine.embedding_dim

    def _axis_vec(axis: int, delta: float = 0.0) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        vec[axis] = 1.0
        if delta:
            vec[(axis + 1) % dim] = delta
        vec /= np.linalg.norm(vec) + 1e-8
        return vec.astype(np.float32)

    engine.embeddings[1] = _axis_vec(0, 0.00)
    engine.embeddings[2] = _axis_vec(0, 0.01)
    engine.embeddings[3] = _axis_vec(0, -0.01)
    engine.embeddings[4] = _axis_vec(1, 0.00)
    engine.vocab_size = len(engine.embeddings)

    consolidator = SleepTimeConsolidator(
        engine,
        redundancy_threshold=0.99,
        max_cluster_size_for_pruning=8,
    )
    consolidator._last_keys = [1, 2, 3, 4]
    consolidator._last_assignments = np.array([0, 0, 0, 1], dtype=np.int32)

    result = consolidator._prune_redundancies()

    assert result["status"] == "completed"
    assert result["merged_pairs"] == 2
    assert result["clusters_examined"] == 1
    assert len(consolidator._last_keys) == 2
    assert set(consolidator._last_keys) == {1, 4}
    assert consolidator._last_assignments.tolist() == [0, 1]
    assert 2 not in engine.embeddings
    assert 3 not in engine.embeddings
    assert np.isclose(np.linalg.norm(engine.embeddings[1]), 1.0, atol=1e-4)
