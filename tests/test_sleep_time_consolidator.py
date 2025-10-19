import time

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sleep_time_consolidator import SleepTimeConsolidator


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
        embeddings.append(_make_cluster(center, 40, rng))

    matrix = np.vstack(embeddings)
    for idx, vec in enumerate(matrix):
        trigram_hash = idx + 1  # deterministic
        engine.embeddings[trigram_hash] = vec.astype(np.float32)

    engine.vocab_size = len(engine.embeddings)
    engine.mark_unconsolidated()
    return engine


def test_sleep_time_consolidation_improves_clusters(tmp_path):
    engine = build_engine()
    metrics_path = tmp_path / "sleep_metrics.jsonl"
    consolidator = SleepTimeConsolidator(
        engine,
        cluster_count=3,
        learning_rate=0.3,
        redundancy_threshold=0.99,
        metrics_path=metrics_path,
    )

    metrics = consolidator.consolidate()

    # Ensure silhouette improved
    cluster_metrics = metrics["cluster_refinement"]
    assert cluster_metrics["silhouette_after"] >= cluster_metrics["silhouette_before"]
    assert not engine.pending_consolidation
    assert engine.last_consolidated_at is not None
    assert metrics_path.exists()

    # Subsequent run on already consolidated embeddings should be idempotent
    time.sleep(0.01)
    second_metrics = consolidator.consolidate()
    assert second_metrics["cluster_refinement"]["silhouette_after"] >= cluster_metrics["silhouette_after"]
