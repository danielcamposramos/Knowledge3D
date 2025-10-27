"""
Sovereign Sleep-Time Consolidator
---------------------------------

Refines the RPN embedding table using sovereign PTX primitives ONLY.
Zero CuPy, minimal NumPy (orchestration only).

The consolidator relies on:
  * Extended RPN kernel (Tier 1-2) for clustering operations
  * VectorResonator PTX kernel for gradual vector blending
  * Modular RPN executor for cosine similarity

All heavy math runs on GPU via sovereign PTX kernels.
CPU is used purely for orchestration and lightweight array bookkeeping.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence, Tuple

import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.sovereign_clustering_ops import SovereignClusteringOps


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


@dataclass
class SleepTimeConsolidator:
    rpn_engine: RPNEmbeddingEngine
    cluster_count: int = 256
    learning_rate: float = 0.2
    redundancy_threshold: float = 0.95
    max_cluster_size_for_pruning: int = 1024
    metrics_path: Path | None = None

    _last_assignments: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_keys: List[int] | None = field(default=None, init=False, repr=False)
    _cohesion_before: float = field(default=0.0, init=False, repr=False)
    _cohesion_after: float = field(default=0.0, init=False, repr=False)
    _clustering_ops: SovereignClusteringOps | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize sovereign GPU components (zero CuPy)."""
        self._clustering_ops = SovereignClusteringOps()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def consolidate(self) -> Dict[str, object]:
        if not self.rpn_engine.embeddings:
            return {
                "status": "skipped",
                "reason": "no_embeddings",
                "timestamp": time.time(),
            }

        start = time.time()
        self.rpn_engine.mark_unconsolidated()

        metrics: Dict[str, object] = {
            "timestamp": start,
            "cluster_refinement": self._refine_clusters(),
            "redundancy_pruning": self._prune_redundancies(),
            "outlier_removal": self._remove_outliers(),
            "swarm_feedback": self._integrate_swarm_feedback(),
        }

        self.rpn_engine.vocab_size = len(self.rpn_engine.embeddings)
        self.rpn_engine.mark_consolidated()

        metrics["elapsed_seconds"] = time.time() - start
        metrics["vocab_size"] = self.rpn_engine.vocab_size
        metrics["pending_consolidation"] = self.rpn_engine.pending_consolidation

        if self.metrics_path is not None:
            self._record_metrics(metrics)

        return metrics

    # ------------------------------------------------------------------ #
    # Stage 1: Cluster refinement (RPN similarities + PTX blending)
    # ------------------------------------------------------------------ #
    def _refine_clusters(self) -> Dict[str, float]:
        keys, matrix = self._embedding_items()
        num_embeddings = matrix.shape[0]

        if num_embeddings < 2:
            self._last_assignments = None
            self._last_keys = keys
            self._cohesion_before = 0.0
            self._cohesion_after = 0.0
            return {
                "clusters": int(max(1, num_embeddings)),
                "cohesion_before": 0.0,
                "cohesion_after": 0.0,
                "improvement": 0.0,
            }

        embeddings = matrix.astype(np.float32, copy=True)
        cluster_target = max(2, min(self.cluster_count, num_embeddings))

        rng = np.random.default_rng(0)
        seeds = rng.choice(num_embeddings, size=cluster_target, replace=False)
        centroids = np.stack([_normalize(embeddings[idx]) for idx in seeds], axis=0)

        assignments = np.zeros(num_embeddings, dtype=np.int32)
        prev = None
        clustering_ops = self._clustering_ops

        for _ in range(25):
            similarity = clustering_ops.cosine_similarity_matrix(embeddings, centroids)
            assignments = clustering_ops.assign_to_clusters(similarity)
            if prev is not None and np.array_equal(assignments, prev):
                break
            prev = assignments.copy()

            centroids, counts = clustering_ops.compute_centroids(
                embeddings,
                assignments,
                cluster_target,
            )
            for cluster_idx, count in enumerate(counts):
                if count == 0:
                    random_idx = int(rng.integers(0, num_embeddings))
                    centroids[cluster_idx] = _normalize(embeddings[random_idx])

        similarity = clustering_ops.cosine_similarity_matrix(embeddings, centroids)
        self._cohesion_before = float(np.mean(similarity[np.arange(num_embeddings), assignments]))

        updated = clustering_ops.blend_toward_centroids(
            embeddings,
            centroids,
            assignments,
            float(self.learning_rate),
        )

        updated_similarity = clustering_ops.cosine_similarity_matrix(updated, centroids)
        self._cohesion_after = float(np.mean(updated_similarity[np.arange(num_embeddings), assignments]))

        for idx, trigram_hash in enumerate(keys):
            self.rpn_engine.embeddings[trigram_hash] = updated[idx]

        self._last_assignments = assignments
        self._last_keys = keys

        improvement = self._cohesion_after - self._cohesion_before
        return {
            "clusters": int(cluster_target),
            "cohesion_before": float(self._cohesion_before),
            "cohesion_after": float(self._cohesion_after),
            "improvement": float(improvement),
        }

    # ------------------------------------------------------------------ #
    # Stage 2: Redundancy pruning (RPN similarity with centroid)
    # ------------------------------------------------------------------ #
    def _prune_redundancies(self) -> Dict[str, int | float]:
        if self._last_assignments is None or self._last_keys is None:
            return {"merged_pairs": 0, "reduction": 0.0}

        assignments = self._last_assignments
        keys = list(self._last_keys)
        embedding_map = self.rpn_engine.embeddings
        merged_pairs = 0

        clustering_ops = self._clustering_ops
        removed: List[int] = []

        for cluster_id in np.unique(assignments):
            member_idx = np.where(assignments == cluster_id)[0]
            if member_idx.size <= 1:
                continue
            if member_idx.size > self.max_cluster_size_for_pruning:
                continue

            member_vectors = np.stack([embedding_map[keys[i]] for i in member_idx], axis=0).astype(
                np.float32
            )

            local_assignments = np.zeros(member_idx.size, dtype=np.int32)
            centroid_matrix, _ = clustering_ops.compute_centroids(
                member_vectors,
                local_assignments,
                n_clusters=1,
            )
            centroid = centroid_matrix[0]

            sims = clustering_ops.cosine_similarity_matrix(
                member_vectors,
                centroid[np.newaxis, :],
            )[:, 0]

            high_members = [
                (cluster_member_idx, member_vectors[i], sims[i])
                for i, cluster_member_idx in enumerate(member_idx)
                if sims[i] >= self.redundancy_threshold
            ]

            if len(high_members) <= 1:
                continue

            high_members.sort(key=lambda item: float(item[2]), reverse=True)
            keeper_idx, keeper_vec, _ = high_members[0]
            keeper_key = keys[keeper_idx]
            aggregate = keeper_vec.astype(np.float32, copy=True)
            blend_count = 1

            for idx, vec, _ in high_members[1:]:
                blend_count += 1
                alpha = 1.0 / blend_count
                aggregate = clustering_ops.blend_toward_centroids(
                    aggregate[np.newaxis, :],
                    vec[np.newaxis, :],
                    np.zeros(1, dtype=np.int32),
                    float(alpha),
                )[0]

                removed.append(keys[idx])
                merged_pairs += 1

            embedding_map[keeper_key] = aggregate

        for trigram_hash in removed:
            if trigram_hash in embedding_map:
                del embedding_map[trigram_hash]

        self._last_keys = [k for k in keys if k not in removed]
        self._last_assignments = None

        reduction = (merged_pairs / max(1, len(keys))) * 100.0
        return {"merged_pairs": merged_pairs, "reduction": reduction}

    # ------------------------------------------------------------------ #
    # Remaining stages (placeholders)
    # ------------------------------------------------------------------ #
    def _remove_outliers(self) -> Dict[str, object]:
        return {"status": "skipped", "reason": "hit_count_not_tracked"}

    def _integrate_swarm_feedback(self) -> Dict[str, object]:
        return {"status": "skipped", "reason": "phase_d3_pending"}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _embedding_items(self) -> Tuple[List[int], np.ndarray]:
        embeddings: MutableMapping[int, np.ndarray] = self.rpn_engine.embeddings
        keys = list(embeddings.keys())
        matrix = np.vstack([embeddings[k] for k in keys]).astype(np.float32)
        return keys, matrix

    def _record_metrics(self, metrics: Dict[str, object]) -> None:
        if self.metrics_path is None:
            return
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(metrics, default=float)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
