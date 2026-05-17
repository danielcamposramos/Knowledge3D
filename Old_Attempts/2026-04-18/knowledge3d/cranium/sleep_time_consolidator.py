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
from knowledge3d.cranium.sleep.memory_pressure_trigger import MemoryPressureTrigger
from knowledge3d.cranium.ptx_runtime.sleep_cluster_kernels import SleepClusterKernels
from knowledge3d.cranium.sovereign_clustering_ops import SovereignClusteringOps
from knowledge3d.cranium.ternary_utils import ternary_route


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
    max_assignment_iterations: int = 8
    redundancy_threshold: float = 0.95
    max_cluster_size_for_pruning: int = 1024
    outlier_similarity_threshold: float = 0.55
    outlier_min_cluster_size: int = 4
    outlier_std_factor: float = 2.0
    memory_pressure_threshold_ratio: float = 0.82
    memory_pressure_reserve_bytes: int = 512 * 1024 * 1024
    metrics_path: Path | None = None

    _last_assignments: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_keys: List[int] | None = field(default=None, init=False, repr=False)
    _cohesion_before: float = field(default=0.0, init=False, repr=False)
    _cohesion_after: float = field(default=0.0, init=False, repr=False)
    _clustering_ops: SovereignClusteringOps | None = field(default=None, init=False, repr=False)
    _sleep_kernels: SleepClusterKernels | None = field(default=None, init=False, repr=False)
    _memory_trigger: MemoryPressureTrigger | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize sovereign GPU components (zero CuPy)."""
        self._clustering_ops = SovereignClusteringOps()
        self._sleep_kernels = SleepClusterKernels()
        self._memory_trigger = MemoryPressureTrigger(
            threshold_ratio=float(self.memory_pressure_threshold_ratio),
            reserve_bytes=int(self.memory_pressure_reserve_bytes),
        )

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
        if not self.rpn_engine.pending_consolidation and self.rpn_engine.last_consolidated_at is not None:
            return {
                "status": "skipped",
                "reason": "already_consolidated",
                "timestamp": time.time(),
                "vocab_size": self.rpn_engine.vocab_size,
                "pending_consolidation": self.rpn_engine.pending_consolidation,
                "last_consolidated_at": self.rpn_engine.last_consolidated_at,
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

    def memory_pressure_snapshot(self) -> Dict[str, object]:
        return self._memory_trigger.snapshot().as_dict()

    def should_trigger_memory_pressure(self) -> bool:
        return bool(self._memory_trigger.snapshot().should_consolidate)

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
                "silhouette_before": 0.0,
                "silhouette_after": 0.0,
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

        for _ in range(max(1, int(self.max_assignment_iterations))):
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

        sleep_kernels = self._sleep_kernels
        silhouette_before_scores = sleep_kernels.compute_silhouette_scores(
            embeddings,
            assignments,
            cluster_target,
        )
        silhouette_before = float(np.mean(silhouette_before_scores)) if silhouette_before_scores.size else 0.0

        updated = sleep_kernels.refine_embeddings(
            embeddings,
            centroids,
            assignments,
            float(self.learning_rate),
        )

        updated_similarity = clustering_ops.cosine_similarity_matrix(updated, centroids)
        self._cohesion_after = float(np.mean(updated_similarity[np.arange(num_embeddings), assignments]))
        silhouette_after_scores = sleep_kernels.compute_silhouette_scores(
            updated,
            assignments,
            cluster_target,
        )
        silhouette_after = float(np.mean(silhouette_after_scores)) if silhouette_after_scores.size else 0.0

        for idx, trigram_hash in enumerate(keys):
            self.rpn_engine.embeddings[trigram_hash] = updated[idx]

        self._last_assignments = assignments
        self._last_keys = keys

        improvement = self._cohesion_after - self._cohesion_before
        return {
            "clusters": int(cluster_target),
            "silhouette_before": float(silhouette_before),
            "silhouette_after": float(silhouette_after),
            "cohesion_before": float(self._cohesion_before),
            "cohesion_after": float(self._cohesion_after),
            "improvement": float(improvement),
        }

    # ------------------------------------------------------------------ #
    # Stage 2: Redundancy pruning (RPN similarity with centroid)
    # ------------------------------------------------------------------ #
    def _prune_redundancies(self) -> Dict[str, int | float]:
        if self._last_assignments is None or self._last_keys is None:
            return {"status": "skipped", "merged_pairs": 0, "reduction": 0.0}

        assignments = self._last_assignments
        keys = list(self._last_keys)
        embedding_map = self.rpn_engine.embeddings
        merged_pairs = 0
        uncertain_candidates = 0
        clusters_examined = 0

        clustering_ops = self._clustering_ops
        removed: List[int] = []

        for cluster_id in np.unique(assignments):
            member_idx = np.where(assignments == cluster_id)[0]
            if member_idx.size <= 1:
                continue
            if member_idx.size > self.max_cluster_size_for_pruning:
                continue

            clusters_examined += 1
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
            keeper_local_idx = int(np.argmax(sims))
            max_sim = float(sims[keeper_local_idx])
            span = max(max_sim - float(self.redundancy_threshold), 1e-6)
            remove_mask = np.zeros(member_idx.size, dtype=bool)

            for local_idx, sim in enumerate(sims):
                if local_idx == keeper_local_idx:
                    continue
                if float(sim) < float(self.redundancy_threshold):
                    continue

                confidence = float(np.clip((float(sim) - float(self.redundancy_threshold)) / span, 0.0, 1.0))
                decision = ternary_route(confidence, threshold_low=0.2, threshold_high=0.8)
                if decision == 1:
                    remove_mask[local_idx] = True
                elif decision == 0:
                    uncertain_candidates += 1

            if not np.any(remove_mask):
                continue

            keeper_idx = int(member_idx[keeper_local_idx])
            keeper_key = keys[keeper_idx]
            retained_vectors = member_vectors[~remove_mask]
            retained_assignments = np.zeros(retained_vectors.shape[0], dtype=np.int32)
            retained_centroid_matrix, _ = clustering_ops.compute_centroids(
                retained_vectors,
                retained_assignments,
                n_clusters=1,
            )
            embedding_map[keeper_key] = retained_centroid_matrix[0]

            removed_local_indices = np.where(remove_mask)[0]
            for local_idx in removed_local_indices.tolist():
                removed.append(keys[int(member_idx[local_idx])])
                merged_pairs += 1

        for trigram_hash in removed:
            if trigram_hash in embedding_map:
                del embedding_map[trigram_hash]

        if removed:
            keep_mask = np.asarray([k not in removed for k in keys], dtype=bool)
            self._last_keys = [keys[idx] for idx, keep in enumerate(keep_mask) if keep]
            self._last_assignments = assignments[keep_mask].astype(np.int32, copy=False)
        else:
            self._last_keys = list(keys)
            self._last_assignments = assignments

        reduction = (merged_pairs / max(1, len(keys))) * 100.0
        return {
            "status": "completed",
            "merged_pairs": int(merged_pairs),
            "uncertain_candidates": int(uncertain_candidates),
            "clusters_examined": int(clusters_examined),
            "reduction": float(reduction),
        }

    # ------------------------------------------------------------------ #
    # Remaining stages (placeholders)
    # ------------------------------------------------------------------ #
    def _remove_outliers(self) -> Dict[str, object]:
        if self._last_assignments is None or self._last_keys is None:
            return {"status": "skipped", "reason": "no_cluster_state"}

        assignments = np.asarray(self._last_assignments, dtype=np.int32)
        keys = list(self._last_keys)
        embedding_map = self.rpn_engine.embeddings
        clustering_ops = self._clustering_ops

        removed: List[int] = []
        examined_members = 0
        uncertain_candidates = 0
        clusters_examined = 0

        for cluster_id in np.unique(assignments):
            member_idx = np.where(assignments == cluster_id)[0]
            if member_idx.size < self.outlier_min_cluster_size:
                continue

            member_keys = [keys[i] for i in member_idx]
            member_vectors = np.stack(
                [embedding_map[k] for k in member_keys],
                axis=0,
            ).astype(np.float32)
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

            mean_sim = float(np.mean(sims))
            std_sim = float(np.std(sims))
            adaptive_floor = max(
                float(self.outlier_similarity_threshold),
                mean_sim - float(self.outlier_std_factor) * std_sim,
            )
            max_sim = float(np.max(sims))
            cluster_removed: List[int] = []
            clusters_examined += 1

            for local_idx, sim in enumerate(sims):
                examined_members += 1
                span = max(max_sim - adaptive_floor, 1e-6)
                confidence = float(np.clip((float(sim) - adaptive_floor) / span, 0.0, 1.0))
                decision = ternary_route(confidence, threshold_low=0.2, threshold_high=0.8)
                if float(sim) < adaptive_floor and decision == -1:
                    cluster_removed.append(local_idx)
                elif decision == 0:
                    uncertain_candidates += 1

            # Never collapse a cluster into a singleton or empty set.
            if member_idx.size - len(cluster_removed) < 2:
                continue

            for local_idx in cluster_removed:
                removed.append(member_keys[local_idx])

        for trigram_hash in removed:
            embedding_map.pop(trigram_hash, None)

        if removed:
            keep_mask = np.asarray([k not in removed for k in keys], dtype=bool)
            self._last_keys = [keys[idx] for idx, keep in enumerate(keep_mask) if keep]
            self._last_assignments = assignments[keep_mask].astype(np.int32, copy=False)

        reduction = (len(removed) / max(1, len(keys))) * 100.0
        return {
            "status": "completed",
            "removed_outliers": int(len(removed)),
            "uncertain_candidates": int(uncertain_candidates),
            "examined_members": int(examined_members),
            "clusters_examined": int(clusters_examined),
            "reduction": float(reduction),
        }

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
