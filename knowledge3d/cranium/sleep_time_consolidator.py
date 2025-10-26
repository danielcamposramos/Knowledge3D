"""
Phase D Sleep-Time Consolidation utilities.

The consolidator tightens the RPN embedding space after ingestion so that new
knowledge is integrated once during an idle window instead of being reinforced
by repeated re-ingestion. The initial implementation focuses on:

* Cluster refinement via k-means centroids
* Redundancy pruning within clusters (high cosine similarity merges)

Outlier removal and swarm-feedback refinement are stubbed with informative
messages so the Phase D roadmap can extend them without touching the public API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence, Tuple

import cupy as cp
import numpy as np

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine
from knowledge3d.cranium.rpn_executor import get_rpn_executor
from knowledge3d.cranium.clustering_rpn import compile_cosine_similarity_rpn


def _normalize_rows_gpu(matrix: cp.ndarray) -> cp.ndarray:
    """L2-normalise each row in a GPU matrix."""
    norms = cp.linalg.norm(matrix, axis=1, keepdims=True)
    norms = cp.where(norms < 1e-8, 1.0, norms)
    return matrix / norms


def _to_numpy(array: cp.ndarray) -> np.ndarray:
    """Convert a CuPy array to a contiguous NumPy float32 array."""
    return cp.asnumpy(array).astype(np.float32, copy=False)


@dataclass
class SleepTimeConsolidator:
    """
    Consolidate RPN embeddings during idle windows.

    Parameters
    ----------
    rpn_engine:
        The embedding engine whose vectors will be refined.
    cluster_count:
        Number of clusters for k-means refinement.
    learning_rate:
        How far to move embeddings toward cluster centroids (0-1).
    redundancy_threshold:
        Cosine similarity threshold for merging redundant trigrams.
    max_cluster_size_for_pruning:
        Safety cap to avoid quadratic explosion in giant clusters.
    metrics_path:
        Optional JSONL file for logging consolidation metrics.
    """

    rpn_engine: RPNEmbeddingEngine
    cluster_count: int = 256
    learning_rate: float = 0.2
    redundancy_threshold: float = 0.95
    max_cluster_size_for_pruning: int = 1024
    metrics_path: Path | None = None

    # internal state cached per run
    _last_assignments: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_keys: List[int] | None = field(default=None, init=False, repr=False)
    _gpu_embeddings: cp.ndarray | None = field(default=None, init=False, repr=False)
    _gpu_assignments: cp.ndarray | None = field(default=None, init=False, repr=False)
    _cohesion_before: float = field(default=0.0, init=False, repr=False)
    _cohesion_after: float = field(default=0.0, init=False, repr=False)

    def consolidate(self) -> Dict[str, object]:
        """
        Run the full consolidation pipeline.

        Returns
        -------
        Dict with per-stage metrics and summary totals.
        """
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
    # Stage 1: Cluster refinement
    # ------------------------------------------------------------------ #
    def _refine_clusters(self) -> Dict[str, float]:
        keys, matrix = self._embedding_items()
        num_embeddings = matrix.shape[0]

        if num_embeddings < 2:
            self._gpu_embeddings = None
            self._gpu_assignments = None
            self._last_assignments = None
            self._last_keys = keys
            return {
                "clusters": int(max(1, num_embeddings)),
                "cohesion_before": 0.0,
                "cohesion_after": 0.0,
                "improvement": 0.0,
            }

        cluster_target = max(2, min(self.cluster_count, num_embeddings))

        data_gpu = cp.asarray(matrix, dtype=cp.float32)
        data_gpu = _normalize_rows_gpu(data_gpu)
        original_gpu = data_gpu.copy()

        rng = cp.random.default_rng(seed=0)
        choices = rng.choice(num_embeddings, size=cluster_target, replace=False)
        centroids = data_gpu[choices].copy()

        assignments = None
        for _ in range(25):
            similarity = data_gpu @ centroids.T
            new_assignments = cp.argmax(similarity, axis=1)
            if assignments is not None and cp.all(new_assignments == assignments):
                assignments = new_assignments
                break

            assignments = new_assignments
            for idx in range(cluster_target):
                member_mask = assignments == idx
                member_count = int(member_mask.sum().get())
                if member_count == 0:
                    random_idx = int(rng.integers(0, num_embeddings))
                    centroids[idx] = data_gpu[random_idx]
                    continue
                cluster_vectors = data_gpu[member_mask]
                centroid = cluster_vectors.mean(axis=0)
                norm = cp.linalg.norm(centroid)
                if float(norm) > 1e-6:
                    centroids[idx] = centroid / norm
                else:
                    centroids[idx] = cluster_vectors[0]

        if assignments is None:
            similarity = data_gpu @ centroids.T
            assignments = cp.argmax(similarity, axis=1)

        centroid_assign = centroids[assignments]
        rpn_executor = get_rpn_executor()
        sample_count = min(15, num_embeddings)
        sample_scores_before: List[float] = []
        for sample_idx in range(sample_count):
            cluster_idx = int(assignments[sample_idx].item())
            program = compile_cosine_similarity_rpn(
                _to_numpy(original_gpu[sample_idx]),
                _to_numpy(centroids[cluster_idx]),
            )
            score = rpn_executor.execute_single(
                instance_id=sample_idx % rpn_executor.MAX_INSTANCES,
                op_codes=program["op_codes"],
                scalars=program["scalars"],
                vectors=program["vectors"],
            )
            sample_scores_before.append(score)
        self._cohesion_before = float(np.mean(sample_scores_before)) if sample_scores_before else 0.0

        updated_gpu = _normalize_rows_gpu(
            original_gpu + self.learning_rate * (centroid_assign - original_gpu)
        )

        updated_similarity = updated_gpu @ centroids.T
        updated_assignments = cp.argmax(updated_similarity, axis=1)
        updated_centroids = centroids[updated_assignments]
        sample_scores_after: List[float] = []
        for sample_idx in range(sample_count):
            cluster_idx = int(updated_assignments[sample_idx].item())
            program = compile_cosine_similarity_rpn(
                _to_numpy(updated_gpu[sample_idx]),
                _to_numpy(updated_centroids[sample_idx]),
            )
            score = rpn_executor.execute_single(
                instance_id=sample_idx % rpn_executor.MAX_INSTANCES,
                op_codes=program["op_codes"],
                scalars=program["scalars"],
                vectors=program["vectors"],
            )
            sample_scores_after.append(score)
        self._cohesion_after = float(np.mean(sample_scores_after)) if sample_scores_after else 0.0

        self._gpu_embeddings = updated_gpu
        self._gpu_assignments = updated_assignments
        self._last_assignments = cp.asnumpy(updated_assignments)
        self._last_keys = keys

        updated_matrix = _to_numpy(updated_gpu)
        for idx, trigram_hash in enumerate(keys):
            self.rpn_engine.embeddings[trigram_hash] = updated_matrix[idx]

        improvement = self._cohesion_after - self._cohesion_before
        return {
            "clusters": int(cluster_target),
            "cohesion_before": float(self._cohesion_before),
            "cohesion_after": float(self._cohesion_after),
            "improvement": float(improvement),
        }

    # ------------------------------------------------------------------ #
    # Stage 2: Redundancy pruning
    # ------------------------------------------------------------------ #
    def _prune_redundancies(self) -> Dict[str, int | float]:
        if (
            self._gpu_embeddings is None
            or self._gpu_assignments is None
            or self._last_keys is None
        ):
            return {"merged_pairs": 0, "reduction": 0.0}

        data_gpu = self._gpu_embeddings
        assignments_gpu = self._gpu_assignments
        keys = self._last_keys

        keep_mask = cp.ones(data_gpu.shape[0], dtype=cp.bool_)
        merged_pairs = 0

        unique_clusters = cp.unique(assignments_gpu)
        for cluster_id in cp.asnumpy(unique_clusters):
            cluster_indices = cp.where(assignments_gpu == cluster_id)[0]
            cluster_size = int(cluster_indices.size)
            if cluster_size <= 1:
                continue
            if cluster_size > self.max_cluster_size_for_pruning:
                continue

            cluster_vectors = data_gpu[cluster_indices]
            centroid = cluster_vectors.mean(axis=0)
            norm = cp.linalg.norm(centroid)
            if float(norm) > 1e-6:
                centroid = centroid / norm

            sims = cluster_vectors @ centroid
            high_mask = sims >= self.redundancy_threshold
            high_indices = cluster_indices[cp.where(high_mask)[0]]

            if high_indices.size <= 1:
                continue

            best_idx = int(high_indices[cp.argmax(sims[cp.where(high_mask)[0]])].item())
            merged_vec = cluster_vectors[cp.where(high_mask)[0]].mean(axis=0)
            merged_norm = cp.linalg.norm(merged_vec)
            if float(merged_norm) > 1e-6:
                merged_vec = merged_vec / merged_norm

            data_gpu[best_idx] = merged_vec

            for idx in cp.asnumpy(high_indices):
                if idx != best_idx:
                    keep_mask[idx] = False
                    merged_pairs += 1

        keep_mask_cpu = cp.asnumpy(keep_mask)
        updated_embeddings = _to_numpy(data_gpu)

        for idx, trigram_hash in enumerate(list(keys)):
            if keep_mask_cpu[idx]:
                self.rpn_engine.embeddings[trigram_hash] = updated_embeddings[idx]
            elif trigram_hash in self.rpn_engine.embeddings:
                del self.rpn_engine.embeddings[trigram_hash]

        self._last_keys = [keys[i] for i, keep in enumerate(keep_mask_cpu) if keep]
        if self._last_keys:
            self._last_assignments = assignments_gpu[keep_mask].get().astype(np.int32)
        else:
            self._last_assignments = None

        self._gpu_embeddings = data_gpu[keep_mask]
        self._gpu_assignments = assignments_gpu[keep_mask]

        reduction = (merged_pairs / max(1, len(keys))) * 100.0
        return {"merged_pairs": merged_pairs, "reduction": reduction}

    # ------------------------------------------------------------------ #
    # Stage 3 / 4 placeholders
    # ------------------------------------------------------------------ #
    def _remove_outliers(self) -> Dict[str, object]:
        # Usage tracking is not yet instrumented in the ingestion loop.
        return {"status": "skipped", "reason": "hit_count_not_tracked"}

    def _integrate_swarm_feedback(self) -> Dict[str, object]:
        return {"status": "skipped", "reason": "phase_d3_pending"}

    # ------------------------------------------------------------------ #
    # Helper routines
    # ------------------------------------------------------------------ #
    def _embedding_items(self) -> Tuple[List[int], np.ndarray]:
        embeddings: MutableMapping[int, np.ndarray] = self.rpn_engine.embeddings
        keys = list(embeddings.keys())
        matrix = np.vstack([embeddings[k] for k in keys]).astype(np.float32)
        return keys, matrix

    def _record_metrics(self, metrics: Dict[str, object]) -> None:
        target = self.metrics_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(metrics, default=float)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
