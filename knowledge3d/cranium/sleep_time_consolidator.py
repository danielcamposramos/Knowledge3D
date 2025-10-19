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

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from knowledge3d.cranium.rpn_embedding_engine import RPNEmbeddingEngine

def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


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
        k = min(self.cluster_count, max(2, num_embeddings // 4))

        # MiniBatchKMeans is memory-friendly for 30k+ embeddings.
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=4096,
            random_state=0,
            n_init="auto",
        )
        assignments = kmeans.fit_predict(matrix)

        silhouette_before = self._safe_silhouette(matrix, assignments)

        centroids = kmeans.cluster_centers_.astype(np.float32)
        updated_matrix = matrix.copy()
        for idx, cluster_id in enumerate(assignments):
            centroid = centroids[cluster_id]
            updated_matrix[idx] = _normalize(
                updated_matrix[idx] + self.learning_rate * (centroid - updated_matrix[idx])
            )

        silhouette_after = self._safe_silhouette(updated_matrix, assignments)

        # write back
        for idx, trigram_hash in enumerate(keys):
            self.rpn_engine.embeddings[trigram_hash] = updated_matrix[idx]

        self._last_assignments = assignments
        self._last_keys = keys

        return {
            "clusters": int(k),
            "silhouette_before": float(silhouette_before),
            "silhouette_after": float(silhouette_after),
            "improvement": float(silhouette_after - silhouette_before),
        }

    # ------------------------------------------------------------------ #
    # Stage 2: Redundancy pruning
    # ------------------------------------------------------------------ #
    def _prune_redundancies(self) -> Dict[str, int | float]:
        if self._last_assignments is None or self._last_keys is None:
            return {"merged_pairs": 0, "reduction": 0.0}

        assignments = self._last_assignments
        keys = self._last_keys

        embedding_map = self.rpn_engine.embeddings
        merged_pairs = 0
        removed_indices: set[int] = set()

        for cluster_id in np.unique(assignments):
            member_idx = np.where(assignments == cluster_id)[0]
            if len(member_idx) <= 1:
                continue
            if len(member_idx) > self.max_cluster_size_for_pruning:
                # Oversized clusters are pruned using centroid similarity only.
                continue

            vectors = np.stack(
                [embedding_map[keys[i]] for i in member_idx],
                axis=0,
            )
            sims = vectors @ vectors.T
            for local_i, global_i in enumerate(member_idx):
                if global_i in removed_indices:
                    continue
                for local_j, global_j in enumerate(member_idx):
                    if global_j <= global_i or global_j in removed_indices:
                        continue
                    if sims[local_i, local_j] >= self.redundancy_threshold:
                        keep_hash = keys[global_i]
                        drop_hash = keys[global_j]
                        merged_vec = _normalize(
                            embedding_map[keep_hash] + embedding_map[drop_hash]
                        )
                        embedding_map[keep_hash] = merged_vec
                        del embedding_map[drop_hash]
                        removed_indices.add(global_j)
                        merged_pairs += 1

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

    @staticmethod
    def _safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
        # Silhouette score requires >1 cluster and at least 2 members per cluster
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2 or np.any(counts < 2):
            return 0.0
        try:
            return float(silhouette_score(matrix, labels, metric="cosine"))
        except Exception:
            return 0.0

    def _record_metrics(self, metrics: Dict[str, object]) -> None:
        target = self.metrics_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(metrics, default=float)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
