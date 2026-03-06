"""High-level clustering helpers built on the sovereign RPN executor."""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium.bridges.sovereign_bridges import VectorResonator
from knowledge3d.cranium.clustering_rpn import compute_similarity_matrix_rpn
from knowledge3d.cranium.ptx_runtime.sleep_cluster_kernels import SleepClusterKernels


class SovereignClusteringOps:
    """
    Convenience wrapper that keeps clustering logic consistent while the lower-level
    RPN executor migrates away from CuPy.

    The heavy similarity math still runs on GPU via `compute_similarity_matrix_rpn`,
    which now uses the sovereign executor. Aggregation and blending follow the
    previously validated CPU/PTX hybrid path.
    """

    def __init__(self) -> None:
        self._resonator = VectorResonator()
        self._sleep_kernels = SleepClusterKernels()

    # ------------------------------------------------------------------ #
    # Similarity + assignments
    # ------------------------------------------------------------------ #
    def cosine_similarity_matrix(
        self,
        vectors: np.ndarray,
        centroids: np.ndarray,
        *,
        batch_size: int = 15,
    ) -> np.ndarray:
        """GPU-backed cosine similarity via the RPN executor."""
        return compute_similarity_matrix_rpn(vectors, centroids, batch_size=batch_size)

    def assign_to_clusters(self, similarities: np.ndarray) -> np.ndarray:
        """PTX-backed argmax routing across centroid similarities."""
        sims = np.asarray(similarities, dtype=np.float32)
        if sims.ndim != 2:
            raise ValueError(f"expected 2D similarity matrix, got shape={sims.shape}")
        if sims.shape[0] == 0:
            return np.empty((0,), dtype=np.int32)
        return self._sleep_kernels.assign_to_best_centroid(sims)

    # ------------------------------------------------------------------ #
    # Centroid updates
    # ------------------------------------------------------------------ #
    def compute_centroids(
        self,
        vectors: np.ndarray,
        assignments: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """PTX-backed centroid accumulation with host-side normalization."""
        vecs = np.asarray(vectors, dtype=np.float32)
        asn = np.asarray(assignments, dtype=np.int32)
        if vecs.ndim != 2:
            raise ValueError(f"expected 2D vectors, got shape={vecs.shape}")
        if asn.ndim != 1:
            raise ValueError(f"expected 1D assignments, got shape={asn.shape}")
        if vecs.shape[0] != asn.shape[0]:
            raise ValueError("vectors and assignments length mismatch")
        return self._sleep_kernels.accumulate_centroids(vecs, asn, int(n_clusters))

    def blend_toward_centroids(
        self,
        vectors: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        """Vector resonator blending (unchanged behaviour)."""
        updated = np.empty_like(vectors)
        for idx, vec in enumerate(vectors):
            centroid = centroids[assignments[idx]]
            blended = self._resonator.resonate(
                vec.astype(np.float32),
                centroid.astype(np.float32),
                float(learning_rate),
            )
            norm = np.linalg.norm(blended)
            if norm > 1e-8:
                blended /= norm
            updated[idx] = blended.astype(np.float32)
        return updated

    def find_redundant_pairs(
        self,
        vectors: np.ndarray,
        assignments: np.ndarray,
        threshold: float,
        cluster_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPU-powered pairwise similarities for redundancy pruning."""
        mask = assignments == cluster_id
        member_indices = np.where(mask)[0]
        if member_indices.size < 2:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        member_vectors = vectors[member_indices]
        sims = compute_similarity_matrix_rpn(member_vectors, member_vectors)

        redundant: list[int] = []
        targets: list[int] = []
        for i in range(member_indices.size):
            for j in range(i + 1, member_indices.size):
                if sims[i, j] >= threshold:
                    redundant.append(member_indices[j])
                    targets.append(member_indices[i])

        return (
            np.asarray(redundant, dtype=np.int32),
            np.asarray(targets, dtype=np.int32),
        )


__all__ = ["SovereignClusteringOps"]
