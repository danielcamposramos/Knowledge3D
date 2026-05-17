"""Canonical PTX-backed sleep-time clustering kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

SLEEP_CLUSTER_PTX = Path(__file__).parent.parent / "ptx" / "sleep_cluster_refiner.ptx"

_MODULE = None
_FUNCTIONS: dict[str, loader.CUfunction] = {}


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = loader.load_module_from_file(str(SLEEP_CLUSTER_PTX))
    return _MODULE


def _get_function(name: str) -> loader.CUfunction:
    fn = _FUNCTIONS.get(name)
    if fn is None:
        fn = loader.get_function(_get_module(), name)
        _FUNCTIONS[name] = fn
    return fn


def _as_float32_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D float32 matrix, got shape={arr.shape}")
    return arr


def _as_int32_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(vector, dtype=np.int32))
    if arr.ndim != 1:
        raise ValueError(f"expected 1D int32 vector, got shape={arr.shape}")
    return arr


class SleepClusterKernels:
    """PTX wrappers for embedding refinement and silhouette scoring."""

    def __init__(self) -> None:
        _get_module()

    def refine_embeddings(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        learning_rate: float,
    ) -> np.ndarray:
        emb = _as_float32_matrix(embeddings)
        ctr = _as_float32_matrix(centroids)
        asn = _as_int32_vector(assignments)
        if emb.shape[0] != asn.shape[0]:
            raise ValueError("embeddings and assignments length mismatch")
        if emb.shape[1] != ctr.shape[1]:
            raise ValueError("embeddings and centroids dimension mismatch")

        n, dim = emb.shape
        k = int(ctr.shape[0])
        if n == 0:
            return np.empty_like(emb)

        updated = np.array(emb, copy=True, dtype=np.float32, order="C")
        d_emb = loader.gpu_malloc(updated.nbytes)
        d_ctr = loader.gpu_malloc(ctr.nbytes)
        d_asn = loader.gpu_malloc(asn.nbytes)
        try:
            loader.memcpy_htod(d_emb, updated.ctypes.data_as(ctypes.c_void_p), updated.nbytes)
            loader.memcpy_htod(d_ctr, ctr.ctypes.data_as(ctypes.c_void_p), ctr.nbytes)
            loader.memcpy_htod(d_asn, asn.ctypes.data_as(ctypes.c_void_p), asn.nbytes)
            loader.launch(
                _get_function("refine_embeddings_to_centroids"),
                grid=((n + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_emb,
                    ctypes.c_int(int(n)),
                    ctypes.c_int(int(dim)),
                    d_ctr,
                    d_asn,
                    ctypes.c_int(int(k)),
                    ctypes.c_float(float(learning_rate)),
                ],
            )
            loader.memcpy_dtoh(updated.ctypes.data_as(ctypes.c_void_p), d_emb, updated.nbytes)
            return updated
        finally:
            loader.gpu_free(d_emb)
            loader.gpu_free(d_ctr)
            loader.gpu_free(d_asn)

    def assign_to_best_centroid(self, similarities: np.ndarray) -> np.ndarray:
        sims = _as_float32_matrix(similarities)
        n, k = sims.shape
        if n == 0:
            return np.empty((0,), dtype=np.int32)

        assignments = np.empty((n,), dtype=np.int32)
        d_sims = loader.gpu_malloc(sims.nbytes)
        d_out = loader.gpu_malloc(assignments.nbytes)
        try:
            loader.memcpy_htod(d_sims, sims.ctypes.data_as(ctypes.c_void_p), sims.nbytes)
            loader.launch(
                _get_function("assign_to_best_centroid"),
                grid=((n + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_sims,
                    d_out,
                    ctypes.c_int(int(n)),
                    ctypes.c_int(int(k)),
                ],
            )
            loader.memcpy_dtoh(
                assignments.ctypes.data_as(ctypes.c_void_p),
                d_out,
                assignments.nbytes,
            )
            return assignments
        finally:
            loader.gpu_free(d_sims)
            loader.gpu_free(d_out)

    def accumulate_centroids(
        self,
        embeddings: np.ndarray,
        assignments: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        emb = _as_float32_matrix(embeddings)
        asn = _as_int32_vector(assignments)
        if emb.shape[0] != asn.shape[0]:
            raise ValueError("embeddings and assignments length mismatch")
        if n_clusters < 0:
            raise ValueError("n_clusters must be >= 0")

        n, dim = emb.shape
        if n == 0 or n_clusters == 0:
            return (
                np.zeros((int(n_clusters), dim), dtype=np.float32),
                np.zeros((int(n_clusters),), dtype=np.int32),
            )

        sums = np.zeros((int(n_clusters), dim), dtype=np.float32)
        counts = np.zeros((int(n_clusters),), dtype=np.int32)
        centroids = np.zeros((int(n_clusters), dim), dtype=np.float32)
        d_emb = loader.gpu_malloc(emb.nbytes)
        d_asn = loader.gpu_malloc(asn.nbytes)
        d_sums = loader.gpu_malloc(sums.nbytes)
        d_counts = loader.gpu_malloc(counts.nbytes)
        d_centroids = loader.gpu_malloc(centroids.nbytes)
        try:
            loader.memcpy_htod(d_emb, emb.ctypes.data_as(ctypes.c_void_p), emb.nbytes)
            loader.memcpy_htod(d_asn, asn.ctypes.data_as(ctypes.c_void_p), asn.nbytes)
            loader.memcpy_htod(d_sums, sums.ctypes.data_as(ctypes.c_void_p), sums.nbytes)
            loader.memcpy_htod(d_counts, counts.ctypes.data_as(ctypes.c_void_p), counts.nbytes)
            loader.launch(
                _get_function("accumulate_centroid_sums"),
                grid=((n * dim + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_emb,
                    d_asn,
                    d_sums,
                    d_counts,
                    ctypes.c_int(int(n)),
                    ctypes.c_int(int(dim)),
                    ctypes.c_int(int(n_clusters)),
                ],
            )
            loader.launch(
                _get_function("finalize_centroids"),
                grid=((int(n_clusters) * dim + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_sums,
                    d_counts,
                    d_centroids,
                    ctypes.c_int(int(n_clusters)),
                    ctypes.c_int(int(dim)),
                ],
            )
            loader.memcpy_dtoh(counts.ctypes.data_as(ctypes.c_void_p), d_counts, counts.nbytes)
            loader.memcpy_dtoh(
                centroids.ctypes.data_as(ctypes.c_void_p),
                d_centroids,
                centroids.nbytes,
            )
        finally:
            loader.gpu_free(d_emb)
            loader.gpu_free(d_asn)
            loader.gpu_free(d_sums)
            loader.gpu_free(d_counts)
            loader.gpu_free(d_centroids)

        return centroids.astype(np.float32, copy=False), counts

    def compute_silhouette_scores(
        self,
        embeddings: np.ndarray,
        assignments: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        emb = _as_float32_matrix(embeddings)
        asn = _as_int32_vector(assignments)
        if emb.shape[0] != asn.shape[0]:
            raise ValueError("embeddings and assignments length mismatch")
        n, dim = emb.shape
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        out = np.empty((n,), dtype=np.float32)
        d_scores = loader.gpu_malloc(out.nbytes)
        d_emb = loader.gpu_malloc(emb.nbytes)
        d_asn = loader.gpu_malloc(asn.nbytes)
        try:
            loader.memcpy_htod(d_emb, emb.ctypes.data_as(ctypes.c_void_p), emb.nbytes)
            loader.memcpy_htod(d_asn, asn.ctypes.data_as(ctypes.c_void_p), asn.nbytes)
            loader.launch(
                _get_function("compute_silhouette_scores"),
                grid=((n + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_scores,
                    d_emb,
                    d_asn,
                    ctypes.c_int(int(n)),
                    ctypes.c_int(int(dim)),
                    ctypes.c_int(int(n_clusters)),
                ],
            )
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_scores, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_scores)
            loader.gpu_free(d_emb)
            loader.gpu_free(d_asn)


__all__ = ["SleepClusterKernels"]
