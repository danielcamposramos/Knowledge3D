from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence

import numpy as np

from knowledge3d.cranium.sovereign import loader


class CosineSimilarityBridge:
    """GPU batch cosine similarity via sovereign PTX."""

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "cosine_similarity.ptx"
        module = loader.load_module_from_file(str(ptx_path))
        self.batch_kernel = loader.get_function(module, "cosine_similarity_batch")
        self.matrix_kernel = loader.get_function(module, "cosine_similarity_matrix")
        self.norm_kernel = loader.get_function(module, "compute_norm")

    def _compute_norm(self, vec: Sequence[float]) -> float:
        n = len(vec)
        if n == 0:
            return 0.0
        buf = (ctypes.c_float * n)(*vec)
        d_vec = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        d_norm = loader.gpu_malloc(ctypes.sizeof(ctypes.c_float))
        try:
            loader.memcpy_htod(d_vec, ctypes.cast(buf, ctypes.c_void_p), n * ctypes.sizeof(ctypes.c_float))
            block = (256, 1, 1)
            shared = block[0] * ctypes.sizeof(ctypes.c_float)
            loader.launch(
                self.norm_kernel,
                grid=(1, 1, 1),
                block=block,
                params=[ctypes.c_uint64(d_vec.value), ctypes.c_uint64(d_norm.value), ctypes.c_int(n)],
                shared_mem=shared,
            )
            loader.synchronize()
            out = (ctypes.c_float * 1)()
            loader.memcpy_dtoh(ctypes.cast(out, ctypes.c_void_p), d_norm, ctypes.sizeof(ctypes.c_float))
            return float(out[0])
        finally:
            loader.gpu_free(d_vec)
            loader.gpu_free(d_norm)

    def compute_similarities(self, candidates: List[List[float]], expected: List[float]) -> List[float]:
        n = len(candidates)
        if n == 0:
            return []
        d = len(expected)
        flat: List[float] = []
        for cand in candidates:
            flat.extend(float(v) for v in cand)

        cand_buf = (ctypes.c_float * (n * d))(*flat)
        exp_buf = (ctypes.c_float * d)(*expected)
        d_cand = loader.gpu_malloc(n * d * ctypes.sizeof(ctypes.c_float))
        d_exp = loader.gpu_malloc(d * ctypes.sizeof(ctypes.c_float))
        d_scores = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))

        try:
            loader.memcpy_htod(d_cand, ctypes.cast(cand_buf, ctypes.c_void_p), n * d * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_htod(d_exp, ctypes.cast(exp_buf, ctypes.c_void_p), d * ctypes.sizeof(ctypes.c_float))

            norm_expected = self._compute_norm(expected)
            if norm_expected <= 0.0:
                loader.memcpy_dtoh  # no-op marker to satisfy linter
                return [0.0 for _ in range(n)]

            block = (256, 1, 1)
            grid = ((n + block[0] - 1) // block[0], 1, 1)
            loader.launch(
                self.batch_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_cand.value),
                    ctypes.c_uint64(d_exp.value),
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_int(n),
                    ctypes.c_int(d),
                ],
            )
            loader.synchronize()

            scores_buf = (ctypes.c_float * n)()
            loader.memcpy_dtoh(ctypes.cast(scores_buf, ctypes.c_void_p), d_scores, n * ctypes.sizeof(ctypes.c_float))
            norm_expected_inv = 1.0 / norm_expected
            return [float(s * norm_expected_inv) for s in scores_buf]
        finally:
            loader.gpu_free(d_cand)
            loader.gpu_free(d_exp)
            loader.gpu_free(d_scores)

    def compute_similarity_matrix(self, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
        src = np.ascontiguousarray(np.asarray(sources, dtype=np.float32))
        tgt = np.ascontiguousarray(np.asarray(targets, dtype=np.float32))
        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError(
                f"expected 2D sources/targets, got {src.shape=} {tgt.shape=}"
            )
        if src.shape[1] != tgt.shape[1]:
            raise ValueError(
                f"source/target dimension mismatch: {src.shape[1]} != {tgt.shape[1]}"
            )

        n, d = src.shape
        k = tgt.shape[0]
        if n == 0 or k == 0:
            return np.empty((n, k), dtype=np.float32)

        out = np.empty((n, k), dtype=np.float32)
        d_src = loader.gpu_malloc(src.nbytes)
        d_tgt = loader.gpu_malloc(tgt.nbytes)
        d_out = loader.gpu_malloc(out.nbytes)
        try:
            loader.memcpy_htod(d_src, src.ctypes.data_as(ctypes.c_void_p), src.nbytes)
            loader.memcpy_htod(d_tgt, tgt.ctypes.data_as(ctypes.c_void_p), tgt.nbytes)
            block = (256, 1, 1)
            grid = (((n * k) + block[0] - 1) // block[0], 1, 1)
            loader.launch(
                self.matrix_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_src.value),
                    ctypes.c_uint64(d_tgt.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int(n),
                    ctypes.c_int(k),
                    ctypes.c_int(d),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_src)
            loader.gpu_free(d_tgt)
            loader.gpu_free(d_out)

    def compute_similarity_topk(
        self,
        sources: np.ndarray,
        targets: np.ndarray,
        *,
        k: int,
        exclude_self: bool = False,
        similarity_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        src = np.ascontiguousarray(np.asarray(sources, dtype=np.float32))
        tgt = np.ascontiguousarray(np.asarray(targets, dtype=np.float32))
        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError(
                f"expected 2D sources/targets, got {src.shape=} {tgt.shape=}"
            )
        if src.shape[0] == 0 or tgt.shape[0] == 0 or int(k) <= 0:
            shape = (src.shape[0], 0)
            return np.empty(shape, dtype=np.int32), np.empty(shape, dtype=np.float32)

        matrix = self.compute_similarity_matrix(src, tgt)
        if matrix.size == 0:
            shape = (src.shape[0], 0)
            return np.empty(shape, dtype=np.int32), np.empty(shape, dtype=np.float32)

        work = np.asarray(matrix, dtype=np.float32).copy()
        if exclude_self and work.shape[0] == work.shape[1]:
            diag = np.arange(work.shape[0], dtype=np.int32)
            work[diag, diag] = -np.inf

        limit = max(1, min(int(k), work.shape[1]))
        partition = np.argpartition(-work, limit - 1, axis=1)[:, :limit]
        row_ids = np.arange(work.shape[0], dtype=np.int32)[:, None]
        top_scores = work[row_ids, partition]
        order = np.argsort(-top_scores, axis=1)
        ordered_idx = partition[row_ids, order].astype(np.int32, copy=False)
        ordered_scores = top_scores[row_ids, order].astype(np.float32, copy=False)

        if similarity_threshold is not None:
            threshold = float(similarity_threshold)
            ordered_idx = ordered_idx.copy()
            ordered_scores = ordered_scores.copy()
            mask = ~np.isfinite(ordered_scores) | (ordered_scores < threshold)
            ordered_idx[mask] = -1
            ordered_scores[mask] = -np.inf

        return ordered_idx, ordered_scores


__all__ = ["CosineSimilarityBridge"]
