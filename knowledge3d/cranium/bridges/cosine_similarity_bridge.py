from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence

from knowledge3d.cranium.sovereign import loader


class CosineSimilarityBridge:
    """GPU batch cosine similarity via sovereign PTX."""

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "cosine_similarity.ptx"
        module = loader.load_module_from_file(str(ptx_path))
        self.batch_kernel = loader.get_function(module, "cosine_similarity_batch")
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


__all__ = ["CosineSimilarityBridge"]
