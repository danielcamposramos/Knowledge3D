"""Sovereign cosine similarity bridge (pure-ctypes launcher over PTX).

Replaces the numpy-native host helper archived to Old_Attempts/2026-04-18/ by
the Absolute Sovereignty Purge. The launcher dispatches the
``cosine_similarity_batch`` kernel from
``knowledge3d/cranium/ptx/cosine_similarity.ptx`` via the sovereign loader.

Public surface mirrors the previous bridge so legacy callers (rarely-used
grammar-galaxy discovery path) keep working:

    bridge = CosineSimilarityBridge()
    scores = bridge.compute_similarities([candidate_embedding_a, ...], query_embedding)
    # scores: list[float], one per candidate, cosine similarity ∈ [-1, 1]

Embeddings are accepted as ``Iterable[float]`` or any object exposing
``__iter__``; they are staged into contiguous ``ctypes.c_float`` buffers before
the H→D copy. No numpy. No cupy. No fallbacks.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from knowledge3d.cranium.sovereign import loader


_PTX_RELPATH = "ptx/cosine_similarity.ptx"
_KERNEL_ENTRY = "cosine_similarity_batch"
_BLOCK_X = 256


def _flatten_embedding(embedding: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in embedding:
        out.append(float(value))
    return out


class CosineSimilarityBridge:
    """PTX-backed batch cosine similarity (1 query × N candidates)."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).resolve().parent.parent / _PTX_RELPATH
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"cosine_similarity.ptx not found at {ptx_path} — "
                "sovereign cosine bridge cannot be constructed."
            )
        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernel = loader.get_function(self._module, _KERNEL_ENTRY)

    def compute_similarities(
        self,
        candidate_embeddings: Sequence[Iterable[float]],
        query_embedding: Iterable[float],
    ) -> List[float]:
        """Return cosine(candidate_i, query) for each candidate.

        ``candidate_embeddings``: sequence of N embeddings, each length D.
        ``query_embedding``: single embedding, length D.

        Returns a list of N floats; empty list if either input is empty.
        """
        query_flat = _flatten_embedding(query_embedding)
        d = len(query_flat)
        if d == 0:
            return []
        n = len(candidate_embeddings)
        if n == 0:
            return []

        # Contiguous row-major candidate matrix (N, D) as ctypes float buffer.
        total = n * d
        candidates_buf = (ctypes.c_float * total)()
        for i, emb in enumerate(candidate_embeddings):
            row = _flatten_embedding(emb)
            if len(row) != d:
                raise ValueError(
                    f"CosineSimilarityBridge: candidate {i} has dim {len(row)}, "
                    f"expected {d}"
                )
            base = i * d
            for j in range(d):
                candidates_buf[base + j] = row[j]

        query_buf = (ctypes.c_float * d)(*query_flat)
        scores_buf = (ctypes.c_float * n)()

        d_candidates = loader.gpu_malloc(ctypes.sizeof(candidates_buf))
        d_query = loader.gpu_malloc(ctypes.sizeof(query_buf))
        d_scores = loader.gpu_malloc(ctypes.sizeof(scores_buf))
        try:
            loader.memcpy_htod(
                d_candidates,
                ctypes.cast(candidates_buf, ctypes.c_void_p),
                ctypes.sizeof(candidates_buf),
            )
            loader.memcpy_htod(
                d_query,
                ctypes.cast(query_buf, ctypes.c_void_p),
                ctypes.sizeof(query_buf),
            )

            grid_x = (n + _BLOCK_X - 1) // _BLOCK_X
            loader.launch(
                self._kernel,
                grid=(grid_x, 1, 1),
                block=(_BLOCK_X, 1, 1),
                params=[
                    ctypes.c_uint64(d_candidates.value),
                    ctypes.c_uint64(d_query.value),
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_uint(n),
                    ctypes.c_uint(d),
                ],
            )
            loader.synchronize()

            loader.memcpy_dtoh(
                ctypes.cast(scores_buf, ctypes.c_void_p),
                d_scores,
                ctypes.sizeof(scores_buf),
            )
        finally:
            loader.gpu_free(d_candidates)
            loader.gpu_free(d_query)
            loader.gpu_free(d_scores)

        return [float(scores_buf[i]) for i in range(n)]


__all__ = ["CosineSimilarityBridge"]
