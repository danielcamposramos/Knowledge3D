"""Canonical PTX-backed glyph similarity clustering kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

SLEEP_GLYPH_PTX = Path(__file__).parent.parent / "ptx" / "sleep_glyph_consolidator.ptx"

_MODULE = None
_FUNCTIONS: dict[str, loader.CUfunction] = {}


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = loader.load_module_from_file(str(SLEEP_GLYPH_PTX))
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


class SleepGlyphKernels:
    """PTX wrappers for greedy glyph similarity clustering."""

    def __init__(self) -> None:
        _get_module()

    def cluster_by_similarity(
        self,
        embeddings: np.ndarray,
        similarity_threshold: float,
    ) -> np.ndarray:
        emb = _as_float32_matrix(embeddings)
        n, dim = emb.shape
        if n == 0:
            return np.empty((0,), dtype=np.int32)

        out = np.empty((n,), dtype=np.int32)
        d_out = loader.gpu_malloc(out.nbytes)
        d_emb = loader.gpu_malloc(emb.nbytes)
        try:
            loader.memcpy_htod(d_emb, emb.ctypes.data_as(ctypes.c_void_p), emb.nbytes)
            loader.launch(
                _get_function("cluster_glyphs_by_similarity"),
                grid=((n + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_out,
                    d_emb,
                    ctypes.c_int(int(n)),
                    ctypes.c_int(int(dim)),
                    ctypes.c_float(float(similarity_threshold)),
                ],
            )
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_out)
            loader.gpu_free(d_emb)


__all__ = ["SleepGlyphKernels"]
