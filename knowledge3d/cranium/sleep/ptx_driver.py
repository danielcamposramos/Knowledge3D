"""Sovereign sleep consolidation driver (pure-ctypes PTX launcher).

Replaces the archived numpy-native ``SleepTimeConsolidation`` /
``GlyphConsolidator`` pair (moved to ``Old_Attempts/2026-04-18/`` by the
Absolute Sovereignty Purge) with a ctypes launcher over three PTX kernels:

    knowledge3d/cranium/ptx/sleep_time_micro.ptx
        ``sleep_time_micro(in, n, out, stride)``
        — micro-consolidation tick over the working ring buffer

    knowledge3d/cranium/ptx/sleep_cluster_refiner.ptx
        ``assign_to_best_centroid(embeddings, assignments, n, k)``
        — reassign embeddings to their nearest centroid (one kernel
        from a five-kernel toolkit; the scheduler calls the others on
        demand via :meth:`refine_clusters_extras`)

    knowledge3d/cranium/ptx/sleep_glyph_consolidator.ptx
        ``cluster_glyphs_by_similarity(glyphs, cluster_ids, n, d, thr)``
        — merge near-duplicate glyphs into shared cluster ids

All I/O travels through ``ctypes`` buffers. No numpy, no cupy, no scipy,
no sympy, no torch, no fallbacks. Per
``feedback_no_fallbacks_ever_including_sleeptime.md`` — if the GPU is
unavailable, the loader raises and we let it propagate.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from knowledge3d.cranium.sovereign import loader


_PTX_DIR = Path(__file__).resolve().parent.parent / "ptx"
_DEFAULT_BLOCK = 256


def _flatten_floats(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def _staged_floats(values: Sequence[float]):
    n = len(values)
    buf = (ctypes.c_float * n)(*[float(v) for v in values])
    dptr = loader.gpu_malloc(ctypes.sizeof(buf))
    loader.memcpy_htod(dptr, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf))
    return buf, dptr, n


class SleepConsolidationDriver:
    """Sovereign PTX driver for sleep-time consolidation.

    Lazily loads each PTX module on first use; the entire module is cached
    for the process lifetime.
    """

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._kernels: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Module / kernel lazy-loading
    # ------------------------------------------------------------------ #
    def _get_kernel(self, module_name: str, entry: str):
        key = f"{module_name}:{entry}"
        cached = self._kernels.get(key)
        if cached is not None:
            return cached
        module = self._modules.get(module_name)
        if module is None:
            ptx_path = _PTX_DIR / f"{module_name}.ptx"
            if not ptx_path.exists():
                raise FileNotFoundError(
                    f"sleep PTX module not found at {ptx_path} — "
                    "sovereign sleep driver cannot dispatch."
                )
            module = loader.load_module_from_file(str(ptx_path))
            self._modules[module_name] = module
        fn = loader.get_function(module, entry)
        self._kernels[key] = fn
        return fn

    # ------------------------------------------------------------------ #
    # Kernel launchers
    # ------------------------------------------------------------------ #
    def sleep_time_micro(
        self, ring_buffer: Sequence[float], *, stride: int = 1
    ) -> List[float]:
        """Run the micro-consolidation tick over ``ring_buffer``.

        ``stride`` is passed as param_3 (u32) — it is an opaque hint
        consumed by the kernel (e.g., strided merge factor).
        """
        if not ring_buffer:
            return []
        _, d_in, n = _staged_floats(ring_buffer)
        d_out = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
        try:
            kernel = self._get_kernel("sleep_time_micro", "sleep_time_micro")
            block = (_DEFAULT_BLOCK, 1, 1)
            grid = ((n + _DEFAULT_BLOCK - 1) // _DEFAULT_BLOCK, 1, 1)
            loader.launch(
                kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint(n),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_uint(int(stride)),
                ],
            )
            loader.synchronize()
            out_buf = (ctypes.c_float * n)()
            loader.memcpy_dtoh(
                ctypes.cast(out_buf, ctypes.c_void_p),
                d_out,
                ctypes.sizeof(out_buf),
            )
            return [float(out_buf[i]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def cluster_glyphs(
        self,
        glyph_embeddings: Sequence[Sequence[float]],
        *,
        similarity_threshold: float = 0.85,
    ) -> List[int]:
        """Cluster glyphs by cosine similarity; return ``cluster_id`` per glyph."""
        n = len(glyph_embeddings)
        if n == 0:
            return []
        flat: List[float] = []
        d_dim: Optional[int] = None
        for i, emb in enumerate(glyph_embeddings):
            row = _flatten_floats(emb)
            if d_dim is None:
                d_dim = len(row)
            elif len(row) != d_dim:
                raise ValueError(
                    f"cluster_glyphs: row {i} dim {len(row)} != expected {d_dim}"
                )
            flat.extend(row)
        dim = int(d_dim or 0)
        if dim == 0:
            return [0 for _ in range(n)]

        FlatType = ctypes.c_float * (n * dim)
        h_in = FlatType(*flat)
        d_in = loader.gpu_malloc(ctypes.sizeof(h_in))
        d_ids = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int32))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), ctypes.sizeof(h_in))
            kernel = self._get_kernel(
                "sleep_glyph_consolidator", "cluster_glyphs_by_similarity"
            )
            block = (_DEFAULT_BLOCK, 1, 1)
            grid = ((n + _DEFAULT_BLOCK - 1) // _DEFAULT_BLOCK, 1, 1)
            loader.launch(
                kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_ids.value),
                    ctypes.c_uint(n),
                    ctypes.c_uint(dim),
                    ctypes.c_float(float(similarity_threshold)),
                ],
            )
            loader.synchronize()
            ids_buf = (ctypes.c_int32 * n)()
            loader.memcpy_dtoh(
                ctypes.cast(ids_buf, ctypes.c_void_p),
                d_ids,
                ctypes.sizeof(ids_buf),
            )
            return [int(ids_buf[i]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_ids)

    def assign_to_best_centroid(
        self,
        embeddings: Sequence[Sequence[float]],
        n_centroids: int,
    ) -> List[int]:
        """Reassign embeddings to their nearest centroid index [0..k)."""
        n = len(embeddings)
        if n == 0 or n_centroids <= 0:
            return []
        flat: List[float] = []
        for emb in embeddings:
            flat.extend(_flatten_floats(emb))

        FlatType = ctypes.c_float * len(flat)
        h_in = FlatType(*flat)
        d_in = loader.gpu_malloc(ctypes.sizeof(h_in))
        d_ids = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_int32))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(h_in, ctypes.c_void_p), ctypes.sizeof(h_in))
            kernel = self._get_kernel(
                "sleep_cluster_refiner", "assign_to_best_centroid"
            )
            block = (_DEFAULT_BLOCK, 1, 1)
            grid = ((n + _DEFAULT_BLOCK - 1) // _DEFAULT_BLOCK, 1, 1)
            loader.launch(
                kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_in.value),
                    ctypes.c_uint64(d_ids.value),
                    ctypes.c_uint(n),
                    ctypes.c_uint(int(n_centroids)),
                ],
            )
            loader.synchronize()
            ids_buf = (ctypes.c_int32 * n)()
            loader.memcpy_dtoh(
                ctypes.cast(ids_buf, ctypes.c_void_p),
                d_ids,
                ctypes.sizeof(ids_buf),
            )
            return [int(ids_buf[i]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_ids)

    # ------------------------------------------------------------------ #
    # High-level orchestration
    # ------------------------------------------------------------------ #
    def consolidate(
        self,
        *,
        ring_buffer: Optional[Sequence[float]] = None,
        glyph_embeddings: Optional[Sequence[Sequence[float]]] = None,
        glyph_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """One sovereign consolidation tick.

        Runs as many of the three PTX kernels as have data available.
        Returns a metrics dict with keys compatible with
        ``SleepScheduler._save_metrics`` (``rpn`` + ``glyph`` sub-dicts).
        """
        t0 = time.perf_counter()

        rpn_metrics: Dict[str, Any] = {"processed": 0}
        if ring_buffer is not None and len(ring_buffer) > 0:
            consolidated = self.sleep_time_micro(ring_buffer)
            rpn_metrics = {
                "processed": len(consolidated),
                "delta_norm": sum((float(a) - float(b)) ** 2
                                  for a, b in zip(consolidated, ring_buffer)) ** 0.5,
            }

        glyph_metrics: Dict[str, Any] = {"glyphs": 0, "clusters": 0}
        if glyph_embeddings:
            cluster_ids = self.cluster_glyphs(
                glyph_embeddings, similarity_threshold=glyph_threshold
            )
            glyph_metrics = {
                "glyphs": len(cluster_ids),
                "clusters": len(set(cluster_ids)),
                "threshold": float(glyph_threshold),
            }

        elapsed = time.perf_counter() - t0
        return {
            "rpn": rpn_metrics,
            "glyph": glyph_metrics,
            "elapsed_seconds": float(elapsed),
        }


__all__ = ["SleepConsolidationDriver"]
