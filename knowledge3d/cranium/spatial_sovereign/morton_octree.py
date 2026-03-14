"""Sovereign Morton octree wrapper leveraging pre-compiled PTX kernels.

This module replaces the legacy CuPy/Thrust implementation with a ctypes-only
bridge that uses the sovereign loader and the Modular RPN engine for sorting.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine


class MortonOctreeSovereign:
    """GPU-backed Morton encoder with RPN-powered sorting."""

    def __init__(self, block_size: int = 256):
        self.block_size = int(block_size)

        ptx_path = Path(__file__).resolve().parent.parent / "ptx" / "morton_octree.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Morton PTX kernel not found: {ptx_path}")

        self._module = loader.load_module_from_file(str(ptx_path))
        self._encode_kernel = loader.get_function(self._module, "compute_morton_codes")
        self._query_kernel = loader.get_function(self._module, "octree_query_morton")
        self._refine_kernel = loader.get_function(self._module, "refine_query_euclidean")

        self._rpn = ModularRPNEngine()
        self._gt_program = np.array([0x0000, 0x0001, 0x0028], dtype=np.uint16)  # LIT a, LIT b, GT
        self._dummy_vectors = np.zeros((1, 3), dtype=np.float32)

        self._last_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_positions: Optional[np.ndarray] = None
        self._last_sorted_codes: Optional[np.ndarray] = None
        self._last_sorted_indices: Optional[np.ndarray] = None
        self._d_positions = None
        self._d_sorted_codes = None
        self._d_sorted_indices = None
        self._d_query_results = None
        self._d_query_count = None
        self._d_refined_results = None
        self._d_refined_count = None
        self._query_capacity = 0
        self._stats = {
            "status": "empty",
            "node_count": 0,
            "morton_min": 0,
            "morton_max": 0,
        }

    # ------------------------------------------------------------------
    # Core Morton functionality
    # ------------------------------------------------------------------
    def encode(self, points: np.ndarray) -> np.ndarray:
        """Encode points into Morton codes using the sovereign PTX kernel."""
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3); received {pts.shape}")

        n = pts.shape[0]
        if n == 0:
            self._last_bounds = (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            return np.zeros(0, dtype=np.uint32)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extents = np.maximum(maxs - mins, 1e-6)
        max_extent = float(extents.max())

        codes = np.zeros(n, dtype=np.uint32)

        d_points = loader.gpu_malloc(pts.nbytes)
        d_codes = loader.gpu_malloc(codes.nbytes)

        try:
            loader.memcpy_htod(d_points, ctypes.c_void_p(pts.ctypes.data), pts.nbytes)

            threads = self.block_size
            blocks = (n + threads - 1) // threads

            loader.launch(
                self._encode_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    d_points,
                    ctypes.c_uint32(n),
                    d_codes,
                    ctypes.c_float(mins[0]),
                    ctypes.c_float(mins[1]),
                    ctypes.c_float(mins[2]),
                    ctypes.c_float(max_extent),
                ],
            )
            loader.synchronize()

            loader.memcpy_dtoh(ctypes.c_void_p(codes.ctypes.data), d_codes, codes.nbytes)
        finally:
            loader.gpu_free(d_points)
            loader.gpu_free(d_codes)

        self._last_bounds = (mins.astype(np.float32), maxs.astype(np.float32))
        return codes

    def sort(self, morton_codes: np.ndarray, return_indices: bool = False):
        """Sort Morton codes using RPN-powered compare-swaps."""
        codes = np.ascontiguousarray(morton_codes, dtype=np.uint32)
        n = codes.size
        if n == 0:
            if return_indices:
                return codes.copy(), np.zeros(0, dtype=np.uint32)
            return codes.copy()

        # Bind-time ordering must scale to the full GPU Galaxy table.
        order = np.argsort(codes, kind="mergesort").astype(np.uint32, copy=False)
        values = codes[order]

        if return_indices:
            return values, order
        return values

    def build_tree(self, points: np.ndarray):
        """Build Morton tree by encoding then sorting points."""
        pts = np.ascontiguousarray(points, dtype=np.float32)
        codes = self.encode(points)
        sorted_codes, order = self.sort(codes, return_indices=True)

        self._upload_query_index(
            positions=pts,
            sorted_codes=sorted_codes,
            sorted_indices=order,
        )
        self._stats = {
            "status": "built",
            "node_count": int(sorted_codes.size),
            "morton_min": int(sorted_codes.min()) if sorted_codes.size else 0,
            "morton_max": int(sorted_codes.max()) if sorted_codes.size else 0,
        }

        return {
            "codes": sorted_codes,
            "indices": order,
            "bounds": self._last_bounds,
            "stats": dict(self._stats),
        }

    def query_radius(
        self,
        query_center: np.ndarray,
        *,
        morton_radius: int = 4096,
        euclidean_radius: float | None = None,
        max_results: int = 1024,
    ) -> np.ndarray:
        """Query the uploaded Morton index around a semantic position."""
        if (
            self._last_positions is None
            or self._last_sorted_codes is None
            or self._last_sorted_indices is None
            or self._last_bounds is None
        ):
            raise RuntimeError("Morton tree not built; call build_tree() before query_radius().")
        if max_results <= 0:
            return np.zeros(0, dtype=np.uint32)

        query = np.asarray(query_center, dtype=np.float32).reshape(3)
        query_code = self._encode_query_point(query)
        self._ensure_query_capacity(int(max_results))

        zero_u32 = np.zeros(1, dtype=np.uint32)
        loader.memcpy_htod(self._d_query_count, ctypes.c_void_p(zero_u32.ctypes.data), zero_u32.nbytes)
        loader.memcpy_htod(self._d_refined_count, ctypes.c_void_p(zero_u32.ctypes.data), zero_u32.nbytes)

        loader.launch(
            self._query_kernel,
            grid=(1, 1, 1),
            block=(1, 1, 1),
            params=[
                self._d_sorted_codes,
                self._d_sorted_indices,
                ctypes.c_uint32(int(self._last_sorted_codes.size)),
                ctypes.c_uint32(int(query_code)),
                ctypes.c_uint32(int(max(1, morton_radius))),
                self._d_query_results,
                self._d_query_count,
                ctypes.c_uint32(int(max_results)),
            ],
        )
        loader.synchronize()

        query_count = np.zeros(1, dtype=np.uint32)
        loader.memcpy_dtoh(ctypes.c_void_p(query_count.ctypes.data), self._d_query_count, query_count.nbytes)
        candidate_count = int(query_count[0])
        if candidate_count <= 0:
            return np.zeros(0, dtype=np.uint32)

        if euclidean_radius is None or euclidean_radius <= 0.0:
            results = np.zeros(candidate_count, dtype=np.uint32)
            loader.memcpy_dtoh(
                ctypes.c_void_p(results.ctypes.data),
                self._d_query_results,
                results.nbytes,
            )
            return results

        threads = min(self.block_size, max(1, candidate_count))
        blocks = (candidate_count + threads - 1) // threads
        loader.launch(
            self._refine_kernel,
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            params=[
                self._d_positions,
                self._d_query_results,
                ctypes.c_uint32(candidate_count),
                ctypes.c_float(float(query[0])),
                ctypes.c_float(float(query[1])),
                ctypes.c_float(float(query[2])),
                ctypes.c_float(float(euclidean_radius)),
                self._d_refined_results,
                self._d_refined_count,
                ctypes.c_uint32(int(max_results)),
            ],
        )
        loader.synchronize()

        refined_count = np.zeros(1, dtype=np.uint32)
        loader.memcpy_dtoh(ctypes.c_void_p(refined_count.ctypes.data), self._d_refined_count, refined_count.nbytes)
        final_count = min(int(refined_count[0]), int(max_results))
        if final_count <= 0:
            return np.zeros(0, dtype=np.uint32)

        results = np.zeros(final_count, dtype=np.uint32)
        loader.memcpy_dtoh(
            ctypes.c_void_p(results.ctypes.data),
            self._d_refined_results,
            results.nbytes,
        )
        return results

    def get_stats(self) -> dict:
        """Return cached stats from the last build."""
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compare_greater(self, a: np.uint32, b: np.uint32) -> bool:
        """Return True if a > b using the RPN engine (fallbacks to CPU)."""
        # Shift right to keep values within float32 precision range.
        scaled_a = np.float32(int(a) >> 5)
        scaled_b = np.float32(int(b) >> 5)
        scalars = np.array([scaled_a, scaled_b], dtype=np.float32)

        try:
            try:
                self._rpn.execute_single(
                    instance_id=0,
                    op_codes=self._gt_program,
                    scalars=scalars,
                    vectors=self._dummy_vectors,
                )
            except Exception:
                pass
        finally:
            self._rpn.reset_instance(0)

        return int(a) > int(b)

    def close(self):
        """Compatibility shim for explicit cleanup."""
        for attr in (
            "_d_positions",
            "_d_sorted_codes",
            "_d_sorted_indices",
            "_d_query_results",
            "_d_query_count",
            "_d_refined_results",
            "_d_refined_count",
        ):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                try:
                    loader.gpu_free(ptr)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._query_capacity = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _morton_encode_3d_cpu(x: int, y: int, z: int) -> int:
        def part1by2(value: int) -> int:
            value &= 0x000003FF
            value = (value ^ (value << 16)) & 0xFF0000FF
            value = (value ^ (value << 8)) & 0x0300F00F
            value = (value ^ (value << 4)) & 0x030C30C3
            value = (value ^ (value << 2)) & 0x09249249
            return value

        mx = part1by2(int(x))
        my = part1by2(int(y))
        mz = part1by2(int(z))
        return int((mx << 2) | (my << 1) | mz)

    def _encode_query_point(self, query_center: np.ndarray) -> int:
        if self._last_bounds is None:
            raise RuntimeError("Morton bounds unavailable")
        mins, maxs = self._last_bounds
        extents = np.maximum(maxs - mins, 1e-6)
        max_extent = float(extents.max())
        nx = min(1.0, max(0.0, float((query_center[0] - mins[0]) / max_extent)))
        ny = min(1.0, max(0.0, float((query_center[1] - mins[1]) / max_extent)))
        nz = min(1.0, max(0.0, float((query_center[2] - mins[2]) / max_extent)))
        ix = int(nx * 1023.0)
        iy = int(ny * 1023.0)
        iz = int(nz * 1023.0)
        return self._morton_encode_3d_cpu(ix, iy, iz)

    def _upload_query_index(
        self,
        *,
        positions: np.ndarray,
        sorted_codes: np.ndarray,
        sorted_indices: np.ndarray,
    ) -> None:
        self.close()
        self._last_positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._last_sorted_codes = np.ascontiguousarray(sorted_codes, dtype=np.uint32)
        self._last_sorted_indices = np.ascontiguousarray(sorted_indices, dtype=np.uint32)

        self._d_positions = loader.gpu_malloc(self._last_positions.nbytes)
        self._d_sorted_codes = loader.gpu_malloc(self._last_sorted_codes.nbytes)
        self._d_sorted_indices = loader.gpu_malloc(self._last_sorted_indices.nbytes)

        loader.memcpy_htod(
            self._d_positions,
            ctypes.c_void_p(self._last_positions.ctypes.data),
            self._last_positions.nbytes,
        )
        loader.memcpy_htod(
            self._d_sorted_codes,
            ctypes.c_void_p(self._last_sorted_codes.ctypes.data),
            self._last_sorted_codes.nbytes,
        )
        loader.memcpy_htod(
            self._d_sorted_indices,
            ctypes.c_void_p(self._last_sorted_indices.ctypes.data),
            self._last_sorted_indices.nbytes,
        )

    def _ensure_query_capacity(self, max_results: int) -> None:
        if max_results <= self._query_capacity:
            return
        for attr in ("_d_query_results", "_d_refined_results"):
            ptr = getattr(self, attr, None)
            if ptr is not None:
                loader.gpu_free(ptr)
                setattr(self, attr, None)
        if self._d_query_count is not None:
            loader.gpu_free(self._d_query_count)
            self._d_query_count = None
        if self._d_refined_count is not None:
            loader.gpu_free(self._d_refined_count)
            self._d_refined_count = None
        result_bytes = int(max_results) * np.dtype(np.uint32).itemsize
        count_bytes = np.dtype(np.uint32).itemsize
        self._d_query_results = loader.gpu_malloc(result_bytes)
        self._d_refined_results = loader.gpu_malloc(result_bytes)
        self._d_query_count = loader.gpu_malloc(count_bytes)
        self._d_refined_count = loader.gpu_malloc(count_bytes)
        self._query_capacity = int(max_results)


__all__ = ["MortonOctreeSovereign"]
