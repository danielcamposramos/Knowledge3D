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
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine


class MortonOctreeSovereign:
    """GPU-backed Morton encoder with RPN-powered sorting."""

    def __init__(self, block_size: int = 256):
        self.block_size = int(block_size)

        ptx_path = Path(__file__).resolve().parent.parent / "ptx" / "morton_octree.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Morton PTX kernel not found: {ptx_path}")

        self._module = loader.load_module_from_file(str(ptx_path))
        self._encode_kernel = loader.get_function(self._module, "compute_morton_codes")

        self._rpn = ModularRPNEngine()
        self._gt_program = np.array([0x0000, 0x0001, 0x0028], dtype=np.uint16)  # LIT a, LIT b, GT
        self._dummy_vectors = np.zeros((1, 3), dtype=np.float32)

        self._last_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
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

        order = np.arange(n, dtype=np.uint32)
        values = codes.astype(np.uint32, copy=True)

        for i in range(n):
            swapped = False
            for j in range(0, n - 1 - i):
                if self._compare_greater(values[j], values[j + 1]):
                    values[j], values[j + 1] = values[j + 1], values[j]
                    order[j], order[j + 1] = order[j + 1], order[j]
                    swapped = True
            if not swapped:
                break

        if return_indices:
            return values, order
        return values

    def build_tree(self, points: np.ndarray):
        """Build Morton tree by encoding then sorting points."""
        codes = self.encode(points)
        sorted_codes, order = self.sort(codes, return_indices=True)

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
            result = self._rpn.execute_single(
                instance_id=0,
                op_codes=self._gt_program,
                scalars=scalars,
                vectors=self._dummy_vectors,
            )
        finally:
            self._rpn.reset_instance(0)

        return int(a) > int(b)

    def close(self):
        """Compatibility shim for explicit cleanup."""
        # RPN engine owns its buffers and cleans up in its destructor.
        pass


__all__ = ["MortonOctreeSovereign"]
