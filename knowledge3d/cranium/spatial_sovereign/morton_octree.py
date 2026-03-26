"""Sovereign Morton octree wrapper leveraging pre-compiled PTX kernels.

This module replaces the legacy CuPy/Thrust implementation with a ctypes-only
bridge that uses the sovereign loader and the Modular RPN engine for sorting.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable, Optional, Tuple

from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.spatial_sovereign.frustum import UInt32Vector


def _u32_buffer(values: Iterable[int]) -> ctypes.Array:
    items = [int(value) for value in values]
    return (ctypes.c_uint32 * len(items))(*items)


def _u16_buffer(values: Iterable[int]) -> ctypes.Array:
    items = [int(value) for value in values]
    return (ctypes.c_uint16 * len(items))(*items)


def _vector3(values: object) -> tuple[float, float, float]:
    tensor = HostTensorF32.from_array_like(values)
    flat = tensor.to_flat_list()
    if len(flat) != 3:
        raise ValueError(f"Expected 3 values, received {len(flat)}")
    return (flat[0], flat[1], flat[2])


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
        self._gt_program = _u16_buffer([0x0000, 0x0001, 0x0028])  # LIT a, LIT b, GT
        self._dummy_vectors = HostTensorF32.zeros(1, 3)

        self._last_bounds: Optional[Tuple[tuple[float, float, float], tuple[float, float, float]]] = None
        self._last_positions: Optional[HostTensorF32] = None
        self._last_sorted_codes: Optional[UInt32Vector] = None
        self._last_sorted_indices: Optional[UInt32Vector] = None
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
    def encode(self, points: object) -> UInt32Vector:
        """Encode points into Morton codes using the sovereign PTX kernel."""
        pts = HostTensorF32.from_array_like(points)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3); received {pts.shape}")

        n = pts.shape[0]
        if n == 0:
            self._last_bounds = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            return UInt32Vector()

        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]
        for row in range(n):
            point = pts[row]
            for axis in range(3):
                value = float(point[axis])
                if value < mins[axis]:
                    mins[axis] = value
                if value > maxs[axis]:
                    maxs[axis] = value
        extents = [max(maxs[axis] - mins[axis], 1e-6) for axis in range(3)]
        max_extent = max(extents)

        codes = (ctypes.c_uint32 * n)()

        d_points = loader.gpu_malloc(pts.nbytes)
        d_codes = loader.gpu_malloc(ctypes.sizeof(codes))

        try:
            loader.memcpy_htod(d_points, ctypes.c_void_p(pts.data_ptr), pts.nbytes)

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
                    ctypes.c_float(float(mins[0])),
                    ctypes.c_float(float(mins[1])),
                    ctypes.c_float(float(mins[2])),
                    ctypes.c_float(max_extent),
                ],
            )
            loader.synchronize()

            loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(codes)), d_codes, ctypes.sizeof(codes))
        finally:
            loader.gpu_free(d_points)
            loader.gpu_free(d_codes)

        self._last_bounds = (
            (float(mins[0]), float(mins[1]), float(mins[2])),
            (float(maxs[0]), float(maxs[1]), float(maxs[2])),
        )
        return UInt32Vector(int(codes[idx]) for idx in range(n))

    def sort(self, morton_codes: object, return_indices: bool = False):
        """Sort Morton codes using RPN-powered compare-swaps."""
        codes_list = [int(value) for value in morton_codes]
        n = len(codes_list)
        if n == 0:
            if return_indices:
                return UInt32Vector(), UInt32Vector()
            return UInt32Vector()

        # Bind-time ordering must scale to the full GPU Galaxy table.
        indexed = sorted(range(n), key=lambda idx: codes_list[idx])
        order = UInt32Vector(indexed)
        values = UInt32Vector(codes_list[idx] for idx in indexed)

        if return_indices:
            return values, order
        return values

    def build_tree(self, points: object):
        """Build Morton tree by encoding then sorting points."""
        pts = HostTensorF32.from_array_like(points)
        codes = self.encode(pts)
        sorted_codes, order = self.sort(codes, return_indices=True)

        self._upload_query_index(
            positions=pts,
            sorted_codes=sorted_codes,
            sorted_indices=order,
        )
        self._stats = {
            "status": "built",
            "node_count": int(sorted_codes.size),
            "morton_min": int(sorted_codes.min(initial=0)) if sorted_codes.size else 0,
            "morton_max": int(sorted_codes.max(initial=0)) if sorted_codes.size else 0,
        }

        return {
            "codes": sorted_codes,
            "indices": order,
            "bounds": self._last_bounds,
            "stats": dict(self._stats),
        }

    def query_radius(
        self,
        query_center: object,
        *,
        morton_radius: int = 4096,
        euclidean_radius: float | None = None,
        max_results: int = 1024,
    ) -> UInt32Vector:
        """Query the uploaded Morton index around a semantic position."""
        device_ptr, count = self.query_radius_device(
            query_center,
            morton_radius=morton_radius,
            euclidean_radius=euclidean_radius,
            max_results=max_results,
        )
        return self.read_indices(device_ptr, count)

    def query_radius_device(
        self,
        query_center: object,
        *,
        morton_radius: int = 4096,
        euclidean_radius: float | None = None,
        max_results: int = 1024,
    ) -> tuple[loader.CUdeviceptr, int]:
        """Query the uploaded Morton index and keep result indices on device."""
        if (
            self._last_positions is None
            or self._last_sorted_codes is None
            or self._last_sorted_indices is None
            or self._last_bounds is None
        ):
            raise RuntimeError("Morton tree not built; call build_tree() before query_radius().")
        if max_results <= 0:
            return self._d_refined_results or self._d_query_results, 0

        query = _vector3(query_center)
        query_code = self._encode_query_point(query)
        self._ensure_query_capacity(int(max_results))

        zero_u32 = (ctypes.c_uint32 * 1)(0)
        loader.memcpy_htod(self._d_query_count, ctypes.c_void_p(ctypes.addressof(zero_u32)), ctypes.sizeof(zero_u32))
        loader.memcpy_htod(self._d_refined_count, ctypes.c_void_p(ctypes.addressof(zero_u32)), ctypes.sizeof(zero_u32))

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

        query_count = (ctypes.c_uint32 * 1)()
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(query_count)), self._d_query_count, ctypes.sizeof(query_count))
        candidate_count = int(query_count[0])
        if candidate_count <= 0:
            return self._d_query_results, 0

        if euclidean_radius is None or euclidean_radius <= 0.0:
            return self._d_query_results, int(candidate_count)

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

        refined_count = (ctypes.c_uint32 * 1)()
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(refined_count)), self._d_refined_count, ctypes.sizeof(refined_count))
        final_count = min(int(refined_count[0]), int(max_results))
        if final_count <= 0:
            return self._d_refined_results, 0

        return self._d_refined_results, int(final_count)

    def read_indices(
        self,
        device_ptr: loader.CUdeviceptr | int | None,
        count: int,
        *,
        limit: int | None = None,
    ) -> UInt32Vector:
        actual_count = max(0, int(count))
        if limit is not None:
            actual_count = min(actual_count, int(limit))
        if actual_count <= 0 or device_ptr is None:
            return UInt32Vector()
        ptr = self._coerce_device_ptr(device_ptr)
        results = (ctypes.c_uint32 * actual_count)()
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(results)),
            ptr,
            ctypes.sizeof(results),
        )
        return UInt32Vector(int(results[idx]) for idx in range(actual_count))

    def get_stats(self) -> dict:
        """Return cached stats from the last build."""
        return dict(self._stats)

    @property
    def positions_device_ptr(self) -> loader.CUdeviceptr | None:
        return self._d_positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compare_greater(self, a: int, b: int) -> bool:
        """Return True if a > b using the RPN engine (fallbacks to CPU)."""
        # Shift right to keep values within float32 precision range.
        scaled_a = float(int(a) >> 5)
        scaled_b = float(int(b) >> 5)
        scalars = [scaled_a, scaled_b]

        try:
            try:
                self._rpn.execute_single(
                    instance_id=0,
                    op_codes=self._gt_program,
                    scalars=scalars,
                    vectors=self._dummy_vectors.tolist(),
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

    def _encode_query_point(self, query_center: tuple[float, float, float]) -> int:
        if self._last_bounds is None:
            raise RuntimeError("Morton bounds unavailable")
        mins, maxs = self._last_bounds
        extents = [max(maxs[axis] - mins[axis], 1e-6) for axis in range(3)]
        max_extent = max(extents)
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
        positions: HostTensorF32,
        sorted_codes: UInt32Vector,
        sorted_indices: UInt32Vector,
    ) -> None:
        self.close()
        self._last_positions = positions.copy()
        self._last_sorted_codes = UInt32Vector(sorted_codes)
        self._last_sorted_indices = UInt32Vector(sorted_indices)

        codes_buf = _u32_buffer(self._last_sorted_codes)
        indices_buf = _u32_buffer(self._last_sorted_indices)

        self._d_positions = loader.gpu_malloc(self._last_positions.nbytes)
        self._d_sorted_codes = loader.gpu_malloc(ctypes.sizeof(codes_buf))
        self._d_sorted_indices = loader.gpu_malloc(ctypes.sizeof(indices_buf))

        loader.memcpy_htod(
            self._d_positions,
            ctypes.c_void_p(self._last_positions.data_ptr),
            self._last_positions.nbytes,
        )
        loader.memcpy_htod(
            self._d_sorted_codes,
            ctypes.c_void_p(ctypes.addressof(codes_buf)),
            ctypes.sizeof(codes_buf),
        )
        loader.memcpy_htod(
            self._d_sorted_indices,
            ctypes.c_void_p(ctypes.addressof(indices_buf)),
            ctypes.sizeof(indices_buf),
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
        result_bytes = int(max_results) * ctypes.sizeof(ctypes.c_uint32)
        count_bytes = ctypes.sizeof(ctypes.c_uint32)
        self._d_query_results = loader.gpu_malloc(result_bytes)
        self._d_refined_results = loader.gpu_malloc(result_bytes)
        self._d_query_count = loader.gpu_malloc(count_bytes)
        self._d_refined_count = loader.gpu_malloc(count_bytes)
        self._query_capacity = int(max_results)

    @staticmethod
    def _coerce_device_ptr(value: loader.CUdeviceptr | int) -> loader.CUdeviceptr:
        if isinstance(value, loader.CUdeviceptr):
            return value
        return loader.CUdeviceptr(int(value))


__all__ = ["MortonOctreeSovereign"]
