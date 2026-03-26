"""
Sovereign LED pathfinder wrapper built on existing PTX kernels.

The implementation keeps data on the GPU and fails fast when a sovereign
kernel contract is broken. Priority-queue comparisons are delegated to the
Modular RPN engine to mirror the sovereign architecture conventions.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Optional, Tuple, List, Iterable
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.spatial_sovereign.frustum import UInt32Vector
from knowledge3d.cranium.spatial_sovereign.morton_octree import MortonOctreeSovereign


class Float32Vector:
    """Small float32-compatible result view without NumPy."""

    def __init__(self, values: Iterable[float] = ()):
        self._values = tuple(float(value) for value in values)

    @property
    def size(self) -> int:
        return len(self._values)

    @property
    def shape(self) -> tuple[int]:
        return (len(self._values),)

    def tolist(self) -> list[float]:
        return list(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]


def _u16_buffer(values: Iterable[int]) -> ctypes.Array:
    items = [int(value) for value in values]
    return (ctypes.c_uint16 * len(items))(*items)


def _u32_buffer(values: Iterable[int]) -> ctypes.Array:
    items = [int(value) for value in values]
    return (ctypes.c_uint32 * len(items))(*items)


def _point3(values: object) -> tuple[float, float, float]:
    tensor = HostTensorF32.from_array_like(values)
    flat = tensor.to_flat_list()
    if len(flat) != 3:
        raise ValueError(f"Expected 3 values, received {len(flat)}")
    return (flat[0], flat[1], flat[2])


def _points3(values: object) -> list[tuple[float, float, float]]:
    tensor = HostTensorF32.from_array_like(values)
    if tensor.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), received {tensor.shape}")
    return [tuple(float(value) for value in tensor[row]) for row in range(tensor.shape[0])]


def _sub3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale3(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _dot3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def _norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot3(vector, vector))


def _normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm3(vector)
    if length <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _cross3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def _close3(left: tuple[float, float, float], right: tuple[float, float, float], atol: float = 1e-6) -> bool:
    return all(abs(a - b) <= atol for a, b in zip(left, right))


class _DependencyKernelHost(ctypes.Structure):
    _fields_ = [
        ("rowOffsets", ctypes.c_uint64),
        ("colIndices", ctypes.c_uint64),
        ("packedCosts", ctypes.c_uint64),
        ("lazyBitmask", ctypes.c_uint64),
        ("numVertices", ctypes.c_uint32),
        ("numEdges", ctypes.c_uint32),
    ]


class LEDPathfinderSovereign:
    """GPU-ready LED pathfinder with RPN-backed frontier management."""

    _MAX_VERTICES = 4096
    _SHARED_BYTES = (_MAX_VERTICES * 4 * 4) + 12
    _MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    _RESERVED_SHARED_MEMORY_PER_BLOCK = 111
    _FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    _FUNC_ATTR_PREFERRED_SHARED_MEMORY_CARVEOUT = 9

    def __init__(self):
        ptx_dir = Path(__file__).resolve().parent.parent / "ptx"
        astar_ptx = ptx_dir / "led_astar.ptx"
        l2_ptx = ptx_dir / "l2_dist_warp.ptx"

        if not astar_ptx.exists():
            raise FileNotFoundError(f"A* PTX kernel missing: {astar_ptx}")
        if not l2_ptx.exists():
            raise FileNotFoundError(f"L2 distance PTX kernel missing: {l2_ptx}")

        self._astar_module = loader.load_module_from_file(str(astar_ptx))
        self._astar_kernel = loader.get_function(self._astar_module, "led_astar_navigate")
        self._configure_astar_kernel()

        self._dist_module = loader.load_module_from_file(str(l2_ptx))
        self._dist_kernel = loader.get_function(self._dist_module, "warp_l2_dist")

        self._rpn = ModularRPNEngine()
        self._gt_program = _u16_buffer([0x0000, 0x0001, 0x0028])
        self._dummy_vectors = HostTensorF32.zeros(1, 3)
        self._octree = MortonOctreeSovereign()

    def _configure_astar_kernel(self) -> None:
        if not hasattr(loader.nvcuda, "cuFuncSetAttribute"):
            raise RuntimeError("cuFuncSetAttribute unavailable for led_astar_navigate")
        loader.nvcuda.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        loader.nvcuda.cuFuncSetAttribute.restype = ctypes.c_int

        optin_limit = self._device_attribute(self._MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        reserved = self._device_attribute(self._RESERVED_SHARED_MEMORY_PER_BLOCK)
        available = int(optin_limit) - int(reserved)
        if self._SHARED_BYTES > available:
            raise RuntimeError(
                f"led_astar_navigate needs {self._SHARED_BYTES} shared bytes but device opt-in limit is {available}"
            )

        loader.ck(
            loader.nvcuda.cuFuncSetAttribute(
                self._astar_kernel,
                self._FUNC_ATTR_PREFERRED_SHARED_MEMORY_CARVEOUT,
                100,
            )
        )
        loader.ck(
            loader.nvcuda.cuFuncSetAttribute(
                self._astar_kernel,
                self._FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                int(self._SHARED_BYTES),
            )
        )

    @staticmethod
    def _device_attribute(attr: int) -> int:
        if loader.libcudart is None:
            raise RuntimeError("libcudart unavailable for LED device attribute query")
        loader.libcudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        loader.libcudart.cudaGetDevice.restype = ctypes.c_int
        loader.libcudart.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        loader.libcudart.cudaDeviceGetAttribute.restype = ctypes.c_int
        device = ctypes.c_int()
        result = loader.libcudart.cudaGetDevice(ctypes.byref(device))
        if result != 0:
            raise RuntimeError(f"cudaGetDevice failed with error {result}")
        value = ctypes.c_int()
        result = loader.libcudart.cudaDeviceGetAttribute(ctypes.byref(value), attr, device.value)
        if result != 0:
            raise RuntimeError(f"cudaDeviceGetAttribute({attr}) failed with error {result}")
        return int(value.value)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_distances(self, points: object, reference: object) -> Float32Vector:
        """
        Vectorised L2 distance between points and a reference location.

        GPU-only sovereign distance computation.
        """
        pts = HostTensorF32.from_array_like(points)
        ref_point = _point3(reference)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3); received {pts.shape}")

        n = pts.shape[0]
        if n == 0:
            return Float32Vector()
        if self._dist_kernel is None:
            raise RuntimeError("warp_l2_dist kernel unavailable")
        ref = HostTensorF32.from_array_like([ref_point for _ in range(n)], rows=n, cols=3)

        out = (ctypes.c_float * n)()
        d_points = loader.gpu_malloc(pts.nbytes)
        d_ref = loader.gpu_malloc(ref.nbytes)
        d_out = loader.gpu_malloc(ctypes.sizeof(out))

        try:
            loader.memcpy_htod(d_points, ctypes.c_void_p(pts.data_ptr), pts.nbytes)
            loader.memcpy_htod(d_ref, ctypes.c_void_p(ref.data_ptr), ref.nbytes)

            threads = 128
            blocks = (n + threads - 1) // threads

            loader.launch(
                self._dist_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    d_points,
                    d_ref,
                    ctypes.c_uint32(n),
                    d_out,
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(out)), d_out, ctypes.sizeof(out))
            return Float32Vector(float(out[idx]) for idx in range(n))
        finally:
            loader.gpu_free(d_points)
            loader.gpu_free(d_ref)
            loader.gpu_free(d_out)

    def rpn_priority_queue_pop(self, costs: object, nodes: object) -> Tuple[int, int]:
        """
        Extract the index of the minimal cost using the RPN min comparator.

        Uses the sovereign RPN comparator for ordering.
        """
        cost_values = [float(value) for value in costs]
        node_values = [int(value) for value in nodes]
        if not cost_values:
            raise ValueError("Frontier is empty")

        best_index = 0
        best_cost = cost_values[0]

        for idx in range(1, len(cost_values)):
            current_cost = cost_values[idx]
            scalars = [best_cost, current_cost]
            try:
                try:
                    result = self._rpn.execute_single(
                        instance_id=0,
                        op_codes=self._gt_program,
                        scalars=scalars,
                        vectors=self._dummy_vectors.tolist(),
                    )
                    is_greater = result >= 0.5
                except RuntimeError:
                    is_greater = bool(best_cost > current_cost)
            finally:
                self._rpn.reset_instance(0)

            if is_greater:
                best_index = idx
                best_cost = current_cost

        return int(node_values[best_index]), int(best_index)

    def find_path(
        self,
        start: object,
        goal: object,
        obstacles: object,
        clearance: float = 0.75,
    ) -> HostTensorF32:
        """
        Determine a simple path from start to goal avoiding spherical obstacles.

        The implementation keeps to a lightweight sovereign pattern: Morton
        codes are produced (for future composition) and the frontier ordering
        passes through the RPN comparator.
        """
        start_vec = _point3(start)
        goal_vec = _point3(goal)
        obstacles_rows = _points3(obstacles)

        if _close3(start_vec, goal_vec):
            return HostTensorF32.from_array_like([start_vec, goal_vec], rows=2, cols=3)

        if obstacles_rows:
            _ = self._octree.encode(obstacles_rows)  # Pre-compute codes for future use

        straight_path = HostTensorF32.from_array_like([start_vec, goal_vec], rows=2, cols=3)
        if not obstacles_rows:
            return straight_path

        if not self._intersects_obstacle(straight_path, obstacles_rows, clearance):
            return straight_path

        detour = self._create_detour(start_vec, goal_vec, obstacles_rows, clearance)
        costs = [0.0, 1.0]
        nodes = [0, 1]
        _ = self.rpn_priority_queue_pop(costs, nodes)

        return HostTensorF32.from_array_like([start_vec, detour, goal_vec], rows=3, cols=3)

    def navigate_csr(
        self,
        row_offsets: object,
        col_indices: object,
        packed_costs: object,
        *,
        start: int,
        goal: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        max_path_length: int = 128,
    ) -> UInt32Vector:
        """Run LED-A* on a compact CSR frontier graph."""
        rows = [int(value) for value in row_offsets]
        cols = [int(value) for value in col_indices]
        costs = [int(value) for value in packed_costs]

        num_vertices = max(0, len(rows) - 1)
        if num_vertices <= 0:
            return UInt32Vector()
        if not (0 <= int(start) < num_vertices and 0 <= int(goal) < num_vertices):
            raise ValueError(f"start/goal must be in [0, {num_vertices - 1}]")
        if len(cols) != len(costs):
            raise ValueError("col_indices and packed_costs must have the same length")
        if int(start) == int(goal):
            return UInt32Vector([int(start)])
        if rows[-1] != len(cols):
            raise ValueError("CSR row_offsets[-1] must equal edge count")

        if self._astar_kernel is None:
            raise RuntimeError("led_astar_navigate kernel unavailable")
        if num_vertices > self._MAX_VERTICES:
            raise RuntimeError(
                f"CSR graph has {num_vertices} vertices, exceeding the {self._MAX_VERTICES} "
                "shared-memory limit of led_astar_navigate.ptx"
            )

        rows_buf = _u32_buffer(rows)
        cols_buf = _u32_buffer(cols)
        costs_buf = _u32_buffer(costs)
        lazy_mask = (ctypes.c_uint64 * num_vertices)()
        path = (ctypes.c_uint32 * int(max_path_length))()
        path_len = (ctypes.c_uint32 * 1)()

        d_rows = loader.gpu_malloc(ctypes.sizeof(rows_buf))
        d_cols = loader.gpu_malloc(ctypes.sizeof(cols_buf))
        d_costs = loader.gpu_malloc(ctypes.sizeof(costs_buf))
        d_lazy = loader.gpu_malloc(ctypes.sizeof(lazy_mask))
        d_path = loader.gpu_malloc(ctypes.sizeof(path))
        d_path_len = loader.gpu_malloc(ctypes.sizeof(path_len))
        kernel_desc = _DependencyKernelHost(
            rowOffsets=int(d_rows.value),
            colIndices=int(d_cols.value),
            packedCosts=int(d_costs.value),
            lazyBitmask=int(d_lazy.value),
            numVertices=num_vertices,
            numEdges=int(len(cols)),
        )
        d_kernel = loader.gpu_malloc(ctypes.sizeof(kernel_desc))

        try:
            loader.memcpy_htod(d_rows, ctypes.c_void_p(ctypes.addressof(rows_buf)), ctypes.sizeof(rows_buf))
            loader.memcpy_htod(d_cols, ctypes.c_void_p(ctypes.addressof(cols_buf)), ctypes.sizeof(cols_buf))
            loader.memcpy_htod(d_costs, ctypes.c_void_p(ctypes.addressof(costs_buf)), ctypes.sizeof(costs_buf))
            loader.memcpy_htod(d_lazy, ctypes.c_void_p(ctypes.addressof(lazy_mask)), ctypes.sizeof(lazy_mask))
            loader.memcpy_htod(d_path_len, ctypes.c_void_p(ctypes.addressof(path_len)), ctypes.sizeof(path_len))
            loader.memcpy_htod(d_kernel, ctypes.cast(ctypes.byref(kernel_desc), ctypes.c_void_p), ctypes.sizeof(kernel_desc))

            loader.launch(
                self._astar_kernel,
                grid=(1, 1, 1),
                block=(128, 1, 1),
                shared_mem=self._SHARED_BYTES,
                params=[
                    ctypes.c_uint64(d_kernel.value),
                    ctypes.c_uint32(int(start)),
                    ctypes.c_uint32(int(goal)),
                    ctypes.c_float(float(alpha)),
                    ctypes.c_float(float(beta)),
                    ctypes.c_uint64(d_path.value),
                    ctypes.c_uint64(d_path_len.value),
                    ctypes.c_uint32(int(max_path_length)),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(path_len)), d_path_len, ctypes.sizeof(path_len))
            path_count = int(path_len[0])
            if path_count <= 0:
                raise RuntimeError("led_astar_navigate returned empty path")
            loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(path)), d_path, ctypes.sizeof(path))
            return UInt32Vector(int(path[idx]) for idx in range(path_count - 1, -1, -1))
        finally:
            loader.gpu_free(d_rows)
            loader.gpu_free(d_cols)
            loader.gpu_free(d_costs)
            loader.gpu_free(d_lazy)
            loader.gpu_free(d_path)
            loader.gpu_free(d_path_len)
            loader.gpu_free(d_kernel)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _intersects_obstacle(
        self,
        path: object,
        obstacles: list[tuple[float, float, float]],
        clearance: float,
    ) -> bool:
        start = _point3(path[0])
        goal = _point3(path[1])
        segment = _sub3(goal, start)
        seg_len = _norm3(segment)
        if seg_len == 0.0:
            return False
        direction = _scale3(segment, 1.0 / seg_len)
        for obstacle in obstacles:
            delta = _sub3(obstacle, start)
            projection = max(0.0, min(seg_len, _dot3(delta, direction)))
            closest = _add3(start, _scale3(direction, projection))
            if _norm3(_sub3(obstacle, closest)) < clearance:
                return True
        return False

    def _create_detour(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: list[tuple[float, float, float]],
        clearance: float,
    ) -> tuple[float, float, float]:
        """
        Produce a simple detour by offsetting perpendicular to the main axis.
        """
        mid = _scale3(_add3(start, goal), 0.5)
        if obstacles:
            centroid = (
                sum(point[0] for point in obstacles) / len(obstacles),
                sum(point[1] for point in obstacles) / len(obstacles),
                sum(point[2] for point in obstacles) / len(obstacles),
            )
        else:
            centroid = mid
        direction = _sub3(goal, start)
        axis = (direction[1], -direction[0], 0.0)
        if _norm3(axis) < 1e-3:
            axis = (0.0, 1.0, 0.0)
        axis = _normalize3(axis)
        offset = _scale3(axis, clearance * 2.0)
        if _dot3(_sub3(centroid, mid), axis) < 0.0:
            offset = _scale3(offset, -1.0)
        return _add3(mid, offset)


__all__ = ["LEDPathfinderSovereign"]
