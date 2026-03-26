"""
Sovereign frustum culling wrapper built on pre-compiled PTX kernels.

This module replaces the legacy CuPy-based implementation with a thin Python
layer that talks directly to the CUDA Driver API through the sovereign loader.
"""
from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path
from typing import Iterable, Optional

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.sovereign import loader


class UInt32Vector:
    """Small uint32 result view without NumPy dependency."""

    def __init__(self, values: Iterable[int] = ()):
        self._values = tuple(int(value) for value in values)

    @property
    def size(self) -> int:
        return len(self._values)

    @property
    def shape(self) -> tuple[int]:
        return (len(self._values),)

    def tolist(self) -> list[int]:
        return list(self._values)

    def min(self, initial: int | None = None) -> int:
        if self._values:
            current = min(self._values)
            return current if initial is None else min(current, int(initial))
        if initial is None:
            raise ValueError("min() arg is an empty sequence")
        return int(initial)

    def max(self, initial: int | None = None) -> int:
        if self._values:
            current = max(self._values)
            return current if initial is None else max(current, int(initial))
        if initial is None:
            raise ValueError("max() arg is an empty sequence")
        return int(initial)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]


def _float_list(values: object) -> list[float]:
    rows, cols = HostTensorF32.from_array_like(values).shape
    tensor = HostTensorF32.from_array_like(values, rows=rows, cols=cols)
    return tensor.to_flat_list()


def _matrix4(values: object) -> HostTensorF32:
    matrix = HostTensorF32.from_array_like(values, rows=4, cols=4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4, received {matrix.shape}")
    return matrix


def _vector3(values: object) -> tuple[float, float, float]:
    flat = _float_list(values)
    if len(flat) != 3:
        raise ValueError(f"Vector must contain exactly 3 elements, received {len(flat)}")
    return (flat[0], flat[1], flat[2])


def _cross3(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def _dot3(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm3(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot3(vector, vector))


def _normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm3(vector)
    if length <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def matmul_4x4(left: object, right: object) -> HostTensorF32:
    a = _matrix4(left)
    b = _matrix4(right)
    out = HostTensorF32.zeros(4, 4)
    result: list[float] = []
    for row in range(4):
        for col in range(4):
            accum = 0.0
            for inner in range(4):
                accum += float(a[row, inner]) * float(b[inner, col])
            result.append(float(accum))
    out.set_flat(result)
    return out


def matvec_4(matrix: object, vector: Iterable[float]) -> list[float]:
    mat = _matrix4(matrix)
    values = [float(item) for item in vector]
    if len(values) != 4:
        raise ValueError(f"Expected 4-vector, received {len(values)} values")
    out: list[float] = []
    for row in range(4):
        accum = 0.0
        for col in range(4):
            accum += float(mat[row, col]) * values[col]
        out.append(float(accum))
    return out


class FrustumCuller:
    _MODULE_HANDLE: Optional[loader.CUmodule] = None
    _KERNEL_HANDLE: Optional[loader.CUfunction] = None
    _VIEW_PROJ_PTR: Optional[loader.CUdeviceptr] = None
    _VIEW_MATRIX_PTR: Optional[loader.CUdeviceptr] = None

    """GPU frustum culling using Kimi's warp-level SIMD PTX kernel."""

    def __init__(self, block_size: int = 128, enable_profiling: bool = False):
        self.block_size = int(block_size)
        self.enable_profiling = enable_profiling

        ptx_path = Path(__file__).resolve().parent.parent / "ptx" / "frustum_cull_simd.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Frustum PTX not found at {ptx_path}")

        if FrustumCuller._MODULE_HANDLE is None:
            module = loader.load_module_from_file(str(ptx_path))
            kernel = loader.get_function(module, "warp_frustum_cull_simd")
            view_proj_ptr, view_proj_bytes = loader.get_global(module, "view_proj")
            view_matrix_ptr, view_bytes = loader.get_global(module, "view_matrix")
            if view_proj_bytes < 64 or view_bytes < 64:
                raise RuntimeError("Unexpected constant buffer size for frustum kernel")

            FrustumCuller._MODULE_HANDLE = module
            FrustumCuller._KERNEL_HANDLE = kernel
            FrustumCuller._VIEW_PROJ_PTR = view_proj_ptr
            FrustumCuller._VIEW_MATRIX_PTR = view_matrix_ptr

        self._module = FrustumCuller._MODULE_HANDLE
        self._kernel = FrustumCuller._KERNEL_HANDLE
        self._view_proj_ptr = FrustumCuller._VIEW_PROJ_PTR
        self._view_matrix_ptr = FrustumCuller._VIEW_MATRIX_PTR

        self._flags_ptr: Optional[loader.CUdeviceptr] = None
        self._flags_capacity = 0  # bytes
        self._zero_template = (ctypes.c_uint8 * 0)()
        self._view_proj_uploaded = False

        # Statistics
        self.total_culls = 0
        self.total_input_nodes = 0
        self.total_output_nodes = 0
        self.total_time_ms = 0.0
        self._last_elapsed_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # Resource management helpers
    # ------------------------------------------------------------------
    def _ensure_flag_capacity(self, required_count: int) -> None:
        required_bytes = int(required_count)
        if required_bytes <= self._flags_capacity and self._flags_ptr is not None:
            return

        if self._flags_ptr is not None:
            loader.gpu_free(self._flags_ptr)

        padded = int(required_bytes * 1.2) + 64
        self._flags_ptr = loader.gpu_malloc(padded)
        self._flags_capacity = padded
        self._zero_template = (ctypes.c_uint8 * self._flags_capacity)()

    def close(self):
        """Release device resources."""
        if self._flags_ptr is not None:
            loader.gpu_free(self._flags_ptr)
            self._flags_ptr = None
            self._flags_capacity = 0
            self._zero_template = (ctypes.c_uint8 * 0)()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors should never raise.
            pass

    # ------------------------------------------------------------------
    # Constant memory uploads
    # ------------------------------------------------------------------
    def upload_view_projection(self, view_proj: object, view: Optional[object] = None) -> None:
        """Upload view-projection and view matrices into constant memory."""
        vp = _matrix4(view_proj)
        view_matrix = _matrix4(view if view is not None else view_proj)

        loader.memcpy_htod(self._view_proj_ptr, ctypes.c_void_p(vp.data_ptr), vp.nbytes)
        loader.memcpy_htod(self._view_matrix_ptr, ctypes.c_void_p(view_matrix.data_ptr), view_matrix.nbytes)
        self._view_proj_uploaded = True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def cull_nodes(
        self,
        positions: object,
        candidate_indices: Optional[object] = None,
        view_proj: Optional[object] = None,
        view: Optional[object] = None,
    ) -> UInt32Vector:
        """Cull candidates based on current view."""
        positions_host = HostTensorF32.from_array_like(positions)
        if positions_host.ndim != 2 or positions_host.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), received {positions_host.shape}")

        if candidate_indices is None:
            candidate_values = list(range(positions_host.shape[0]))
        else:
            candidate_values = [int(value) for value in candidate_indices]

        count = int(len(candidate_values))
        if count == 0:
            return UInt32Vector()

        if view_proj is not None:
            self.upload_view_projection(view_proj, view)
        elif not self._view_proj_uploaded:
            raise RuntimeError("View-projection matrix not uploaded. Call upload_view_projection() first.")

        candidates_buf = (ctypes.c_uint32 * count)(*(int(value) for value in candidate_values))

        pos_ptr = loader.gpu_malloc(positions_host.nbytes)
        cand_ptr = loader.gpu_malloc(ctypes.sizeof(candidates_buf))
        self._ensure_flag_capacity(count)

        loader.memcpy_htod(pos_ptr, ctypes.c_void_p(positions_host.data_ptr), positions_host.nbytes)
        loader.memcpy_htod(cand_ptr, ctypes.c_void_p(ctypes.addressof(candidates_buf)), ctypes.sizeof(candidates_buf))
        if count > 0:
            loader.memcpy_htod(self._flags_ptr, ctypes.c_void_p(ctypes.addressof(self._zero_template)), count)

        count_c = ctypes.c_uint32(count)
        grid_x = (count + self.block_size - 1) // self.block_size

        start_ms = time.perf_counter() if self.enable_profiling else None
        try:
            loader.launch(
                self._kernel,
                (grid_x, 1, 1),
                (self.block_size, 1, 1),
                [pos_ptr, cand_ptr, count_c, self._flags_ptr],
            )
            loader.synchronize()
        finally:
            loader.gpu_free(pos_ptr)
            loader.gpu_free(cand_ptr)

        if start_ms is not None:
            elapsed_ms = (time.perf_counter() - start_ms) * 1000.0
            self.total_time_ms += elapsed_ms
        else:
            elapsed_ms = 0.0

        flags_host = (ctypes.c_uint8 * count)()
        loader.memcpy_dtoh(ctypes.c_void_p(ctypes.addressof(flags_host)), self._flags_ptr, count)

        visible_indices = UInt32Vector(
            int(candidates_buf[idx]) for idx in range(count) if int(flags_host[idx]) != 0
        )

        # Update statistics
        self.total_culls += 1
        self.total_input_nodes += count
        self.total_output_nodes += int(visible_indices.size)
        if start_ms is not None:
            self._last_elapsed_ms = elapsed_ms
        else:
            self._last_elapsed_ms = None

        return visible_indices

    def cull_nodes_device(
        self,
        positions_ptr: loader.CUdeviceptr | int,
        d_candidate_indices: loader.CUdeviceptr | int,
        candidate_count: int,
        *,
        view_proj: Optional[object] = None,
        view: Optional[object] = None,
    ) -> tuple[loader.CUdeviceptr, int]:
        """Cull using device-resident positions and candidate indices."""
        count = int(candidate_count)
        if count <= 0:
            return self._flags_ptr or loader.CUdeviceptr(0), 0
        if view_proj is not None:
            self.upload_view_projection(view_proj, view)
        elif not self._view_proj_uploaded:
            raise RuntimeError("View-projection matrix not uploaded. Call upload_view_projection() first.")
        self._ensure_flag_capacity(count)
        loader.memcpy_htod(
            self._flags_ptr,
            ctypes.c_void_p(ctypes.addressof(self._zero_template)),
            count,
        )
        count_c = ctypes.c_uint32(count)
        grid_x = (count + self.block_size - 1) // self.block_size
        loader.launch(
            self._kernel,
            (grid_x, 1, 1),
            (self.block_size, 1, 1),
            [
                self._coerce_device_ptr(positions_ptr),
                self._coerce_device_ptr(d_candidate_indices),
                count_c,
                self._flags_ptr,
            ],
        )
        loader.synchronize()
        return self._flags_ptr, count

    def read_flags(
        self,
        count: int,
        *,
        flags_ptr: loader.CUdeviceptr | int | None = None,
    ) -> list[int]:
        actual_count = max(0, int(count))
        if actual_count <= 0:
            return []
        ptr = flags_ptr if flags_ptr is not None else self._flags_ptr
        if ptr is None:
            return []
        flags_host = (ctypes.c_uint8 * actual_count)()
        loader.memcpy_dtoh(
            ctypes.c_void_p(ctypes.addressof(flags_host)),
            self._coerce_device_ptr(ptr),
            actual_count,
        )
        return [int(flags_host[idx]) for idx in range(actual_count)]

    def cull_from_octree(
        self,
        candidates: object,
        positions: object,
        view_proj: object,
        view: Optional[object] = None,
    ) -> UInt32Vector:
        """Compat helper mirroring the legacy CuPy API."""
        return self.cull_nodes(positions, candidates, view_proj=view_proj, view=view)

    # ------------------------------------------------------------------
    # Statistics utilities
    # ------------------------------------------------------------------
    def get_statistics(self) -> dict:
        if self.total_culls == 0:
            return {
                "total_culls": 0,
                "avg_input_size": 0,
                "avg_output_size": 0,
                "avg_reduction": 0.0,
                "avg_time_ms": 0.0,
            }

        avg_input = self.total_input_nodes / self.total_culls
        avg_output = self.total_output_nodes / self.total_culls
        avg_reduction = 1.0 - (avg_output / avg_input) if avg_input > 0 else 0.0
        avg_time = self.total_time_ms / self.total_culls

        return {
            "total_culls": self.total_culls,
            "avg_input_size": int(round(avg_input)),
            "avg_output_size": int(round(avg_output)),
            "avg_reduction": avg_reduction,
            "avg_time_ms": avg_time,
        }

    def reset_statistics(self) -> None:
        self.total_culls = 0
        self.total_input_nodes = 0
        self.total_output_nodes = 0
        self.total_time_ms = 0.0
        self._last_elapsed_ms = None

    @staticmethod
    def _coerce_device_ptr(value: loader.CUdeviceptr | int) -> loader.CUdeviceptr:
        if isinstance(value, loader.CUdeviceptr):
            return value
        return loader.CUdeviceptr(int(value))


# ----------------------------------------------------------------------
# Matrix helpers (re-exported for compatibility with legacy tests)
# ----------------------------------------------------------------------
def create_perspective_matrix(
    fov_degrees: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> HostTensorF32:
    fov_rad = math.radians(float(fov_degrees))
    f = 1.0 / math.tan(fov_rad / 2.0)

    return HostTensorF32.from_array_like(
        [
            [f / float(aspect_ratio), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (float(far) + float(near)) / (float(near) - float(far)), (2.0 * float(far) * float(near)) / (float(near) - float(far))],
            [0.0, 0.0, -1.0, 0.0],
        ],
        rows=4,
        cols=4,
    )


def create_view_matrix(eye: object, target: object, up: object) -> HostTensorF32:
    eye_vec = _vector3(eye)
    target_vec = _vector3(target)
    up_vec = _vector3(up)

    forward = _normalize3(
        (
            target_vec[0] - eye_vec[0],
            target_vec[1] - eye_vec[1],
            target_vec[2] - eye_vec[2],
        )
    )
    right = _normalize3(_cross3(forward, up_vec))
    up_actual = _cross3(right, forward)

    return HostTensorF32.from_array_like(
        [
            [right[0], right[1], right[2], -_dot3(right, eye_vec)],
            [up_actual[0], up_actual[1], up_actual[2], -_dot3(up_actual, eye_vec)],
            [-forward[0], -forward[1], -forward[2], _dot3(forward, eye_vec)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        rows=4,
        cols=4,
    )


__all__ = ["FrustumCuller", "UInt32Vector", "create_perspective_matrix", "create_view_matrix", "matmul_4x4", "matvec_4"]
