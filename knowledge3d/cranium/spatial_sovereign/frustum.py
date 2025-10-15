"""
Sovereign frustum culling wrapper built on pre-compiled PTX kernels.

This module replaces the legacy CuPy-based implementation with a thin Python
layer that talks directly to the CUDA Driver API through the sovereign loader.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Optional

import numpy as np

from knowledge3d.cranium.sovereign import loader


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
        self._zero_template = np.zeros(0, dtype=np.uint8)
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
        self._zero_template = np.zeros(self._flags_capacity, dtype=np.uint8)

    def close(self):
        """Release device resources."""
        if self._flags_ptr is not None:
            loader.gpu_free(self._flags_ptr)
            self._flags_ptr = None
            self._flags_capacity = 0
            self._zero_template = np.zeros(0, dtype=np.uint8)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors should never raise.
            pass

    # ------------------------------------------------------------------
    # Constant memory uploads
    # ------------------------------------------------------------------
    def upload_view_projection(self, view_proj: np.ndarray, view: Optional[np.ndarray] = None) -> None:
        """Upload view-projection and view matrices into constant memory."""
        vp = np.asarray(view_proj, dtype=np.float32)
        if vp.shape != (4, 4):
            raise ValueError(f"view_proj must be 4x4, received {vp.shape}")

        view_matrix = np.asarray(view if view is not None else view_proj, dtype=np.float32)
        if view_matrix.shape != (4, 4):
            raise ValueError(f"view must be 4x4, received {view_matrix.shape}")

        vp_flat = np.ascontiguousarray(vp.ravel())
        view_flat = np.ascontiguousarray(view_matrix.ravel())

        loader.memcpy_htod(self._view_proj_ptr, ctypes.c_void_p(vp_flat.ctypes.data), vp_flat.nbytes)
        loader.memcpy_htod(self._view_matrix_ptr, ctypes.c_void_p(view_flat.ctypes.data), view_flat.nbytes)
        self._view_proj_uploaded = True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def cull_nodes(
        self,
        positions: np.ndarray,
        candidate_indices: Optional[np.ndarray] = None,
        view_proj: Optional[np.ndarray] = None,
        view: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Cull candidates based on current view."""
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be shape (N, 3), received {positions.shape}")

        if candidate_indices is None:
            candidate_indices = np.arange(positions.shape[0], dtype=np.uint32)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=np.uint32)

        if candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be 1-D array")

        count = int(candidate_indices.size)
        if count == 0:
            return np.zeros(0, dtype=np.uint32)

        if view_proj is not None:
            self.upload_view_projection(view_proj, view)
        elif not self._view_proj_uploaded:
            raise RuntimeError("View-projection matrix not uploaded. Call upload_view_projection() first.")

        positions_contig = np.ascontiguousarray(positions)
        candidates_contig = np.ascontiguousarray(candidate_indices)

        pos_ptr = loader.gpu_malloc(positions_contig.nbytes)
        cand_ptr = loader.gpu_malloc(candidates_contig.nbytes)
        self._ensure_flag_capacity(count)

        loader.memcpy_htod(pos_ptr, ctypes.c_void_p(positions_contig.ctypes.data), positions_contig.nbytes)
        loader.memcpy_htod(cand_ptr, ctypes.c_void_p(candidates_contig.ctypes.data), candidates_contig.nbytes)
        loader.memcpy_htod(self._flags_ptr, ctypes.c_void_p(self._zero_template.ctypes.data), count)

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

        flags_host = np.empty(count, dtype=np.uint8)
        loader.memcpy_dtoh(ctypes.c_void_p(flags_host.ctypes.data), self._flags_ptr, count)

        visible_indices = candidates_contig[flags_host.astype(bool)]

        # Update statistics
        self.total_culls += 1
        self.total_input_nodes += count
        self.total_output_nodes += int(visible_indices.size)
        if start_ms is not None:
            self._last_elapsed_ms = elapsed_ms
        else:
            self._last_elapsed_ms = None

        return visible_indices

    def cull_from_octree(
        self,
        candidates: np.ndarray,
        positions: np.ndarray,
        view_proj: np.ndarray,
        view: Optional[np.ndarray] = None,
    ) -> np.ndarray:
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


# ----------------------------------------------------------------------
# Matrix helpers (re-exported for compatibility with legacy tests)
# ----------------------------------------------------------------------
def create_perspective_matrix(
    fov_degrees: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> np.ndarray:
    fov_rad = np.radians(fov_degrees)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect_ratio
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj


def create_view_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up_actual = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up_actual
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up_actual, eye)
    view[2, 3] = np.dot(forward, eye)

    return view


__all__ = ["FrustumCuller", "create_perspective_matrix", "create_view_matrix"]
