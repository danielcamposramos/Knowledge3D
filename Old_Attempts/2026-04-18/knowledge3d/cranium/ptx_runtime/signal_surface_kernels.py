"""Canonical PTX-backed heightfield surface assembly runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

SIGNAL_SURFACE_PTX = Path(__file__).parent.parent / "ptx" / "signal_surface_ops.ptx"


def _as_float32_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D float32 grid, got shape={arr.shape}")
    return arr


class SignalSurfaceKernels:
    """PTX kernels for heightfield vertex and normal generation."""

    def __init__(self) -> None:
        module = loader.load_module_from_file(str(SIGNAL_SURFACE_PTX))
        self.heightfield_to_vertices_kernel = loader.get_function(module, "heightfield_to_vertices_kernel")
        self.heightfield_to_normals_kernel = loader.get_function(module, "heightfield_to_normals_kernel")

    def heightfield_to_vertices(
        self,
        heightfield: np.ndarray,
        *,
        time_scale: float = 1.0,
        frequency_scale: float = 1.0,
    ) -> np.ndarray:
        grid = _as_float32_grid(heightfield)
        rows, cols = grid.shape
        total = int(rows * cols)
        if total == 0:
            return np.empty((0, 3), dtype=np.float32)

        d_grid = loader.gpu_malloc(grid.nbytes)
        d_out = loader.gpu_malloc(total * 3 * 4)
        try:
            loader.memcpy_htod(d_grid, grid.ctypes.data_as(ctypes.c_void_p), grid.nbytes)
            block = (256, 1, 1)
            grid_dim = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.heightfield_to_vertices_kernel,
                grid=grid_dim,
                block=block,
                params=[
                    d_grid,
                    d_out,
                    ctypes.c_int(rows),
                    ctypes.c_int(cols),
                    ctypes.c_float(float(time_scale)),
                    ctypes.c_float(float(frequency_scale)),
                ],
            )
            out = np.empty((total, 3), dtype=np.float32)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_grid)
            loader.gpu_free(d_out)

    def heightfield_to_normals(
        self,
        heightfield: np.ndarray,
        *,
        time_scale: float = 1.0,
        frequency_scale: float = 1.0,
    ) -> np.ndarray:
        grid = _as_float32_grid(heightfield)
        rows, cols = grid.shape
        total = int(rows * cols)
        if total == 0:
            return np.empty((0, 3), dtype=np.float32)

        d_grid = loader.gpu_malloc(grid.nbytes)
        d_out = loader.gpu_malloc(total * 3 * 4)
        try:
            loader.memcpy_htod(d_grid, grid.ctypes.data_as(ctypes.c_void_p), grid.nbytes)
            block = (256, 1, 1)
            grid_dim = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.heightfield_to_normals_kernel,
                grid=grid_dim,
                block=block,
                params=[
                    d_grid,
                    d_out,
                    ctypes.c_int(rows),
                    ctypes.c_int(cols),
                    ctypes.c_float(float(time_scale)),
                    ctypes.c_float(float(frequency_scale)),
                ],
            )
            out = np.empty((total, 3), dtype=np.float32)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_grid)
            loader.gpu_free(d_out)


__all__ = ["SignalSurfaceKernels"]
