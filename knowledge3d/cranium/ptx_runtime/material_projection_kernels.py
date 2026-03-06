"""Canonical PTX-backed surface material projection runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

PROJECTION_PTX = Path(__file__).parent.parent / "ptx" / "material_projection.ptx"


def _as_float32_rgba(image: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"expected HxWx4 float32 image, got shape={arr.shape}")
    return arr


def _as_float32_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected Nx2 float32 coords, got shape={arr.shape}")
    return arr


def _as_float32_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected Nx3 float32 weights, got shape={arr.shape}")
    return arr


def _as_float32_rgba_rows(rows: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(rows, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected Nx4 float32 rows, got shape={arr.shape}")
    return arr


class MaterialProjectionKernels:
    """PTX kernels for planar sampling and triplanar blending."""

    def __init__(self) -> None:
        module = loader.load_module_from_file(str(PROJECTION_PTX))
        self.sample_planar_kernel = loader.get_function(module, "sample_planar_rgba_kernel")
        self.blend_triplanar_kernel = loader.get_function(module, "blend_triplanar_rgba_kernel")

    def sample_preview(
        self,
        preview: np.ndarray,
        coords: np.ndarray,
        mins: np.ndarray,
        extents: np.ndarray,
        tiling: float,
    ) -> np.ndarray:
        image = _as_float32_rgba(preview)
        coord_arr = _as_float32_coords(coords)
        mins_arr = np.asarray(mins, dtype=np.float32).reshape(2)
        extents_arr = np.maximum(np.asarray(extents, dtype=np.float32).reshape(2), 1e-6)
        vertex_count = int(coord_arr.shape[0])
        if vertex_count == 0:
            return np.empty((0, 4), dtype=np.float32)

        d_preview = loader.gpu_malloc(image.nbytes)
        d_coords = loader.gpu_malloc(coord_arr.nbytes)
        d_out = loader.gpu_malloc(vertex_count * 4 * 4)
        try:
            loader.memcpy_htod(d_preview, image.ctypes.data_as(ctypes.c_void_p), image.nbytes)
            loader.memcpy_htod(d_coords, coord_arr.ctypes.data_as(ctypes.c_void_p), coord_arr.nbytes)
            block = (256, 1, 1)
            grid = ((vertex_count + 255) // 256, 1, 1)
            loader.launch(
                self.sample_planar_kernel,
                grid=grid,
                block=block,
                params=[
                    d_preview,
                    d_coords,
                    d_out,
                    ctypes.c_int(vertex_count),
                    ctypes.c_int(int(image.shape[1])),
                    ctypes.c_int(int(image.shape[0])),
                    ctypes.c_float(float(mins_arr[0])),
                    ctypes.c_float(float(mins_arr[1])),
                    ctypes.c_float(float(extents_arr[0])),
                    ctypes.c_float(float(extents_arr[1])),
                    ctypes.c_float(float(tiling)),
                ],
            )
            out = np.empty((vertex_count, 4), dtype=np.float32)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_preview)
            loader.gpu_free(d_coords)
            loader.gpu_free(d_out)

    def blend_triplanar(
        self,
        yz: np.ndarray,
        xz: np.ndarray,
        xy: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        yz_arr = _as_float32_rgba_rows(yz)
        xz_arr = _as_float32_rgba_rows(xz)
        xy_arr = _as_float32_rgba_rows(xy)
        weight_arr = _as_float32_weights(weights)
        if yz_arr.shape != xz_arr.shape or yz_arr.shape != xy_arr.shape:
            raise ValueError("all triplanar sample planes must have matching Nx4 shape")
        if yz_arr.shape[0] != weight_arr.shape[0]:
            raise ValueError("weights must align with sample count")
        vertex_count = int(yz_arr.shape[0])
        if vertex_count == 0:
            return np.empty((0, 4), dtype=np.float32)

        yz_host = np.ascontiguousarray(yz_arr, dtype=np.float32)
        xz_host = np.ascontiguousarray(xz_arr, dtype=np.float32)
        xy_host = np.ascontiguousarray(xy_arr, dtype=np.float32)
        weights_host = np.ascontiguousarray(weight_arr, dtype=np.float32)

        d_yz = loader.gpu_malloc(yz_host.nbytes)
        d_xz = loader.gpu_malloc(xz_host.nbytes)
        d_xy = loader.gpu_malloc(xy_host.nbytes)
        d_weights = loader.gpu_malloc(weights_host.nbytes)
        d_out = loader.gpu_malloc(yz_host.nbytes)
        try:
            loader.memcpy_htod(d_yz, yz_host.ctypes.data_as(ctypes.c_void_p), yz_host.nbytes)
            loader.memcpy_htod(d_xz, xz_host.ctypes.data_as(ctypes.c_void_p), xz_host.nbytes)
            loader.memcpy_htod(d_xy, xy_host.ctypes.data_as(ctypes.c_void_p), xy_host.nbytes)
            loader.memcpy_htod(d_weights, weights_host.ctypes.data_as(ctypes.c_void_p), weights_host.nbytes)
            block = (256, 1, 1)
            grid = ((vertex_count + 255) // 256, 1, 1)
            loader.launch(
                self.blend_triplanar_kernel,
                grid=grid,
                block=block,
                params=[
                    d_yz,
                    d_xz,
                    d_xy,
                    d_weights,
                    d_out,
                    ctypes.c_int(vertex_count),
                ],
            )
            out = np.empty((vertex_count, 4), dtype=np.float32)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_yz)
            loader.gpu_free(d_xz)
            loader.gpu_free(d_xy)
            loader.gpu_free(d_weights)
            loader.gpu_free(d_out)


__all__ = ["MaterialProjectionKernels"]
