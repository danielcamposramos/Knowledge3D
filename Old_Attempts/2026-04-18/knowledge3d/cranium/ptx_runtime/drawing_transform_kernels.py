"""Canonical sovereign PTX wrappers for Drawing Galaxy transform kernels.

This module centralizes grid-shaped integer transforms used by drawing,
contour extraction, and geometry preparation. Host code only handles upload,
download, and launch orchestration; the transforms themselves execute through
the sovereign CUDA loader against a compiled PTX module.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

TRANSFORM_PTX = Path(__file__).parent.parent / "ptx" / "drawing_transform_ops.ptx"

_MODULE = None
_FUNCTIONS: dict[str, loader.CUfunction] = {}


def _get_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = loader.load_module_from_file(str(TRANSFORM_PTX))
    return _MODULE


def _get_function(name: str) -> loader.CUfunction:
    fn = _FUNCTIONS.get(name)
    if fn is None:
        fn = loader.get_function(_get_module(), name)
        _FUNCTIONS[name] = fn
    return fn


def _as_int32_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int32))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D int32 grid, got shape={arr.shape}")
    return arr


@dataclass
class DeviceIntGrid:
    device_ptr: loader.CUdeviceptr
    shape: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * 4

    def to_numpy(self) -> np.ndarray:
        host = np.empty(self.shape, dtype=np.int32)
        loader.memcpy_dtoh(host.ctypes.data_as(ctypes.c_void_p), self.device_ptr, host.nbytes)
        return host

    def free(self) -> None:
        if getattr(self.device_ptr, "value", 0):
            loader.gpu_free(self.device_ptr)
            self.device_ptr = loader.CUdeviceptr(0)


def _alloc_grid(shape: tuple[int, int], *, fill: int = 0) -> DeviceIntGrid:
    host = np.full(shape, fill, dtype=np.int32)
    device_ptr = loader.gpu_malloc(host.nbytes)
    loader.memcpy_htod(device_ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes)
    return DeviceIntGrid(device_ptr=device_ptr, shape=shape)


def _upload_grid(grid: np.ndarray) -> DeviceIntGrid:
    host = _as_int32_grid(grid)
    device_ptr = loader.gpu_malloc(host.nbytes)
    loader.memcpy_htod(device_ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes)
    return DeviceIntGrid(device_ptr=device_ptr, shape=tuple(host.shape))


def _run_transform_same_shape(name: str, grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    d_in = _upload_grid(host)
    d_out = _alloc_grid(tuple(host.shape))
    try:
        height, width = host.shape
        loader.launch(
            _get_function(name),
            grid=((width + 15) // 16, (height + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def rot90_cw(grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    height, width = host.shape
    d_in = _upload_grid(host)
    d_out = _alloc_grid((width, height))
    try:
        loader.launch(
            _get_function("rot90_cw_kernel"),
            grid=((height + 15) // 16, (width + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def rot90_ccw(grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    height, width = host.shape
    d_in = _upload_grid(host)
    d_out = _alloc_grid((width, height))
    try:
        loader.launch(
            _get_function("rot90_ccw_kernel"),
            grid=((height + 15) // 16, (width + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def flip_h(grid: np.ndarray) -> np.ndarray:
    return _run_transform_same_shape("flip_h_kernel", grid)


def flip_v(grid: np.ndarray) -> np.ndarray:
    return _run_transform_same_shape("flip_v_kernel", grid)


def transpose(grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    height, width = host.shape
    d_in = _upload_grid(host)
    d_out = _alloc_grid((width, height))
    try:
        loader.launch(
            _get_function("transpose_kernel"),
            grid=((height + 15) // 16, (width + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def scale_2x(grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    height, width = host.shape
    d_in = _upload_grid(host)
    d_out = _alloc_grid((height * 2, width * 2))
    try:
        loader.launch(
            _get_function("scale_2x_kernel"),
            grid=(((width * 2) + 15) // 16, ((height * 2) + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def recolor(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    host = _as_int32_grid(grid)
    d_grid = _upload_grid(host)
    try:
        height, width = host.shape
        loader.launch(
            _get_function("recolor_kernel"),
            grid=((width + 15) // 16, (height + 15) // 16, 1),
            block=(16, 16, 1),
            params=[
                d_grid.device_ptr,
                ctypes.c_int(old_color),
                ctypes.c_int(new_color),
                ctypes.c_int(height),
                ctypes.c_int(width),
            ],
        )
        return d_grid.to_numpy()
    finally:
        d_grid.free()


def tile_2x2(grid: np.ndarray) -> np.ndarray:
    host = _as_int32_grid(grid)
    height, width = host.shape
    d_in = _upload_grid(host)
    d_out = _alloc_grid((height * 2, width * 2))
    try:
        loader.launch(
            _get_function("tile_2x2_kernel"),
            grid=(((width * 2) + 15) // 16, ((height * 2) + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_in.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def overlay(grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
    host_a = _as_int32_grid(grid_a)
    host_b = _as_int32_grid(grid_b)
    if host_a.shape != host_b.shape:
        raise ValueError(f"overlay expects equal shapes, got {host_a.shape} and {host_b.shape}")
    height, width = host_a.shape
    d_a = _upload_grid(host_a)
    d_b = _upload_grid(host_b)
    d_out = _alloc_grid((height, width))
    try:
        loader.launch(
            _get_function("overlay_kernel"),
            grid=((width + 15) // 16, (height + 15) // 16, 1),
            block=(16, 16, 1),
            params=[d_a.device_ptr, d_b.device_ptr, d_out.device_ptr, ctypes.c_int(height), ctypes.c_int(width)],
        )
        return d_out.to_numpy()
    finally:
        d_a.free()
        d_b.free()
        d_out.free()


def crop_gpu(grid: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    host = _as_int32_grid(grid)
    if h < 1 or w < 1:
        raise ValueError("crop size must be positive")
    d_in = _upload_grid(host)
    d_out = _alloc_grid((h, w))
    try:
        loader.launch(
            _get_function("crop_kernel"),
            grid=((w + 15) // 16, (h + 15) // 16, 1),
            block=(16, 16, 1),
            params=[
                d_in.device_ptr,
                d_out.device_ptr,
                ctypes.c_int(host.shape[0]),
                ctypes.c_int(host.shape[1]),
                ctypes.c_int(y),
                ctypes.c_int(x),
                ctypes.c_int(h),
                ctypes.c_int(w),
            ],
        )
        return d_out.to_numpy()
    finally:
        d_in.free()
        d_out.free()


def find_bbox_gpu(grid: np.ndarray, color: int = 0) -> tuple[int, int, int, int]:
    host = _as_int32_grid(grid)
    seed = np.array([host.shape[0], host.shape[1], -1, -1], dtype=np.int32)
    d_in = _upload_grid(host)
    d_bbox = _upload_grid(seed.reshape(1, 4))
    try:
        loader.launch(
            _get_function("find_bbox_kernel"),
            grid=((host.shape[1] + 15) // 16, (host.shape[0] + 15) // 16, 1),
            block=(16, 16, 1),
            params=[
                d_in.device_ptr,
                d_bbox.device_ptr,
                ctypes.c_int(host.shape[0]),
                ctypes.c_int(host.shape[1]),
                ctypes.c_int(color),
            ],
        )
        out = d_bbox.to_numpy().reshape(-1)
        return tuple(int(v) for v in out.tolist())
    finally:
        d_in.free()
        d_bbox.free()


def extract_bbox_gpu(grid: np.ndarray, color: int = 0) -> np.ndarray:
    min_y, min_x, max_y, max_x = find_bbox_gpu(grid, color)
    if max_y < 0 or max_x < 0:
        return np.zeros((1, 1), dtype=np.int32)
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    return crop_gpu(grid, min_y, min_x, height, width)


def profile_scan_gpu(grid: np.ndarray, color: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host = _as_int32_grid(grid)
    width = host.shape[1]
    d_in = _upload_grid(host)
    d_top = _alloc_grid((1, width), fill=-1)
    d_bottom = _alloc_grid((1, width), fill=-1)
    d_fill = _alloc_grid((1, width), fill=0)
    try:
        loader.launch(
            _get_function("profile_scan_kernel"),
            grid=(((width) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            params=[
                d_in.device_ptr,
                d_top.device_ptr,
                d_bottom.device_ptr,
                d_fill.device_ptr,
                ctypes.c_int(host.shape[0]),
                ctypes.c_int(host.shape[1]),
                ctypes.c_int(color),
            ],
        )
        top = d_top.to_numpy().reshape(-1)
        bottom = d_bottom.to_numpy().reshape(-1)
        fill = d_fill.to_numpy().reshape(-1)
        return top, bottom, fill
    finally:
        d_in.free()
        d_top.free()
        d_bottom.free()
        d_fill.free()


def row_profile_scan_gpu(grid: np.ndarray, color: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    host = _as_int32_grid(grid)
    height = host.shape[0]
    d_in = _upload_grid(host)
    d_left = _alloc_grid((height, 1), fill=-1)
    d_right = _alloc_grid((height, 1), fill=-1)
    d_fill = _alloc_grid((height, 1), fill=0)
    try:
        loader.launch(
            _get_function("row_profile_scan_kernel"),
            grid=(((height) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            params=[
                d_in.device_ptr,
                d_left.device_ptr,
                d_right.device_ptr,
                d_fill.device_ptr,
                ctypes.c_int(host.shape[0]),
                ctypes.c_int(host.shape[1]),
                ctypes.c_int(color),
            ],
        )
        left = d_left.to_numpy().reshape(-1)
        right = d_right.to_numpy().reshape(-1)
        fill = d_fill.to_numpy().reshape(-1)
        return left, right, fill
    finally:
        d_in.free()
        d_left.free()
        d_right.free()
        d_fill.free()


def smooth_profile_gpu(values: np.ndarray, *, passes: int = 1, invalid_value: int = -1) -> np.ndarray:
    host = np.ascontiguousarray(np.asarray(values, dtype=np.int32)).reshape(-1)
    if host.ndim != 1:
        raise ValueError("smooth_profile_gpu expects 1D int32 input")
    if host.size == 0:
        return host.copy()
    if passes < 1:
        return host.copy()

    d_in = _alloc_grid((1, host.size))
    d_out = _alloc_grid((1, host.size))
    try:
        loader.memcpy_htod(d_in.device_ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes)
        for _ in range(int(passes)):
            loader.launch(
                _get_function("smooth_profile_kernel"),
                grid=(((host.size) + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    d_in.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_int(host.size),
                    ctypes.c_int(invalid_value),
                ],
            )
            loader.synchronize()
            d_in, d_out = d_out, d_in
        return d_in.to_numpy().reshape(-1)
    finally:
        d_in.free()
        d_out.free()


__all__ = [
    "DeviceIntGrid",
    "rot90_cw",
    "rot90_ccw",
    "flip_h",
    "flip_v",
    "transpose",
    "scale_2x",
    "recolor",
    "tile_2x2",
    "overlay",
    "crop_gpu",
    "find_bbox_gpu",
    "extract_bbox_gpu",
    "profile_scan_gpu",
    "row_profile_scan_gpu",
    "smooth_profile_gpu",
]
