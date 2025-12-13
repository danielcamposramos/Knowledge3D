"""
PTX wrappers for Drawing Galaxy transformation kernels.

SOVEREIGN: Uses CuPy RawModule; no numpy in hot path.
"""

from __future__ import annotations

from pathlib import Path
import cupy as cp

_KERNEL_SOURCE = Path(__file__).parent.parent / "kernels" / "drawing_transform_ops.cu"
_MODULE: cp.RawModule | None = None


def _get_module() -> cp.RawModule:
    global _MODULE
    if _MODULE is None:
        code = _KERNEL_SOURCE.read_text()
        _MODULE = cp.RawModule(code=code, options=("-std=c++11",))
    return _MODULE


def _get_function(name: str) -> cp.RawKernel:
    return _get_module().get_function(name)


def rot90_cw(grid: cp.ndarray) -> cp.ndarray:
    """Rotate grid 90° clockwise. Returns new array with swapped dims."""
    h, w = grid.shape
    out = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_function("rot90_cw_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def rot90_ccw(grid: cp.ndarray) -> cp.ndarray:
    """Rotate grid 90° counter-clockwise."""
    h, w = grid.shape
    out = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_function("rot90_ccw_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def flip_h(grid: cp.ndarray) -> cp.ndarray:
    """Flip grid horizontally."""
    h, w = grid.shape
    out = cp.empty_like(grid)
    kernel = _get_function("flip_h_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def flip_v(grid: cp.ndarray) -> cp.ndarray:
    """Flip grid vertically."""
    h, w = grid.shape
    out = cp.empty_like(grid)
    kernel = _get_function("flip_v_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def transpose(grid: cp.ndarray) -> cp.ndarray:
    """Transpose grid (flip diagonal)."""
    h, w = grid.shape
    out = cp.empty((w, h), dtype=grid.dtype)
    kernel = _get_function("transpose_kernel")
    block = (16, 16)
    grid_dim = ((h + 15) // 16, (w + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def scale_2x(grid: cp.ndarray) -> cp.ndarray:
    """Scale grid 2x using nearest neighbor."""
    h, w = grid.shape
    out = cp.empty((h * 2, w * 2), dtype=grid.dtype)
    kernel = _get_function("scale_2x_kernel")
    block = (16, 16)
    grid_dim = ((w * 2 + 15) // 16, (h * 2 + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def recolor(grid: cp.ndarray, old_color: int, new_color: int) -> cp.ndarray:
    """Recolor: replace old_color with new_color in-place."""
    h, w = grid.shape
    kernel = _get_function("recolor_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid, old_color, new_color, h, w))
    return grid


def tile_2x2(grid: cp.ndarray) -> cp.ndarray:
    """Tile grid in 2x2 pattern."""
    h, w = grid.shape
    out = cp.empty((h * 2, w * 2), dtype=grid.dtype)
    kernel = _get_function("tile_2x2_kernel")
    block = (16, 16)
    grid_dim = ((w * 2 + 15) // 16, (h * 2 + 15) // 16)
    kernel(grid_dim, block, (grid, out, h, w))
    return out


def overlay(grid_a: cp.ndarray, grid_b: cp.ndarray) -> cp.ndarray:
    """Overlay grid_a on grid_b (non-zero from a wins)."""
    h, w = grid_a.shape
    out = cp.empty_like(grid_a)
    kernel = _get_function("overlay_kernel")
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    kernel(grid_dim, block, (grid_a, grid_b, out, h, w))
    return out


def crop_gpu(grid: cp.ndarray, y: int, x: int, h: int, w: int) -> cp.ndarray:
    """Crop sub-region using PTX kernel."""
    out = cp.zeros((h, w), dtype=cp.int32)
    block = (16, 16)
    grid_dim = ((w + 15) // 16, (h + 15) // 16)
    _get_function("crop_kernel")(grid_dim, block, (grid, out, grid.shape[0], grid.shape[1], y, x, h, w))
    return out


def find_bbox_gpu(grid: cp.ndarray, color: int = 0) -> tuple[int, int, int, int]:
    """Find bounding box of colored region. Returns (min_y, min_x, max_y, max_x)."""
    bbox = cp.array([grid.shape[0], grid.shape[1], -1, -1], dtype=cp.int32)
    block = (16, 16)
    grid_dim = ((grid.shape[1] + 15) // 16, (grid.shape[0] + 15) // 16)
    _get_function("find_bbox_kernel")(grid_dim, block, (grid, bbox, grid.shape[0], grid.shape[1], color))
    return tuple(int(v) for v in bbox.get())


def extract_bbox_gpu(grid: cp.ndarray, color: int = 0) -> cp.ndarray:
    """Find bbox and crop to it in one operation."""
    min_y, min_x, max_y, max_x = find_bbox_gpu(grid, color)
    if max_y < 0 or max_x < 0:
        return cp.zeros((1, 1), dtype=cp.int32)
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    return crop_gpu(grid, min_y, min_x, h, w)


__all__ = [
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
]
