"""
Drawing Bridge: Convert ARC grids to/from PTX drawing surfaces.

Architecture:
    ARC Grid (List[List[int]])
    → Raster Surface (GPU texture/buffer)
    → PTX Drawing Ops (ROTATE, TRANSLATE, FILL)
    → Raster Surface (modified)
    → ARC Grid (List[List[int]])

This bridge is REQUIRED for sovereign GPU execution of ARC grid transforms.
Until implemented, it raises explicit NotImplementedError to prevent silent
CPU fallbacks.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Sequence

from knowledge3d.cranium.sovereign import loader


class DrawingBridge:
    """Convert ARC grids to drawing surfaces for PTX execution."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "ptx" / "arc_grid_ops.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Missing PTX for drawing bridge: {ptx_path}")
        self.module = loader.load_module_from_file(str(ptx_path))
        self.kernel = loader.get_function(self.module, "arc_grid_op")

    def grid_to_surface(self, grid: Sequence[Sequence[int]]) -> tuple[loader.CUdeviceptr, int, int]:
        """
        Convert ARC grid to GPU raster surface.

        Args:
            grid: 2D grid of color indices (0-9)

        Returns:
            (surface_id, width, height)
        """
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        flat: List[int] = []
        for row in grid:
            if len(row) != width:
                raise ValueError("Grid rows must have consistent width")
            flat.extend(int(v) & 0xFF for v in row)

        nbytes = width * height
        d_surface = loader.gpu_malloc(nbytes)
        if nbytes:
            buf = (ctypes.c_ubyte * nbytes)(*flat)
            loader.memcpy_htod(d_surface, ctypes.cast(buf, ctypes.c_void_p), nbytes)
        return d_surface, width, height

    def execute_on_surface(
        self,
        surface_id: loader.CUdeviceptr,
        *,
        src_w: int,
        src_h: int,
        dst_w: int,
        dst_h: int,
        op: int,
        p1: int = 0,
        p2: int = 0,
    ) -> loader.CUdeviceptr:
        """
        Execute PTX drawing ops on surface.

        Args:
            surface_id: GPU surface handle
            src_w, src_h: Source dimensions
            dst_w, dst_h: Destination dimensions
            op: Operation code (rotate/flip/translate/recolor)
            p1, p2: Operation parameters

        Returns:
            destination surface pointer
        """
        nbytes = dst_w * dst_h
        d_out = loader.gpu_malloc(nbytes if nbytes > 0 else 1)
        block = (16, 16, 1)
        grid_x = (dst_w + block[0] - 1) // block[0]
        grid_y = (dst_h + block[1] - 1) // block[1]
        loader.launch(
            self.kernel,
            grid=(grid_x, grid_y, 1),
            block=block,
            params=[
                ctypes.c_uint64(surface_id.value),
                ctypes.c_uint64(d_out.value),
                ctypes.c_int(src_w),
                ctypes.c_int(src_h),
                ctypes.c_int(dst_w),
                ctypes.c_int(dst_h),
                ctypes.c_int(op),
                ctypes.c_int(p1),
                ctypes.c_int(p2),
            ],
        )
        loader.synchronize()
        return d_out

    def surface_to_grid(self, surface_id: int, width: int, height: int) -> List[List[int]]:
        """
        Read GPU surface back to ARC grid.

        Args:
            surface_id: GPU surface handle
            width, height: Expected grid dimensions

        Returns:
            grid: 2D grid of color indices
        """
        nbytes = width * height
        host_buf = (ctypes.c_ubyte * nbytes)()
        if nbytes:
            loader.memcpy_dtoh(ctypes.cast(host_buf, ctypes.c_void_p), surface_id, nbytes)
        grid: List[List[int]] = []
        for y in range(height):
            row = [int(host_buf[y * width + x]) for x in range(width)]
            grid.append(row)
        loader.gpu_free(surface_id)
        return grid


__all__ = ["DrawingBridge"]
