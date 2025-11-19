"""
Procedural glyph rendering bridge.

Loads the procedural_glyph_rasterizer PTX module and exposes utilities to
render glyph descriptor batches directly on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

from knowledge3d.cranium.sovereign import loader


@dataclass
class RasterizerBatch:
    """Container for rendered glyph tensors."""

    device_ptr: loader.CUdeviceptr
    batch: int
    height: int
    width: int

    def to_numpy(self) -> np.ndarray:
        """Copy the rendered batch back to host memory."""
        host = np.empty((self.batch, self.height, self.width, 4), dtype=np.float32)
        loader.memcpy_dtoh(
            host.ctypes.data_as(ctypes.c_void_p),
            self.device_ptr,
            host.nbytes,
        )
        self.free()
        return host

    def free(self) -> None:
        """Release device memory."""
        if getattr(self.device_ptr, "value", 0):
            loader.gpu_free(self.device_ptr)
            self.device_ptr = loader.CUdeviceptr(0)


class ProceduralGlyphBridge:
    """Thin wrapper around the procedural glyph rasterizer PTX kernel."""

    def __init__(self, ptx_path: Optional[str] = None):
        default_ptx = (
            Path(__file__).parent.parent / "kernels" / "procedural_glyph_rasterizer.ptx"
        )
        ptx_file = ptx_path if ptx_path else str(default_ptx)
        self.module = loader.load_module_from_file(ptx_file)
        self.kernel = loader.get_function(self.module, "procedural_glyph_rasterizer")

    def render(
        self,
        segments: np.ndarray,
        segment_offsets: np.ndarray,
        segment_lengths: np.ndarray,
        transforms: np.ndarray,
        batch: int,
        height: int,
        width: int,
    ) -> RasterizerBatch:
        """
        Render glyphs described by the provided segments.

        Args:
            segments: float32 array (N,9) describing x0,y0,x1,y1,r,g,b,a,width.
            segment_offsets: int32 offsets per glyph.
            segment_lengths: int32 lengths per glyph.
            transforms: float32 array (batch, 4) -> scale, rotation, tx, ty.
            batch: number of glyphs.
            height/width: output resolution.
        """
        segments = np.ascontiguousarray(segments.astype(np.float32, copy=False))
        offsets = np.ascontiguousarray(segment_offsets.astype(np.int32, copy=False))
        lengths = np.ascontiguousarray(segment_lengths.astype(np.int32, copy=False))
        transforms = np.ascontiguousarray(transforms.astype(np.float32, copy=False))

        d_segments = loader.gpu_malloc(segments.nbytes) if segments.size else loader.CUdeviceptr(0)
        d_offsets = loader.gpu_malloc(offsets.nbytes)
        d_lengths = loader.gpu_malloc(lengths.nbytes)
        d_transforms = loader.gpu_malloc(transforms.nbytes)
        output_bytes = batch * height * width * 4 * 4  # RGBA float32
        d_output = loader.gpu_malloc(output_bytes)

        if segments.size:
            loader.memcpy_htod(d_segments, segments.ctypes.data_as(ctypes.c_void_p), segments.nbytes)
        loader.memcpy_htod(d_offsets, offsets.ctypes.data_as(ctypes.c_void_p), offsets.nbytes)
        loader.memcpy_htod(d_lengths, lengths.ctypes.data_as(ctypes.c_void_p), lengths.nbytes)
        loader.memcpy_htod(d_transforms, transforms.ctypes.data_as(ctypes.c_void_p), transforms.nbytes)

        block = (16, 16, 1)
        grid = (
            (width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1],
            batch,
        )

        loader.launch(
            self.kernel,
            grid=grid,
            block=block,
            params=[
                d_segments,
                d_offsets,
                d_lengths,
                d_transforms,
                d_output,
                ctypes.c_int(batch),
                ctypes.c_int(height),
                ctypes.c_int(width),
            ],
        )

        loader.gpu_free(d_segments)
        loader.gpu_free(d_offsets)
        loader.gpu_free(d_lengths)
        loader.gpu_free(d_transforms)

        return RasterizerBatch(device_ptr=d_output, batch=batch, height=height, width=width)
