"""Canonical PTX-backed signal visualization runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

SIGNAL_VIS_PTX = Path(__file__).parent.parent / "ptx" / "signal_visualization.ptx"


def _as_int32_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.int32))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D int32 grid, got shape={arr.shape}")
    return arr


class SignalVisualizationKernels:
    """PTX kernels for signal preview rendering."""

    def __init__(self) -> None:
        module = loader.load_module_from_file(str(SIGNAL_VIS_PTX))
        self.spectrogram_to_rgba_kernel = loader.get_function(module, "spectrogram_to_rgba_kernel")

    def spectrogram_to_rgba(self, spectrogram: np.ndarray) -> np.ndarray:
        grid = _as_int32_grid(spectrogram)
        height, width = grid.shape
        d_in = loader.gpu_malloc(grid.nbytes)
        d_out = loader.gpu_malloc(height * width * 4 * 4)
        try:
            loader.memcpy_htod(d_in, grid.ctypes.data_as(ctypes.c_void_p), grid.nbytes)
            block = (16, 16, 1)
            grid_dim = ((width + 15) // 16, (height + 15) // 16, 1)
            loader.launch(
                self.spectrogram_to_rgba_kernel,
                grid=grid_dim,
                block=block,
                params=[
                    d_in,
                    d_out,
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                ],
            )
            out = np.empty((height, width, 4), dtype=np.float32)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)


__all__ = ["SignalVisualizationKernels"]
