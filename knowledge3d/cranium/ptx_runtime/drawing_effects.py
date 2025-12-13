"""
GPU wrappers for Drawing Galaxy gradients and filters.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Tuple

from knowledge3d.cranium.sovereign import loader

GRADIENT_PTX = Path(__file__).parent.parent / "ptx" / "gradient_rasterizer.ptx"
FILTER_PTX = Path(__file__).parent.parent / "ptx" / "filter_convolution.ptx"


class GradientKernels:
    def __init__(self) -> None:
        module = loader.load_module_from_file(str(GRADIENT_PTX))
        self.linear = loader.get_function(module, "gradient_linear_kernel")
        self.radial = loader.get_function(module, "gradient_radial_kernel")
        self.conic = loader.get_function(module, "gradient_conic_kernel")

    def _launch(self, kernel, output, stops, dims, extra: Tuple[float, ...]) -> None:
        h, w = dims
        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], 1)
        params = list(extra) + [stops, ctypes.c_int(len(stops) // 5), ctypes.c_int(w), ctypes.c_int(h)]
        loader.launch(kernel, grid=grid, block=block, params=params)


class FilterKernels:
    def __init__(self) -> None:
        module = loader.load_module_from_file(str(FILTER_PTX))
        self.blur_h = loader.get_function(module, "blur_horizontal_kernel")
        self.blur_v = loader.get_function(module, "blur_vertical_kernel")
        self.sobel = loader.get_function(module, "sobel_edge_kernel")
        self.sharpen = loader.get_function(module, "sharpen_kernel")


class DrawingEffects:
    """Convenience loader exposing gradient and filter kernels."""

    def __init__(self) -> None:
        self.gradients = GradientKernels()
        self.filters = FilterKernels()
        # For quick checks
        self.gradient_linear_kernel = getattr(self.gradients, "linear", None)
        self.filter_blur_kernel = getattr(self.filters, "blur_h", None)


__all__ = ["GradientKernels", "FilterKernels", "DrawingEffects"]
