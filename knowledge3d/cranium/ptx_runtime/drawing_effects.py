"""Canonical sovereign GPU canvas for Drawing Galaxy gradients and filters.

This module centralizes the real PTX-backed visual editing surface used by
procedural drawing. Host code only orchestrates buffer upload/download and
kernel sequencing; the actual visual operations run on CUDA kernels.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .ternary_gradient_logic import ContrastiveGradientScore, TernaryGradientLogic, TernaryGradientSignature
from .ternary_palette_logic import ContrastivePaletteScore, TernaryPaletteLogic, TernaryPaletteSignature
from knowledge3d.cranium.sovereign import loader

GRADIENT_PTX = Path(__file__).parent.parent / "ptx" / "gradient_rasterizer.ptx"
FILTER_PTX = Path(__file__).parent.parent / "ptx" / "filter_convolution.ptx"


def _as_float32_image(image: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(image.astype(np.float32, copy=False))
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected 2D or 3D image, got shape={arr.shape}")
    if arr.shape[2] not in {1, 3, 4}:
        raise ValueError(f"expected 1, 3, or 4 channels, got shape={arr.shape}")
    return arr


def _as_stop_array(stops: Sequence[Sequence[float]]) -> np.ndarray:
    if not stops:
        raise ValueError("at least one gradient stop is required")
    arr = np.ascontiguousarray(np.asarray(stops, dtype=np.float32))
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("stops must have shape [n,5] as [pos,r,g,b,a]")
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
    arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)
    return arr


def _gaussian_kernel(radius: int) -> np.ndarray:
    if radius < 1:
        raise ValueError("radius must be >= 1")
    sigma = max(radius / 2.0, 0.5)
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return np.ascontiguousarray(kernel.astype(np.float32, copy=False))


@dataclass
class DeviceCanvas:
    """GPU-resident float32 canvas."""

    device_ptr: loader.CUdeviceptr
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * 4

    def to_numpy(self) -> np.ndarray:
        host = np.empty(self.shape, dtype=np.float32)
        loader.memcpy_dtoh(host.ctypes.data_as(ctypes.c_void_p), self.device_ptr, host.nbytes)
        return host

    def free(self) -> None:
        if getattr(self.device_ptr, "value", 0):
            loader.gpu_free(self.device_ptr)
            self.device_ptr = loader.CUdeviceptr(0)


def _alloc_canvas(shape: tuple[int, ...]) -> DeviceCanvas:
    nbytes = int(np.prod(shape)) * 4
    device_ptr = loader.gpu_malloc(nbytes)
    return DeviceCanvas(device_ptr=device_ptr, shape=shape)


def _upload_array(array: np.ndarray) -> DeviceCanvas:
    host = _as_float32_image(array)
    canvas = _alloc_canvas(host.shape)
    loader.memcpy_htod(canvas.device_ptr, host.ctypes.data_as(ctypes.c_void_p), host.nbytes)
    return canvas


class GradientKernels:
    def __init__(self) -> None:
        module = loader.load_module_from_file(str(GRADIENT_PTX))
        self.linear_kernel = loader.get_function(module, "gradient_linear_kernel")
        self.radial_kernel = loader.get_function(module, "gradient_radial_kernel")
        self.conic_kernel = loader.get_function(module, "gradient_conic_kernel")

    def _render(
        self,
        kernel,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        extra: Iterable[ctypes._SimpleCData],
    ) -> np.ndarray:
        stop_arr = _as_stop_array(stops)
        output = _alloc_canvas((height, width, 4))
        d_stops = loader.gpu_malloc(stop_arr.nbytes)
        try:
            loader.memcpy_htod(d_stops, stop_arr.ctypes.data_as(ctypes.c_void_p), stop_arr.nbytes)
            block = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16, 1)
            params = [
                output.device_ptr,
                *list(extra),
                d_stops,
                ctypes.c_int(stop_arr.shape[0]),
                ctypes.c_int(width),
                ctypes.c_int(height),
            ]
            loader.launch(kernel, grid=grid, block=block, params=params)
            return output.to_numpy()
        finally:
            output.free()
            loader.gpu_free(d_stops)

    def linear(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        return self._render(
            self.linear_kernel,
            width,
            height,
            stops,
            (
                ctypes.c_float(x1),
                ctypes.c_float(y1),
                ctypes.c_float(x2),
                ctypes.c_float(y2),
            ),
        )

    def radial(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        cx: float,
        cy: float,
        radius: float,
    ) -> np.ndarray:
        return self._render(
            self.radial_kernel,
            width,
            height,
            stops,
            (
                ctypes.c_float(cx),
                ctypes.c_float(cy),
                ctypes.c_float(radius),
            ),
        )

    def conic(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        cx: float,
        cy: float,
        start_angle: float,
    ) -> np.ndarray:
        return self._render(
            self.conic_kernel,
            width,
            height,
            stops,
            (
                ctypes.c_float(cx),
                ctypes.c_float(cy),
                ctypes.c_float(start_angle),
            ),
        )


class FilterKernels:
    def __init__(self) -> None:
        module = loader.load_module_from_file(str(FILTER_PTX))
        self.blur_h_kernel = loader.get_function(module, "blur_horizontal_kernel")
        self.blur_v_kernel = loader.get_function(module, "blur_vertical_kernel")
        self.sobel_kernel = loader.get_function(module, "sobel_edge_kernel")
        self.sharpen_kernel = loader.get_function(module, "sharpen_kernel")
        self.luma_kernel = loader.get_function(module, "rgba_to_luma_kernel")
        self.alpha_over_kernel = loader.get_function(module, "alpha_over_rgba_kernel")
        self.invert_kernel = loader.get_function(module, "invert_rgba_kernel")

    def blur(self, image: np.ndarray, *, radius: int) -> np.ndarray:
        host = _as_float32_image(image)
        if host.ndim != 3:
            raise ValueError("blur expects HxWxC image")
        height, width, channels = host.shape
        kernel_host = _gaussian_kernel(radius)

        d_in = _upload_array(host)
        d_tmp = _alloc_canvas(host.shape)
        d_out = _alloc_canvas(host.shape)
        d_kernel = loader.gpu_malloc(kernel_host.nbytes)
        try:
            loader.memcpy_htod(d_kernel, kernel_host.ctypes.data_as(ctypes.c_void_p), kernel_host.nbytes)

            block_h = (min(max(width + 2 * radius, 32), 256), 1, 1)
            grid_h = (1, height, channels)
            shared_h = (width + 2 * radius) * 4
            loader.launch(
                self.blur_h_kernel,
                grid=grid_h,
                block=block_h,
                params=[
                    d_in.device_ptr,
                    d_tmp.device_ptr,
                    d_kernel,
                    ctypes.c_int(radius),
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                    ctypes.c_int(channels),
                ],
                shared_mem=shared_h,
            )

            block_v = (1, min(max(height + 2 * radius, 32), 256), 1)
            grid_v = (width, 1, channels)
            shared_v = (height + 2 * radius) * 4
            loader.launch(
                self.blur_v_kernel,
                grid=grid_v,
                block=block_v,
                params=[
                    d_tmp.device_ptr,
                    d_out.device_ptr,
                    d_kernel,
                    ctypes.c_int(radius),
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                    ctypes.c_int(channels),
                ],
                shared_mem=shared_v,
            )
            return d_out.to_numpy()
        finally:
            d_in.free()
            d_tmp.free()
            d_out.free()
            loader.gpu_free(d_kernel)

    def sharpen(self, image: np.ndarray, *, radius: int = 1, amount: float = 1.0) -> np.ndarray:
        host = _as_float32_image(image)
        if host.ndim != 3:
            raise ValueError("sharpen expects HxWxC image")
        height, width, channels = host.shape
        blurred = self.blur(host, radius=radius)

        d_in = _upload_array(host)
        d_blur = _upload_array(blurred)
        d_out = _alloc_canvas(host.shape)
        try:
            total = width * height * channels
            block = (256, 1, 1)
            grid = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.sharpen_kernel,
                grid=grid,
                block=block,
                params=[
                    d_in.device_ptr,
                    d_blur.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_float(amount),
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                    ctypes.c_int(channels),
                ],
            )
            return np.clip(d_out.to_numpy(), 0.0, 1.0)
        finally:
            d_in.free()
            d_blur.free()
            d_out.free()

    def rgba_to_luma(self, image: np.ndarray) -> np.ndarray:
        host = _as_float32_image(image)
        if host.ndim != 3:
            raise ValueError("rgba_to_luma expects HxWxC image")
        height, width, channels = host.shape
        d_in = _upload_array(host)
        d_out = _alloc_canvas((height, width))
        try:
            block = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16, 1)
            loader.launch(
                self.luma_kernel,
                grid=grid,
                block=block,
                params=[
                    d_in.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                    ctypes.c_int(channels),
                ],
            )
            return d_out.to_numpy()
        finally:
            d_in.free()
            d_out.free()

    def sobel_edges(self, image: np.ndarray) -> np.ndarray:
        luma = self.rgba_to_luma(image)
        height, width = luma.shape
        d_in = _upload_array(luma)
        d_out = _alloc_canvas((height, width))
        try:
            block = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16, 1)
            loader.launch(
                self.sobel_kernel,
                grid=grid,
                block=block,
                params=[
                    d_in.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                ],
            )
            return d_out.to_numpy()
        finally:
            d_in.free()
            d_out.free()

    def alpha_over_rgba(self, background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        bg = _as_float32_image(background)
        fg = _as_float32_image(foreground)
        if bg.shape != fg.shape or bg.ndim != 3 or bg.shape[2] != 4:
            raise ValueError("alpha_over_rgba expects matching HxWx4 canvases")
        height, width, _channels = bg.shape
        d_bg = _upload_array(bg)
        d_fg = _upload_array(fg)
        d_out = _alloc_canvas(bg.shape)
        try:
            block = (16, 16, 1)
            grid = ((width + 15) // 16, (height + 15) // 16, 1)
            loader.launch(
                self.alpha_over_kernel,
                grid=grid,
                block=block,
                params=[
                    d_bg.device_ptr,
                    d_fg.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                ],
            )
            return np.clip(d_out.to_numpy(), 0.0, 1.0)
        finally:
            d_bg.free()
            d_fg.free()
            d_out.free()

    def invert(self, image: np.ndarray) -> np.ndarray:
        host = _as_float32_image(image)
        if host.ndim != 3:
            raise ValueError("invert expects HxWxC image")
        height, width, channels = host.shape
        total = height * width * channels
        d_in = _upload_array(host)
        d_out = _alloc_canvas(host.shape)
        try:
            block = (256, 1, 1)
            grid = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.invert_kernel,
                grid=grid,
                block=block,
                params=[
                    d_in.device_ptr,
                    d_out.device_ptr,
                    ctypes.c_int(total),
                    ctypes.c_int(channels),
                ],
            )
            return np.clip(d_out.to_numpy(), 0.0, 1.0)
        finally:
            d_in.free()
            d_out.free()


class DrawingEffects:
    """Single canonical surface for sovereign procedural canvas operations."""

    def __init__(self) -> None:
        self.gradients = GradientKernels()
        self.filters = FilterKernels()
        self.ternary_gradient = TernaryGradientLogic()
        self.ternary_palette = TernaryPaletteLogic()

        # Quick probe attrs used by existing diagnostics/tests.
        self.gradient_linear_kernel = getattr(self.gradients, "linear_kernel", None)
        self.filter_blur_kernel = getattr(self.filters, "blur_h_kernel", None)

    def linear_gradient(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray:
        return self.gradients.linear(width, height, stops, x1=x1, y1=y1, x2=x2, y2=y2)

    def radial_gradient(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        cx: float,
        cy: float,
        radius: float,
    ) -> np.ndarray:
        return self.gradients.radial(width, height, stops, cx=cx, cy=cy, radius=radius)

    def conic_gradient(
        self,
        width: int,
        height: int,
        stops: Sequence[Sequence[float]],
        *,
        cx: float,
        cy: float,
        start_angle: float,
    ) -> np.ndarray:
        return self.gradients.conic(width, height, stops, cx=cx, cy=cy, start_angle=start_angle)

    def blur_rgba(self, image: np.ndarray, *, radius: int) -> np.ndarray:
        return self.filters.blur(image, radius=radius)

    def sharpen_rgba(self, image: np.ndarray, *, radius: int = 1, amount: float = 1.0) -> np.ndarray:
        return self.filters.sharpen(image, radius=radius, amount=amount)

    def edge_map(self, image: np.ndarray) -> np.ndarray:
        return self.filters.sobel_edges(image)

    def alpha_over_rgba(self, background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        return self.filters.alpha_over_rgba(background, foreground)

    def invert_rgba(self, image: np.ndarray) -> np.ndarray:
        return self.filters.invert(image)

    def encode_gradient_signature(
        self,
        stops: Sequence[Sequence[float]],
        *,
        thresholds: tuple[float, float, float, float, float] = (0.08, 0.1, 0.1, 0.1, 0.08),
    ) -> TernaryGradientSignature:
        return self.ternary_gradient.encode_signature(stops, thresholds=thresholds)

    def contrastive_gradient_score(
        self,
        target_stops: Sequence[Sequence[float]],
        candidate_stops: Sequence[Sequence[float]],
        *,
        negative_examples: Sequence[Sequence[Sequence[float]]] = (),
    ) -> ContrastiveGradientScore:
        return self.ternary_gradient.contrastive_score(
            target_stops,
            candidate_stops,
            negative_examples=negative_examples,
        )

    def palette_to_gradient_stops(
        self,
        palette: Sequence[Sequence[float]],
    ) -> list[tuple[float, float, float, float, float]]:
        return self.ternary_palette.palette_to_stops(palette)

    def encode_palette_signature(
        self,
        palette: Sequence[Sequence[float]],
        *,
        thresholds: tuple[float, float, float, float, float] = (0.08, 0.1, 0.1, 0.1, 0.08),
    ) -> TernaryPaletteSignature:
        return self.ternary_palette.encode_signature(palette, thresholds=thresholds)

    def contrastive_palette_score(
        self,
        target_palette: Sequence[Sequence[float]],
        candidate_palette: Sequence[Sequence[float]],
        *,
        negative_examples: Sequence[Sequence[Sequence[float]]] = (),
    ) -> ContrastivePaletteScore:
        return self.ternary_palette.contrastive_score(
            target_palette,
            candidate_palette,
            negative_examples=negative_examples,
        )

    def linear_gradient_from_ternary_cascade(
        self,
        width: int,
        height: int,
        *,
        base_stop: Sequence[float],
        position_layers: Sequence[Sequence[int]],
        color_layers: Sequence[Sequence[Sequence[int]]],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        base_spacing: float | None = None,
        position_step: float = 0.2,
        color_step: float = 0.18,
        alpha_step: float = 0.12,
    ) -> np.ndarray:
        stops = self.ternary_gradient.compose_stops_from_cascade(
            base_stop=base_stop,
            position_layers=position_layers,
            color_layers=color_layers,
            base_spacing=base_spacing,
            position_step=position_step,
            color_step=color_step,
            alpha_step=alpha_step,
        )
        return self.linear_gradient(width, height, stops, x1=x1, y1=y1, x2=x2, y2=y2)


__all__ = [
    "ContrastiveGradientScore",
    "ContrastivePaletteScore",
    "DeviceCanvas",
    "DrawingEffects",
    "FilterKernels",
    "GradientKernels",
    "TernaryGradientLogic",
    "TernaryGradientSignature",
    "TernaryPaletteLogic",
    "TernaryPaletteSignature",
]
