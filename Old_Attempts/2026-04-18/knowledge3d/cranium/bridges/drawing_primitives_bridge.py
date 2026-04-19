"""
Drawing Engine Phases 2-4 bridge.

This bridge keeps the drawing hot path GPU-first by compiling/loading the live
CUDA kernels and routing supported primitives through the sovereign loader.
CPU execution is available only behind an explicit escape hatch for debugging.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_ATMOSPHERE_FOG,
    OP_BEZIER_EVAL,
    OP_BLEND_MULTIPLY,
    OP_BLEND_OVERLAY,
    OP_BLEND_SCREEN,
    OP_DRAW_DOT_EMIT,
    OP_DRAW_FIELD_COEF,
    OP_DRAW_VECTORDOTMAP_DECODE,
    OP_DRAW_VECTORDOTMAP_ENCODE,
    OP_LAYER_BLEND,
    OP_DRAW_REL_LINE,
    OP_SHAPE_INTERSECT,
    OP_SHAPE_SUBTRACT,
    OP_SHAPE_UNION,
    OP_VIGNETTE,
)
from knowledge3d.cranium.sovereign import loader


class DrawingPrimitivesBridge:
    """GPU-backed bridge for drawing primitives and post effects."""

    def __init__(self) -> None:
        self._compile_error: Optional[str] = None
        self.module = None
        self._ptx_path = Path(__file__).parent.parent / "ptx" / "drawing_primitives.ptx"
        self._source_path = Path(__file__).parent.parent / "kernels" / "drawing_primitives.cu"
        self._compile_kernels()
        self._load_kernels()

    def _compile_kernels(self) -> None:
        try:
            if not self._source_path.exists():
                self._compile_error = f"Missing drawing kernel source: {self._source_path}"
                return
            if self._ptx_path.exists() and self._ptx_path.stat().st_mtime >= self._source_path.stat().st_mtime:
                return
            self._ptx_path.parent.mkdir(parents=True, exist_ok=True)
            ptx_text = compile_cuda_file(self._source_path)
            self._ptx_path.write_text(ptx_text, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - environment dependent
            self._compile_error = str(exc)

    def _load_kernels(self) -> None:
        self.bezier_kernel = None
        self.shape_union_kernel = None
        self.shape_intersect_kernel = None
        self.shape_subtract_kernel = None
        self.rel_line_kernel = None
        self.field_coef_kernel = None
        self.dot_emit_kernel = None
        self.vdmap_encode_kernel = None
        self.vdmap_decode_kernel = None
        self.layer_blend_kernel = None
        self.atmosphere_fog_kernel = None
        self.vignette_kernel = None

        if not self._ptx_path.exists():
            if self._compile_error is None:
                self._compile_error = f"Compiled PTX missing: {self._ptx_path}"
            return

        def _load_functions() -> None:
            self.module = loader.load_module_from_file(str(self._ptx_path))
            self.bezier_kernel = loader.get_function(self.module, "bezier_eval_kernel")
            self.shape_union_kernel = loader.get_function(self.module, "shape_union_kernel")
            self.shape_intersect_kernel = loader.get_function(self.module, "shape_intersect_kernel")
            self.shape_subtract_kernel = loader.get_function(self.module, "shape_subtract_kernel")
            self.rel_line_kernel = loader.get_function(self.module, "rel_line_kernel")
            self.field_coef_kernel = loader.get_function(self.module, "field_coef_kernel")
            self.dot_emit_kernel = loader.get_function(self.module, "dot_emit_kernel")
            self.vdmap_encode_kernel = loader.get_function(self.module, "vectordotmap_encode_kernel")
            self.vdmap_decode_kernel = loader.get_function(self.module, "vectordotmap_decode_kernel")
            self.layer_blend_kernel = loader.get_function(self.module, "layer_blend_kernel")
            self.atmosphere_fog_kernel = loader.get_function(self.module, "atmosphere_fog_kernel")
            self.vignette_kernel = loader.get_function(self.module, "vignette_kernel")

        try:
            _load_functions()
        except Exception as exc:  # pragma: no cover - environment dependent
            try:
                ptx_text = compile_cuda_file(self._source_path)
                self._ptx_path.write_text(ptx_text, encoding="utf-8")
                _load_functions()
                self._compile_error = None
            except Exception as reload_exc:
                self._compile_error = str(reload_exc if str(reload_exc) else exc)
                self.module = None

    def _gpu_ready(self, *kernels: object) -> bool:
        return all(kernel is not None for kernel in kernels)

    def _cpu_fallback_enabled(self) -> bool:
        return os.environ.get("K3D_ALLOW_DRAWING_CPU_FALLBACK", "0") == "1"

    def _require_gpu(self, *kernels: object) -> bool:
        if self._gpu_ready(*kernels):
            return True
        if self._cpu_fallback_enabled():
            return False
        detail = self._compile_error or "drawing kernels unavailable"
        raise RuntimeError(f"Sovereign drawing kernel path unavailable: {detail}")

    def _copy_array_htod(self, array: np.ndarray):
        array = np.ascontiguousarray(array, dtype=np.float32)
        device_ptr = loader.gpu_malloc(max(1, int(array.nbytes)))
        loader.memcpy_htod(device_ptr, ctypes.c_void_p(array.ctypes.data), int(array.nbytes))
        return array, device_ptr

    def _copy_array_dtoh(self, array: np.ndarray, device_ptr) -> None:
        loader.memcpy_dtoh(ctypes.c_void_p(array.ctypes.data), device_ptr, int(array.nbytes))

    def _launch_1d(self, kernel, count: int, params: list[object], *, block_size: int = 256) -> None:
        grid_size = (max(1, int(count)) + block_size - 1) // block_size
        loader.launch(kernel, grid=(grid_size, 1, 1), block=(block_size, 1, 1), params=params)
        loader.synchronize()

    def _launch_2d(
        self,
        kernel,
        width: int,
        height: int,
        params: list[object],
        *,
        block_x: int = 16,
        block_y: int = 16,
    ) -> None:
        grid_x = (max(1, int(width)) + block_x - 1) // block_x
        grid_y = (max(1, int(height)) + block_y - 1) // block_y
        loader.launch(kernel, grid=(grid_x, grid_y, 1), block=(block_x, block_y, 1), params=params)
        loader.synchronize()

    def bezier_eval(self, t_values: np.ndarray, control_points: np.ndarray) -> np.ndarray:
        count = int(len(t_values))
        output_points = np.zeros((count, 2), dtype=np.float32)
        if not self._require_gpu(self.bezier_kernel):
            for i, t in enumerate(t_values):
                t2 = t * t
                t3 = t2 * t
                mt = 1.0 - t
                mt2 = mt * mt
                mt3 = mt2 * mt
                p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y = control_points
                output_points[i, 0] = mt3 * p0x + 3.0 * mt2 * t * p1x + 3.0 * mt * t2 * p2x + t3 * p3x
                output_points[i, 1] = mt3 * p0y + 3.0 * mt2 * t * p1y + 3.0 * mt * t2 * p2y + t3 * p3y
            return output_points

        t_values, t_gpu = self._copy_array_htod(np.asarray(t_values, dtype=np.float32))
        control_points, control_gpu = self._copy_array_htod(np.asarray(control_points, dtype=np.float32))
        output_gpu = loader.gpu_malloc(max(1, int(output_points.nbytes)))
        try:
            self._launch_1d(
                self.bezier_kernel,
                count,
                [
                    ctypes.c_uint64(int(t_gpu.value)),
                    ctypes.c_uint64(int(control_gpu.value)),
                    ctypes.c_uint64(int(output_gpu.value)),
                    ctypes.c_int32(count),
                ],
            )
            self._copy_array_dtoh(output_points, output_gpu)
            return output_points
        finally:
            loader.gpu_free(t_gpu)
            loader.gpu_free(control_gpu)
            loader.gpu_free(output_gpu)

    def _shape_operation_cpu(self, shape_a: np.ndarray, shape_b: np.ndarray, operation: str) -> np.ndarray:
        result = np.zeros_like(shape_a, dtype=np.float32)
        for i in range(len(shape_a)):
            if operation == "union":
                result[i, 0] = min(shape_a[i, 0], shape_b[i, 0])
                result[i, 1] = min(shape_a[i, 1], shape_b[i, 1])
                result[i, 2] = max(shape_a[i, 2], shape_b[i, 2])
                result[i, 3] = max(shape_a[i, 3], shape_b[i, 3])
            elif operation == "intersect":
                result[i, 0] = max(shape_a[i, 0], shape_b[i, 0])
                result[i, 1] = max(shape_a[i, 1], shape_b[i, 1])
                result[i, 2] = min(shape_a[i, 2], shape_b[i, 2])
                result[i, 3] = min(shape_a[i, 3], shape_b[i, 3])
            elif operation == "subtract":
                result[i, 0] = max(shape_a[i, 0], shape_b[i, 2])
                result[i, 1] = max(shape_a[i, 1], shape_b[i, 3])
                result[i, 2] = min(shape_a[i, 2], shape_b[i, 0])
                result[i, 3] = min(shape_a[i, 3], shape_b[i, 1])
                if result[i, 0] > result[i, 2]:
                    result[i, 0] = result[i, 2] = 0.0
                if result[i, 1] > result[i, 3]:
                    result[i, 1] = result[i, 3] = 0.0
        return result

    def _shape_gpu(self, kernel, shape_a: np.ndarray, shape_b: np.ndarray, operation: str) -> np.ndarray:
        count = int(len(shape_a))
        result = np.zeros_like(shape_a, dtype=np.float32)
        if not self._require_gpu(kernel):
            return self._shape_operation_cpu(shape_a, shape_b, operation)

        shape_a, d_shape_a = self._copy_array_htod(shape_a)
        shape_b, d_shape_b = self._copy_array_htod(shape_b)
        d_result = loader.gpu_malloc(max(1, int(result.nbytes)))
        try:
            self._launch_1d(
                kernel,
                count,
                [
                    ctypes.c_uint64(int(d_shape_a.value)),
                    ctypes.c_uint64(int(d_shape_b.value)),
                    ctypes.c_uint64(int(d_result.value)),
                    ctypes.c_int32(count),
                ],
            )
            self._copy_array_dtoh(result, d_result)
            return result
        finally:
            loader.gpu_free(d_shape_a)
            loader.gpu_free(d_shape_b)
            loader.gpu_free(d_result)

    def shape_union(self, shape_a: np.ndarray, shape_b: np.ndarray) -> np.ndarray:
        return self._shape_gpu(self.shape_union_kernel, shape_a, shape_b, "union")

    def shape_intersect(self, shape_a: np.ndarray, shape_b: np.ndarray) -> np.ndarray:
        return self._shape_gpu(self.shape_intersect_kernel, shape_a, shape_b, "intersect")

    def shape_subtract(self, shape_a: np.ndarray, shape_b: np.ndarray) -> np.ndarray:
        return self._shape_gpu(self.shape_subtract_kernel, shape_a, shape_b, "subtract")

    def rel_line(self, start_points: np.ndarray, end_points: np.ndarray, canvas_width: float, canvas_height: float) -> np.ndarray:
        count = int(len(start_points))
        output_lines = np.zeros((count, 4), dtype=np.float32)
        if not self._require_gpu(self.rel_line_kernel):
            for i in range(count):
                x0_frac, y0_frac = start_points[i]
                x1_frac, y1_frac = end_points[i]
                output_lines[i, 0] = x0_frac * canvas_width
                output_lines[i, 1] = y0_frac * canvas_height
                output_lines[i, 2] = x1_frac * canvas_width
                output_lines[i, 3] = y1_frac * canvas_height
            return output_lines

        start_points, d_start = self._copy_array_htod(start_points)
        end_points, d_end = self._copy_array_htod(end_points)
        d_output = loader.gpu_malloc(max(1, int(output_lines.nbytes)))
        try:
            self._launch_1d(
                self.rel_line_kernel,
                count,
                [
                    ctypes.c_uint64(int(d_start.value)),
                    ctypes.c_uint64(int(d_end.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_int32(count),
                    ctypes.c_float(float(canvas_width)),
                    ctypes.c_float(float(canvas_height)),
                ],
            )
            self._copy_array_dtoh(output_lines, d_output)
            return output_lines
        finally:
            loader.gpu_free(d_start)
            loader.gpu_free(d_end)
            loader.gpu_free(d_output)

    def field_coef(self, coefficients: np.ndarray, positions: np.ndarray) -> np.ndarray:
        count = int(len(positions))
        field_values = np.zeros(count, dtype=np.float32)
        if not self._require_gpu(self.field_coef_kernel):
            for i in range(count):
                x, y = positions[i]
                x2, y2 = x * x, y * y
                x3, y3 = x2 * x, y2 * y
                xy = x * y
                field_values[i] = (
                    coefficients[0] + coefficients[1] * x + coefficients[2] * y +
                    coefficients[3] * x2 + coefficients[4] * y2 + coefficients[5] * xy +
                    coefficients[6] * x3 + coefficients[7] * y3
                )
            return field_values

        coefficients, d_coeffs = self._copy_array_htod(np.asarray(coefficients, dtype=np.float32))
        positions, d_positions = self._copy_array_htod(positions)
        d_output = loader.gpu_malloc(max(1, int(field_values.nbytes)))
        try:
            self._launch_1d(
                self.field_coef_kernel,
                count,
                [
                    ctypes.c_uint64(int(d_coeffs.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_uint64(int(d_positions.value)),
                    ctypes.c_int32(count),
                ],
            )
            self._copy_array_dtoh(field_values, d_output)
            return field_values
        finally:
            loader.gpu_free(d_coeffs)
            loader.gpu_free(d_positions)
            loader.gpu_free(d_output)

    def dot_emit(self, positions: np.ndarray, field_values: np.ndarray, base_radius: float = 2.0, intensity_scale: float = 1.0) -> np.ndarray:
        count = int(len(positions))
        output_dots = np.zeros((count, 4), dtype=np.float32)
        if not self._require_gpu(self.dot_emit_kernel):
            for i in range(count):
                x, y = positions[i]
                field = field_values[i]
                radius = base_radius * (1.0 + field * 0.5)
                intensity = max(0.0, min(field * intensity_scale, 1.0))
                output_dots[i] = [x, y, intensity, radius]
            return output_dots

        positions, d_positions = self._copy_array_htod(positions)
        field_values, d_field = self._copy_array_htod(np.asarray(field_values, dtype=np.float32))
        d_output = loader.gpu_malloc(max(1, int(output_dots.nbytes)))
        try:
            self._launch_1d(
                self.dot_emit_kernel,
                count,
                [
                    ctypes.c_uint64(int(d_positions.value)),
                    ctypes.c_uint64(int(d_field.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_int32(count),
                    ctypes.c_float(float(base_radius)),
                    ctypes.c_float(float(intensity_scale)),
                ],
            )
            self._copy_array_dtoh(output_dots, d_output)
            return output_dots
        finally:
            loader.gpu_free(d_positions)
            loader.gpu_free(d_field)
            loader.gpu_free(d_output)

    def vectordotmap_encode(self, pixels: np.ndarray, block_size: int = 8) -> np.ndarray:
        height, width, _channels = pixels.shape
        blocks_x = (width + block_size - 1) // block_size
        blocks_y = (height + block_size - 1) // block_size
        field_coeffs = np.zeros((blocks_y, blocks_x, 8), dtype=np.float32)
        if not self._require_gpu(self.vdmap_encode_kernel):
            for by in range(blocks_y):
                for bx in range(blocks_x):
                    start_x = bx * block_size
                    start_y = by * block_size
                    end_x = min(start_x + block_size, width)
                    end_y = min(start_y + block_size, height)
                    block = pixels[start_y:end_y, start_x:end_x, :]
                    coeffs = field_coeffs[by, bx]
                    coeffs[0] = float(np.mean(block[:, :, 0]))
                    coeffs[1] = 0.1
                    coeffs[2] = 0.1
                    coeffs[3] = 0.05
                    coeffs[4] = 0.05
                    coeffs[5] = 0.02
                    coeffs[6] = 0.01
                    coeffs[7] = 0.01
            return field_coeffs

        if block_size > 16:
            raise ValueError("VectorDotMap GPU encode currently supports block_size <= 16")
        pixels, d_pixels = self._copy_array_htod(pixels)
        d_coeffs = loader.gpu_malloc(max(1, int(field_coeffs.nbytes)))
        try:
            self._launch_2d(
                self.vdmap_encode_kernel,
                blocks_x,
                blocks_y,
                [
                    ctypes.c_uint64(int(d_pixels.value)),
                    ctypes.c_uint64(int(d_coeffs.value)),
                    ctypes.c_int32(width),
                    ctypes.c_int32(height),
                    ctypes.c_int32(block_size),
                ],
                block_x=min(block_size, 16),
                block_y=min(block_size, 16),
            )
            self._copy_array_dtoh(field_coeffs, d_coeffs)
            return field_coeffs
        finally:
            loader.gpu_free(d_pixels)
            loader.gpu_free(d_coeffs)

    def vectordotmap_decode(self, field_coeffs: np.ndarray, width: int, height: int, block_size: int = 8) -> np.ndarray:
        pixels = np.zeros((height, width, 4), dtype=np.float32)
        if not self._require_gpu(self.vdmap_decode_kernel):
            blocks_y, blocks_x = field_coeffs.shape[:2]
            for y in range(height):
                for x in range(width):
                    block_x = min(x // block_size, blocks_x - 1)
                    block_y = min(y // block_size, blocks_y - 1)
                    coeffs = field_coeffs[block_y, block_x]
                    local_x = x % block_size
                    local_y = y % block_size
                    u = (2.0 * local_x) / block_size - 1.0
                    v = (2.0 * local_y) / block_size - 1.0
                    u2, v2 = u * u, v * v
                    u3, v3 = u2 * u, v2 * v
                    uv = u * v
                    field = coeffs[0] + coeffs[1] * u + coeffs[2] * v + coeffs[3] * u2 + coeffs[4] * v2 + coeffs[5] * uv + coeffs[6] * u3 + coeffs[7] * v3
                    intensity = max(0.0, min(float(field), 1.0))
                    pixels[y, x] = [intensity, intensity * 0.8, intensity * 0.6, 1.0]
            return pixels

        field_coeffs, d_coeffs = self._copy_array_htod(field_coeffs)
        d_pixels = loader.gpu_malloc(max(1, int(pixels.nbytes)))
        try:
            self._launch_2d(
                self.vdmap_decode_kernel,
                width,
                height,
                [
                    ctypes.c_uint64(int(d_coeffs.value)),
                    ctypes.c_uint64(int(d_pixels.value)),
                    ctypes.c_int32(width),
                    ctypes.c_int32(height),
                    ctypes.c_int32(block_size),
                ],
            )
            self._copy_array_dtoh(pixels, d_pixels)
            return pixels
        finally:
            loader.gpu_free(d_coeffs)
            loader.gpu_free(d_pixels)

    def layer_blend(self, layer_a: np.ndarray, layer_b: np.ndarray, blend_mode: int = 0) -> np.ndarray:
        height, width, _channels = layer_a.shape
        output = np.zeros_like(layer_a, dtype=np.float32)
        if not self._require_gpu(self.layer_blend_kernel):
            for y in range(height):
                for x in range(width):
                    r_a, g_a, b_a, a_a = layer_a[y, x]
                    r_b, g_b, b_b, a_b = layer_b[y, x]
                    if blend_mode == 0:
                        a_out = a_b + a_a * (1.0 - a_b)
                        if a_out > 0:
                            r_out = (r_b * a_b + r_a * a_a * (1.0 - a_b)) / a_out
                            g_out = (g_b * a_b + g_a * a_a * (1.0 - a_b)) / a_out
                            b_out = (b_b * a_b + b_a * a_a * (1.0 - a_b)) / a_out
                        else:
                            r_out = g_out = b_out = 0.0
                    elif blend_mode == 1:
                        r_out = r_a * r_b
                        g_out = g_a * g_b
                        b_out = b_a * b_b
                        a_out = min(a_a + a_b, 1.0)
                    elif blend_mode == 2:
                        r_out = 1.0 - (1.0 - r_a) * (1.0 - r_b)
                        g_out = 1.0 - (1.0 - g_a) * (1.0 - g_b)
                        b_out = 1.0 - (1.0 - b_a) * (1.0 - b_b)
                        a_out = min(a_a + a_b, 1.0)
                    elif blend_mode == 3:
                        r_out = 2.0 * r_a * r_b if r_a < 0.5 else 1.0 - 2.0 * (1.0 - r_a) * (1.0 - r_b)
                        g_out = 2.0 * g_a * g_b if g_a < 0.5 else 1.0 - 2.0 * (1.0 - g_a) * (1.0 - g_b)
                        b_out = 2.0 * b_a * b_b if b_a < 0.5 else 1.0 - 2.0 * (1.0 - b_a) * (1.0 - b_b)
                        a_out = min(a_a + a_b, 1.0)
                    else:
                        r_out, g_out, b_out, a_out = r_a, g_a, b_a, a_a
                    output[y, x] = [r_out, g_out, b_out, a_out]
            return output

        pixel_count = height * width
        layer_a, d_layer_a = self._copy_array_htod(layer_a)
        layer_b, d_layer_b = self._copy_array_htod(layer_b)
        d_output = loader.gpu_malloc(max(1, int(output.nbytes)))
        try:
            self._launch_1d(
                self.layer_blend_kernel,
                pixel_count,
                [
                    ctypes.c_uint64(int(d_layer_a.value)),
                    ctypes.c_uint64(int(d_layer_b.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_int32(pixel_count),
                    ctypes.c_int32(int(blend_mode)),
                ],
            )
            self._copy_array_dtoh(output, d_output)
            return output
        finally:
            loader.gpu_free(d_layer_a)
            loader.gpu_free(d_layer_b)
            loader.gpu_free(d_output)

    def atmosphere_fog(self, scene_color: np.ndarray, fog_density: float, fog_color: Tuple[float, float, float] = (0.8, 0.9, 1.0)) -> np.ndarray:
        height, width, _channels = scene_color.shape
        output = np.zeros_like(scene_color, dtype=np.float32)
        fog_r, fog_g, fog_b = fog_color
        if not self._require_gpu(self.atmosphere_fog_kernel):
            fog_amount = min(float(fog_density), 1.0)
            for y in range(height):
                for x in range(width):
                    r, g, b, a = scene_color[y, x]
                    output[y, x] = [
                        r * (1.0 - fog_amount) + fog_r * fog_amount,
                        g * (1.0 - fog_amount) + fog_g * fog_amount,
                        b * (1.0 - fog_amount) + fog_b * fog_amount,
                        a,
                    ]
            return output

        pixel_count = height * width
        scene_color, d_scene = self._copy_array_htod(scene_color)
        d_output = loader.gpu_malloc(max(1, int(output.nbytes)))
        try:
            self._launch_1d(
                self.atmosphere_fog_kernel,
                pixel_count,
                [
                    ctypes.c_uint64(int(d_scene.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_int32(pixel_count),
                    ctypes.c_float(float(fog_density)),
                    ctypes.c_float(float(fog_r)),
                    ctypes.c_float(float(fog_g)),
                    ctypes.c_float(float(fog_b)),
                ],
            )
            self._copy_array_dtoh(output, d_output)
            return output
        finally:
            loader.gpu_free(d_scene)
            loader.gpu_free(d_output)

    def vignette(self, scene_color: np.ndarray, strength: float = 0.3, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        height, width, _channels = scene_color.shape
        output = np.zeros_like(scene_color, dtype=np.float32)
        if center is None:
            center_x, center_y = width / 2.0, height / 2.0
        else:
            center_x, center_y = center

        if not self._require_gpu(self.vignette_kernel):
            for y in range(height):
                for x in range(width):
                    dx = (x - center_x) / width
                    dy = (y - center_y) / height
                    dist = float(np.sqrt(dx * dx + dy * dy))
                    vignette = 1.0 - strength * dist
                    r, g, b, a = scene_color[y, x]
                    output[y, x] = [r * vignette, g * vignette, b * vignette, a]
            return output

        scene_color, d_scene = self._copy_array_htod(scene_color)
        d_output = loader.gpu_malloc(max(1, int(output.nbytes)))
        try:
            self._launch_2d(
                self.vignette_kernel,
                width,
                height,
                [
                    ctypes.c_uint64(int(d_scene.value)),
                    ctypes.c_uint64(int(d_output.value)),
                    ctypes.c_int32(width),
                    ctypes.c_int32(height),
                    ctypes.c_float(float(strength)),
                    ctypes.c_float(float(center_x)),
                    ctypes.c_float(float(center_y)),
                ],
            )
            self._copy_array_dtoh(output, d_output)
            return output
        finally:
            loader.gpu_free(d_scene)
            loader.gpu_free(d_output)


class DrawingPrimitivesEngine:
    """RPN integration layer for the drawing primitives bridge."""

    def __init__(self) -> None:
        self.bridge = DrawingPrimitivesBridge()
        self.blend_mode = 0
        self.pending_fog: Optional[Tuple[float, float, float, float]] = None
        self.pending_vignette: Optional[Tuple[float, float, float]] = None
        self._bound_layer_a: Optional[np.ndarray] = None
        self._bound_layer_b: Optional[np.ndarray] = None
        self._last_layer_output: Optional[np.ndarray] = None
        self._bound_scene: Optional[np.ndarray] = None
        self._last_scene_output: Optional[np.ndarray] = None

    def bind_layers(self, layer_a: np.ndarray, layer_b: np.ndarray) -> None:
        self._bound_layer_a = np.ascontiguousarray(layer_a, dtype=np.float32)
        self._bound_layer_b = np.ascontiguousarray(layer_b, dtype=np.float32)
        self._last_layer_output = None

    def get_last_layer_output(self) -> Optional[np.ndarray]:
        return None if self._last_layer_output is None else np.array(self._last_layer_output, copy=True)

    def bind_scene(self, scene_color: np.ndarray) -> None:
        self._bound_scene = np.ascontiguousarray(scene_color, dtype=np.float32)
        self._last_scene_output = None

    def get_last_scene_output(self) -> Optional[np.ndarray]:
        return None if self._last_scene_output is None else np.array(self._last_scene_output, copy=True)

    def blend_layers(self, layer_a: np.ndarray, layer_b: np.ndarray) -> np.ndarray:
        blended = self.bridge.layer_blend(layer_a, layer_b, self.blend_mode)
        self._last_layer_output = blended
        return blended

    def apply_scene_effects(self, scene_color: np.ndarray) -> np.ndarray:
        output = scene_color
        if self.pending_fog is not None:
            fog_density, fog_r, fog_g, fog_b = self.pending_fog
            output = self.bridge.atmosphere_fog(output, fog_density, (fog_r, fog_g, fog_b))
        if self.pending_vignette is not None:
            strength, center_x, center_y = self.pending_vignette
            output = self.bridge.vignette(output, strength, (center_x, center_y))
        self._last_scene_output = output
        return output

    def execute_opcode(self, opcode: int, stack: List[float]) -> List[float]:
        try:
            if opcode == OP_BEZIER_EVAL:
                if len(stack) < 9:
                    raise ValueError("Insufficient operands for BEZIER_EVAL")
                p3y = stack.pop()
                p3x = stack.pop()
                p2y = stack.pop()
                p2x = stack.pop()
                p1y = stack.pop()
                p1x = stack.pop()
                p0y = stack.pop()
                p0x = stack.pop()
                t = stack.pop()
                result = self.bridge.bezier_eval(
                    np.array([t], dtype=np.float32),
                    np.array([p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y], dtype=np.float32),
                )
                stack.extend([float(result[0, 0]), float(result[0, 1])])
            elif opcode == OP_SHAPE_UNION:
                if len(stack) < 8:
                    raise ValueError("Insufficient operands for SHAPE_UNION")
                shape_b = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                shape_a = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                result = self.bridge.shape_union(shape_a, shape_b)
                stack.extend(result[0].tolist())
            elif opcode == OP_SHAPE_INTERSECT:
                if len(stack) < 8:
                    raise ValueError("Insufficient operands for SHAPE_INTERSECT")
                shape_b = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                shape_a = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                result = self.bridge.shape_intersect(shape_a, shape_b)
                stack.extend(result[0].tolist())
            elif opcode == OP_SHAPE_SUBTRACT:
                if len(stack) < 8:
                    raise ValueError("Insufficient operands for SHAPE_SUBTRACT")
                shape_b = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                shape_a = np.array([stack.pop() for _ in range(4)][::-1], dtype=np.float32).reshape(1, -1)
                result = self.bridge.shape_subtract(shape_a, shape_b)
                stack.extend(result[0].tolist())
            elif opcode == OP_DRAW_REL_LINE:
                if len(stack) < 6:
                    raise ValueError("Insufficient operands for REL_LINE")
                canvas_h = stack.pop()
                canvas_w = stack.pop()
                y1_frac = stack.pop()
                x1_frac = stack.pop()
                y0_frac = stack.pop()
                x0_frac = stack.pop()
                result = self.bridge.rel_line(
                    np.array([[x0_frac, y0_frac]], dtype=np.float32),
                    np.array([[x1_frac, y1_frac]], dtype=np.float32),
                    canvas_w,
                    canvas_h,
                )
                stack.extend(result[0].tolist())
            elif opcode == OP_DRAW_FIELD_COEF:
                if len(stack) < 10:
                    raise ValueError("Insufficient operands for FIELD_COEF")
                y = stack.pop()
                x = stack.pop()
                coefficients = np.array([stack.pop() for _ in range(8)][::-1], dtype=np.float32)
                result = self.bridge.field_coef(coefficients, np.array([[x, y]], dtype=np.float32))
                stack.append(float(result[0]))
            elif opcode == OP_DRAW_DOT_EMIT:
                if len(stack) < 5:
                    raise ValueError("Insufficient operands for DOT_EMIT")
                intensity_scale = stack.pop()
                base_radius = stack.pop()
                field = stack.pop()
                y = stack.pop()
                x = stack.pop()
                result = self.bridge.dot_emit(
                    np.array([[x, y]], dtype=np.float32),
                    np.array([field], dtype=np.float32),
                    base_radius,
                    intensity_scale,
                )
                stack.extend(result[0].tolist())
            elif opcode == OP_LAYER_BLEND:
                if len(stack) < 1:
                    raise ValueError("Insufficient operands for LAYER_BLEND")
                requested_mode = max(0, int(stack.pop()))
                self.blend_mode = max(0, requested_mode - 1)
                if self.blend_mode not in (0, 1, 2, 3):
                    raise ValueError(f"Unsupported blend mode {self.blend_mode}")
                if self._bound_layer_a is not None and self._bound_layer_b is not None:
                    self._last_layer_output = self.bridge.layer_blend(
                        self._bound_layer_a,
                        self._bound_layer_b,
                        self.blend_mode,
                    )
            elif opcode in (OP_BLEND_MULTIPLY, OP_BLEND_SCREEN, OP_BLEND_OVERLAY):
                self.blend_mode = {
                    OP_BLEND_MULTIPLY: 1,
                    OP_BLEND_SCREEN: 2,
                    OP_BLEND_OVERLAY: 3,
                }[opcode]
                if self._bound_layer_a is not None and self._bound_layer_b is not None:
                    self._last_layer_output = self.bridge.layer_blend(
                        self._bound_layer_a,
                        self._bound_layer_b,
                        self.blend_mode,
                    )
            elif opcode == OP_ATMOSPHERE_FOG:
                if len(stack) < 4:
                    raise ValueError("Insufficient operands for ATMOSPHERE_FOG")
                fog_density = stack.pop()
                fog_b = stack.pop()
                fog_g = stack.pop()
                fog_r = stack.pop()
                self.pending_fog = (float(fog_density), float(fog_r), float(fog_g), float(fog_b))
                if self._bound_scene is not None:
                    self._last_scene_output = self.bridge.atmosphere_fog(
                        self._bound_scene,
                        float(fog_density),
                        (float(fog_r), float(fog_g), float(fog_b)),
                    )
            elif opcode == OP_VIGNETTE:
                if len(stack) < 3:
                    raise ValueError("Insufficient operands for VIGNETTE")
                center_y = stack.pop()
                center_x = stack.pop()
                strength = stack.pop()
                self.pending_vignette = (float(strength), float(center_x), float(center_y))
                if self._bound_scene is not None:
                    self._last_scene_output = self.bridge.vignette(
                        self._bound_scene,
                        float(strength),
                        (float(center_x), float(center_y)),
                    )
            else:
                raise ValueError(f"Unsupported drawing opcode: 0x{opcode:02X}")
        except Exception as exc:
            raise RuntimeError(f"Drawing primitive execution failed: {exc}") from exc
        return stack

    def bind_layers(self, layer_a: np.ndarray, layer_b: np.ndarray) -> None:
        self._bound_layer_a = np.ascontiguousarray(layer_a, dtype=np.float32)
        self._bound_layer_b = np.ascontiguousarray(layer_b, dtype=np.float32)
        self._last_layer_output = None

    def get_last_layer_output(self) -> Optional[np.ndarray]:
        return None if self._last_layer_output is None else np.array(self._last_layer_output, copy=True)

    def bind_scene(self, scene_color: np.ndarray) -> None:
        self._bound_scene = np.ascontiguousarray(scene_color, dtype=np.float32)
        self._last_scene_output = None

    def get_last_scene_output(self) -> Optional[np.ndarray]:
        return None if self._last_scene_output is None else np.array(self._last_scene_output, copy=True)


def create_drawing_engine() -> DrawingPrimitivesEngine:
    return DrawingPrimitivesEngine()
