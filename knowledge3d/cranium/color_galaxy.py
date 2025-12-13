"""
Color Galaxy — sovereign GPU color conversions and named color utilities.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, List, Tuple

from knowledge3d.cranium.sovereign import loader

# Minimal CSS named colors (extend as needed)
CSS_NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
}


class ColorGalaxy:
    """GPU-native color space operations."""

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent / "ptx" / "color_convert.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"color_convert.ptx not found: {ptx_path}")
        module = loader.load_module_from_file(str(ptx_path))
        self.rgb_to_lab_kernel = loader.get_function(module, "rgb_to_lab_batch")
        self.lab_to_rgb_kernel = loader.get_function(module, "lab_to_rgb_batch")
        self.rgb_to_cmyk_kernel = loader.get_function(module, "rgb_to_cmyk_batch")
        self.rgb_to_hsv_kernel = loader.get_function(module, "rgb_to_hsv_batch")
        self.delta_e_kernel = loader.get_function(module, "delta_e_batch")
        self._named_colors_lab: Dict[str, Tuple[float, float, float]] = {}
        self._init_named_colors()

    def _init_named_colors(self) -> None:
        """Precompute Lab values for named colors."""
        for name, (r, g, b) in CSS_NAMED_COLORS.items():
            lab = self.rgb_to_lab([(r / 255.0, g / 255.0, b / 255.0)])[0]
            self._named_colors_lab[name] = lab

    def _gpu_convert(self, flat_input, n, in_channels, out_channels, kernel, extra_params=None) -> List[Tuple[float, ...]]:
        extra_params = extra_params or []
        in_buf = (ctypes.c_float * (n * in_channels))(*flat_input)
        out_buf = (ctypes.c_float * (n * out_channels))()
        d_in = loader.gpu_malloc(ctypes.sizeof(in_buf))
        d_out = loader.gpu_malloc(ctypes.sizeof(out_buf))
        try:
            loader.memcpy_htod(d_in, ctypes.cast(in_buf, ctypes.c_void_p), ctypes.sizeof(in_buf))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            params = [ctypes.c_uint64(d_in.value), ctypes.c_uint64(d_out.value)]
            params.extend(extra_params)
            params.append(ctypes.c_int(n))
            loader.launch(kernel, grid=(grid_x, 1, 1), block=block, params=params)
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out_buf, ctypes.c_void_p), d_out, ctypes.sizeof(out_buf))
            return [tuple(out_buf[i * out_channels : (i + 1) * out_channels]) for i in range(n)]
        finally:
            loader.gpu_free(d_in)
            loader.gpu_free(d_out)

    def rgb_to_lab(self, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        n = len(colors)
        if n == 0:
            return []
        flat = []
        for r, g, b in colors:
            flat.extend([r, g, b])
        return self._gpu_convert(flat, n, 3, 3, self.rgb_to_lab_kernel)

    def lab_to_rgb(self, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        n = len(colors)
        if n == 0:
            return []
        flat = []
        for L, a, b in colors:
            flat.extend([L, a, b])
        return self._gpu_convert(flat, n, 3, 3, self.lab_to_rgb_kernel)

    def rgb_to_cmyk(self, colors: List[Tuple[float, float, float]], gcr: float = 1.0) -> List[Tuple[float, float, float, float]]:
        n = len(colors)
        if n == 0:
            return []
        flat = []
        for r, g, b in colors:
            flat.extend([r, g, b])
        gcr_param = ctypes.c_float(gcr)
        return self._gpu_convert(flat, n, 3, 4, self.rgb_to_cmyk_kernel, [gcr_param])

    def rgb_to_hsv(self, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        n = len(colors)
        if n == 0:
            return []
        flat = []
        for r, g, b in colors:
            flat.extend([r, g, b])
        return self._gpu_convert(flat, n, 3, 3, self.rgb_to_hsv_kernel)

    def delta_e(self, lab_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]) -> List[float]:
        n = len(lab_pairs)
        if n == 0:
            return []
        flat1: List[float] = []
        flat2: List[float] = []
        for a, b in lab_pairs:
            flat1.extend([a[0], a[1], a[2]])
            flat2.extend([b[0], b[1], b[2]])
        in1 = (ctypes.c_float * (n * 3))(*flat1)
        in2 = (ctypes.c_float * (n * 3))(*flat2)
        out = (ctypes.c_float * n)()
        d1 = loader.gpu_malloc(ctypes.sizeof(in1))
        d2 = loader.gpu_malloc(ctypes.sizeof(in2))
        d_out = loader.gpu_malloc(ctypes.sizeof(out))
        try:
            loader.memcpy_htod(d1, ctypes.cast(in1, ctypes.c_void_p), ctypes.sizeof(in1))
            loader.memcpy_htod(d2, ctypes.cast(in2, ctypes.c_void_p), ctypes.sizeof(in2))
            block = (256, 1, 1)
            grid_x = (n + block[0] - 1) // block[0]
            loader.launch(
                self.delta_e_kernel,
                grid=(grid_x, 1, 1),
                block=block,
                params=[ctypes.c_uint64(d1.value), ctypes.c_uint64(d2.value), ctypes.c_uint64(d_out.value), ctypes.c_int(n)],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(out, ctypes.c_void_p), d_out, ctypes.sizeof(out))
            return [float(out[i]) for i in range(n)]
        finally:
            loader.gpu_free(d1)
            loader.gpu_free(d2)
            loader.gpu_free(d_out)

    def find_closest_named(self, color_lab: Tuple[float, float, float]) -> str:
        """CPU-side DeltaE search over cached named colors."""
        best = None
        best_de = float("inf")
        for name, lab in self._named_colors_lab.items():
            dL = color_lab[0] - lab[0]
            da = color_lab[1] - lab[1]
            db = color_lab[2] - lab[2]
            de = (dL * dL + da * da + db * db) ** 0.5
            if de < best_de:
                best_de = de
                best = name
        return best or "unknown"


__all__ = ["ColorGalaxy", "CSS_NAMED_COLORS"]
