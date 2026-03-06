"""PTX-backed procedural temporal frame generation."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

TEMPORAL_FRAME_PTX = Path(__file__).parent.parent / "ptx" / "temporal_frame_ops.ptx"


def _as_seed(seed: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(seed, dtype=np.float32).reshape(-1))
    if arr.size == 0:
        raise ValueError("seed must not be empty")
    return arr


class TemporalFrameKernels:
    """PTX kernels for deterministic temporal frame synthesis."""

    def __init__(self) -> None:
        module = loader.load_module_from_file(str(TEMPORAL_FRAME_PTX))
        self.generate_kernel = loader.get_function(module, "generate_temporal_frames_kernel")

    def generate_frames(
        self,
        seed: np.ndarray,
        *,
        width: int,
        height: int,
        time_points: np.ndarray,
    ) -> np.ndarray:
        seed_arr = _as_seed(seed)
        times = np.ascontiguousarray(np.asarray(time_points, dtype=np.float32).reshape(-1))
        frame_count = int(times.shape[0])
        if frame_count == 0:
            return np.empty((0, int(height), int(width), 3), dtype=np.uint8)

        palette = self._palette_from_seed(seed_arr)
        pattern_selector = int(abs(float(seed_arr[0])) * 10.0) % 3
        freq = float(max(1.0, min(16.0, abs(float(seed_arr[0])) * 6.0 + 4.0)))
        scale = float(1.5 + abs(float(seed_arr[3] if seed_arr.size > 3 else seed_arr[0])) * 1.5)
        shift_x = float(seed_arr[4] if seed_arr.size > 4 else 0.0)
        shift_y = float(seed_arr[5] if seed_arr.size > 5 else 0.0)

        total = int(frame_count * width * height)
        out = np.empty((frame_count, int(height), int(width), 3), dtype=np.uint8)
        d_seed = loader.gpu_malloc(seed_arr.nbytes)
        d_palette = loader.gpu_malloc(palette.nbytes)
        d_times = loader.gpu_malloc(times.nbytes)
        d_out = loader.gpu_malloc(out.nbytes)
        try:
            loader.memcpy_htod(d_seed, seed_arr.ctypes.data_as(ctypes.c_void_p), seed_arr.nbytes)
            loader.memcpy_htod(d_palette, palette.ctypes.data_as(ctypes.c_void_p), palette.nbytes)
            loader.memcpy_htod(d_times, times.ctypes.data_as(ctypes.c_void_p), times.nbytes)
            block = (256, 1, 1)
            grid = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.generate_kernel,
                grid=grid,
                block=block,
                params=[
                    d_seed,
                    d_palette,
                    d_times,
                    d_out,
                    ctypes.c_int(int(seed_arr.size)),
                    ctypes.c_int(frame_count),
                    ctypes.c_int(int(width)),
                    ctypes.c_int(int(height)),
                    ctypes.c_int(pattern_selector),
                    ctypes.c_float(freq),
                    ctypes.c_float(scale),
                    ctypes.c_float(shift_x),
                    ctypes.c_float(shift_y),
                ],
            )
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_seed)
            loader.gpu_free(d_palette)
            loader.gpu_free(d_times)
            loader.gpu_free(d_out)

    def _palette_from_seed(self, seed: np.ndarray) -> np.ndarray:
        padded = seed
        if padded.size < 9:
            padded = np.pad(padded, (0, 9 - padded.size), constant_values=0.0)
        colors = []
        for i in range(0, 9, 3):
            base = padded[i:i + 3]
            color = (np.abs(np.sin(base * 12.9898 + i)) * 255.0).astype(np.float32, copy=False)
            colors.append(np.clip(color, 0.0, 255.0))
        return np.ascontiguousarray(np.stack(colors, axis=0).reshape(-1), dtype=np.float32)


__all__ = ["TemporalFrameKernels"]
