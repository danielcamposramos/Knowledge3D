"""Canonical PTX-backed temporal preset application runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from knowledge3d.cranium.sovereign import loader

TEMPORAL_PRESET_PTX = Path(__file__).parent.parent / "ptx" / "temporal_preset_ops.ptx"

_PRESET_MODE = {
    "ui_idle": 0,
    "ui_focus": 1,
    "world_breathe": 2,
    "world_orbit": 3,
}


def _as_uint8_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise ValueError(f"expected [F,H,W,3] uint8 frames, got shape={arr.shape}")
    return arr


def _as_float32_rgba(image: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"expected [H,W,4] float32 RGBA image, got shape={arr.shape}")
    return arr


class TemporalPresetKernels:
    """PTX kernels for timeline preset application."""

    def __init__(self) -> None:
        module = loader.load_module_from_file(str(TEMPORAL_PRESET_PTX))
        self.apply_kernel = loader.get_function(module, "apply_temporal_preset_kernel")

    def apply_preset(
        self,
        base_frames: np.ndarray,
        overlay_rgba: np.ndarray,
        *,
        preset_key: str,
        time_points: np.ndarray,
        projection_weights: np.ndarray,
        normal_hint: np.ndarray,
    ) -> np.ndarray:
        key = str(preset_key).strip().lower()
        if key not in _PRESET_MODE:
            return _as_uint8_frames(base_frames)

        frames = _as_uint8_frames(base_frames)
        overlay = _as_float32_rgba(overlay_rgba)
        frame_count, height, width, _ = frames.shape
        if overlay.shape[0] != height or overlay.shape[1] != width:
            raise ValueError("overlay_rgba must already match base frame height/width")

        time_arr = np.ascontiguousarray(np.asarray(time_points, dtype=np.float32).reshape(frame_count))
        weights = np.ascontiguousarray(np.asarray(projection_weights, dtype=np.float32))
        alpha_scale = np.clip(np.mean(weights, axis=0), 0.0, 1.0).astype(np.float32, copy=False)
        if alpha_scale.ndim == 0:
            alpha_scale = np.asarray([float(alpha_scale)] * 3, dtype=np.float32)
        elif alpha_scale.shape[0] != 3:
            alpha_scale = np.resize(alpha_scale, 3).astype(np.float32, copy=False)

        overlay_rgb = np.ascontiguousarray(np.clip(overlay[..., :3], 0.0, 1.0).astype(np.float32, copy=False))
        overlay_alpha = np.ascontiguousarray(np.clip(overlay[..., 3], 0.0, 1.0).astype(np.float32, copy=False))
        luma = np.mean(overlay_rgb, axis=2, dtype=np.float32)
        grad_x = np.abs(np.diff(luma, axis=1, append=luma[:, -1:])).astype(np.float32, copy=False)
        grad_y = np.abs(np.diff(luma, axis=0, append=luma[-1:, :])).astype(np.float32, copy=False)
        edge = np.ascontiguousarray(np.clip(grad_x + grad_y, 0.0, 1.0).astype(np.float32, copy=False))
        warmth = np.ascontiguousarray(np.mean(overlay_rgb, axis=2, dtype=np.float32).astype(np.float32, copy=False))
        hint = np.asarray(normal_hint, dtype=np.float32)
        bias = float(np.mean(np.abs(hint))) if hint.size else 0.0
        shifts = np.ascontiguousarray(
            np.rint(time_arr * float(width) * 0.25).astype(np.int32, copy=False)
        )

        total = int(frame_count * height * width)
        if total == 0:
            return np.empty((0, height, width, 3), dtype=np.uint8)

        d_frames = loader.gpu_malloc(frames.nbytes)
        d_overlay_rgb = loader.gpu_malloc(overlay_rgb.nbytes)
        d_overlay_alpha = loader.gpu_malloc(overlay_alpha.nbytes)
        d_edge = loader.gpu_malloc(edge.nbytes)
        d_warmth = loader.gpu_malloc(warmth.nbytes)
        d_time = loader.gpu_malloc(time_arr.nbytes)
        d_shifts = loader.gpu_malloc(shifts.nbytes)
        d_out = loader.gpu_malloc(frames.nbytes)
        try:
            loader.memcpy_htod(d_frames, frames.ctypes.data_as(ctypes.c_void_p), frames.nbytes)
            loader.memcpy_htod(d_overlay_rgb, overlay_rgb.ctypes.data_as(ctypes.c_void_p), overlay_rgb.nbytes)
            loader.memcpy_htod(d_overlay_alpha, overlay_alpha.ctypes.data_as(ctypes.c_void_p), overlay_alpha.nbytes)
            loader.memcpy_htod(d_edge, edge.ctypes.data_as(ctypes.c_void_p), edge.nbytes)
            loader.memcpy_htod(d_warmth, warmth.ctypes.data_as(ctypes.c_void_p), warmth.nbytes)
            loader.memcpy_htod(d_time, time_arr.ctypes.data_as(ctypes.c_void_p), time_arr.nbytes)
            loader.memcpy_htod(d_shifts, shifts.ctypes.data_as(ctypes.c_void_p), shifts.nbytes)

            block = (256, 1, 1)
            grid = ((total + 255) // 256, 1, 1)
            loader.launch(
                self.apply_kernel,
                grid=grid,
                block=block,
                params=[
                    d_frames,
                    d_overlay_rgb,
                    d_overlay_alpha,
                    d_edge,
                    d_warmth,
                    d_time,
                    d_shifts,
                    d_out,
                    ctypes.c_int(frame_count),
                    ctypes.c_int(width),
                    ctypes.c_int(height),
                    ctypes.c_int(_PRESET_MODE[key]),
                    ctypes.c_float(float(alpha_scale[0])),
                    ctypes.c_float(float(alpha_scale[1])),
                    ctypes.c_float(float(alpha_scale[2])),
                    ctypes.c_float(float(bias)),
                ],
            )
            out = np.empty_like(frames)
            loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), d_out, out.nbytes)
            return out
        finally:
            loader.gpu_free(d_frames)
            loader.gpu_free(d_overlay_rgb)
            loader.gpu_free(d_overlay_alpha)
            loader.gpu_free(d_edge)
            loader.gpu_free(d_warmth)
            loader.gpu_free(d_time)
            loader.gpu_free(d_shifts)
            loader.gpu_free(d_out)


__all__ = ["TemporalPresetKernels"]
