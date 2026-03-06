from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def _reference_apply(
    base_frames: np.ndarray,
    overlay_rgba: np.ndarray,
    *,
    preset_key: str,
    time_points: np.ndarray,
    projection_weights: np.ndarray,
    normal_hint: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_frames, dtype=np.float32) / 255.0
    overlay_rgb = np.clip(np.asarray(overlay_rgba, dtype=np.float32)[..., :3], 0.0, 1.0)
    overlay_alpha = np.clip(np.asarray(overlay_rgba, dtype=np.float32)[..., 3:4], 0.0, 1.0)
    alpha_scale = np.clip(np.mean(np.asarray(projection_weights, dtype=np.float32), axis=0), 0.0, 1.0).astype(np.float32)
    if alpha_scale.ndim == 0:
        alpha_scale = alpha_scale.reshape(1, 1, 1, 1)
    else:
        alpha_scale = alpha_scale.reshape(1, 1, 1, -1)
    if preset_key == "ui_idle":
        phase = (0.5 + 0.5 * np.sin((time_points * np.pi * 2.0).reshape(-1, 1, 1, 1))).astype(np.float32)
        blend = 0.06 + 0.08 * phase
        out = base * (1.0 - blend * overlay_alpha * alpha_scale) + overlay_rgb[None, ...] * (blend * overlay_alpha * alpha_scale)
    elif preset_key == "ui_focus":
        luma = np.mean(overlay_rgb, axis=2, keepdims=True)
        grad_x = np.abs(np.diff(luma, axis=1, append=luma[:, -1:, :]))
        grad_y = np.abs(np.diff(luma, axis=0, append=luma[-1:, :, :]))
        edge = np.clip(grad_x + grad_y, 0.0, 1.0)
        pulse = (0.55 + 0.45 * np.sin((time_points * np.pi * 2.0).reshape(-1, 1, 1, 1))).astype(np.float32)
        out = np.clip(base * (1.0 + 0.1 * pulse) + edge[None, ...] * pulse * 0.35, 0.0, 1.0)
    elif preset_key == "world_breathe":
        bias = float(np.mean(np.abs(np.asarray(normal_hint, dtype=np.float32)))) if np.asarray(normal_hint).size else 0.0
        pulse = (0.5 + 0.5 * np.sin((time_points * np.pi * 2.0).reshape(-1, 1, 1, 1))).astype(np.float32)
        warmth = np.mean(overlay_rgb, axis=2, keepdims=True)
        out = np.clip(base * (0.92 + 0.12 * pulse) + warmth[None, ...] * (0.05 + 0.08 * bias * pulse), 0.0, 1.0)
    elif preset_key == "world_orbit":
        shifts = np.rint(time_points * float(base.shape[2]) * 0.25).astype(np.int32)
        rolled = np.stack([np.roll(overlay_rgb, int(shift), axis=1) for shift in shifts], axis=0)
        mix = 0.12 + 0.08 * (0.5 + 0.5 * np.cos((time_points * np.pi * 2.0).reshape(-1, 1, 1, 1)))
        out = np.clip(base * (1.0 - mix) + rolled * mix, 0.0, 1.0)
    else:
        out = base
    return np.clip(np.rint(out * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)


@pytest.mark.cuda
@pytest.mark.parametrize("preset_key", ["ui_idle", "ui_focus", "world_breathe", "world_orbit"])
def test_temporal_preset_kernel_matches_reference(preset_key: str):
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.temporal_preset_kernels import TemporalPresetKernels

    frames = np.asarray(
        np.arange(4 * 6 * 6 * 3, dtype=np.uint8).reshape(4, 6, 6, 3),
        dtype=np.uint8,
    )
    overlay = np.zeros((6, 6, 4), dtype=np.float32)
    overlay[..., 0] = np.linspace(0.0, 1.0, 6, dtype=np.float32)[None, :]
    overlay[..., 1] = np.linspace(1.0, 0.0, 6, dtype=np.float32)[:, None]
    overlay[..., 2] = 0.35
    overlay[..., 3] = 0.7
    time_points = np.asarray([0.0, 0.2, 0.4, 0.6], dtype=np.float32)
    projection_weights = np.asarray([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]], dtype=np.float32)
    normal_hint = np.linspace(-1.0, 1.0, 6 * 6, dtype=np.float32).reshape(6, 6)

    kernels = TemporalPresetKernels()
    out = kernels.apply_preset(
        frames,
        overlay,
        preset_key=preset_key,
        time_points=time_points,
        projection_weights=projection_weights,
        normal_hint=normal_hint,
    )
    ref = _reference_apply(
        frames,
        overlay,
        preset_key=preset_key,
        time_points=time_points,
        projection_weights=projection_weights,
        normal_hint=normal_hint,
    )

    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1)
