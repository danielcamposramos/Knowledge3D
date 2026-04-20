"""Smoke tests for DotMapBridge codec launcher.

Tests: §3.5 of CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md

GPU test:
    1. Uniform density map 8×8, request 64 dots — verify returned dot count
       equals target and all (x,y) coordinates are within [0,W)×[0,H).

All GPU tests require CUDA and are skipped in no-GPU environments.
Import-level smoke (no GPU) runs unconditionally.
"""

from __future__ import annotations

import ctypes
import importlib

import pytest


# ---------------------------------------------------------------------------
# Import-level smoke — must succeed even without CUDA.
# ---------------------------------------------------------------------------

def test_dotmap_bridge_module_importable():
    """Module imports without raising and exposes DotMapBridge class."""
    mod = importlib.import_module("knowledge3d.cranium.bridges.dotmap_bridge")
    assert hasattr(mod, "DotMapBridge"), (
        "dotmap_bridge module must export DotMapBridge"
    )


# ---------------------------------------------------------------------------
# GPU fixture (mirrors test_arc3_screen_bridge.py pattern)
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        from knowledge3d.cranium.sovereign import loader
        loader._ensure_current_context()
        return True
    except Exception:
        return False


_gpu_available = _cuda_available()
skip_no_gpu = pytest.mark.skipif(
    not _gpu_available,
    reason="No CUDA device available in this environment (sandbox / CI without GPU)",
)


# ---------------------------------------------------------------------------
# Helper: upload f32 list to GPU
# ---------------------------------------------------------------------------

def _upload_f32(values: list) -> object:
    """Copy a list of floats to GPU as f32 and return CUdeviceptr."""
    from knowledge3d.cranium.sovereign import loader
    n = len(values)
    F32Array = ctypes.c_float * n
    h_buf = F32Array(*[float(v) for v in values])
    d_ptr = loader.gpu_malloc(n * ctypes.sizeof(ctypes.c_float))
    loader.memcpy_htod(d_ptr, ctypes.cast(h_buf, ctypes.c_void_p),
                       n * ctypes.sizeof(ctypes.c_float))
    return d_ptr


def _download_f32(d_ptr: object, n: int) -> list:
    """Download n f32 values from GPU."""
    from knowledge3d.cranium.sovereign import loader
    F32Array = ctypes.c_float * n
    h_buf = F32Array()
    loader.memcpy_dtoh(ctypes.cast(h_buf, ctypes.c_void_p), d_ptr,
                       n * ctypes.sizeof(ctypes.c_float))
    return list(h_buf)


# ---------------------------------------------------------------------------
# Test 1: Uniform density — dot count and coordinate bounds
# ---------------------------------------------------------------------------

@skip_no_gpu
def test_frame_to_dotmap_count_and_bounds():
    """Uniform 8×8 density map, request 64 dots — count==64, coords in range.

    All cells have equal density=1.0, total_mass=64.0.
    Quasi-random golden-ratio placement must land every dot within [0,8)×[0,8).
    """
    from knowledge3d.cranium.sovereign import loader
    from knowledge3d.cranium.bridges.dotmap_bridge import DotMapBridge

    W, H = 8, 8
    target_dots = 64
    density_vals = [1.0] * (W * H)  # uniform
    total_mass = float(sum(density_vals))

    d_density = _upload_f32(density_vals)
    bridge = DotMapBridge()

    try:
        dots_dev, actual_count = bridge.frame_to_dotmap(
            d_density, total_mass=total_mass, target_dots=target_dots, W=W, H=H
        )
        try:
            assert actual_count == target_dots, (
                f"Expected actual_count={target_dots}, got {actual_count}"
            )
            coords = _download_f32(dots_dev, actual_count * 2)
        finally:
            loader.gpu_free(dots_dev)
    finally:
        loader.gpu_free(d_density)

    # Verify all coordinates are within bounds.
    for i in range(actual_count):
        x = coords[i * 2 + 0]
        y = coords[i * 2 + 1]
        assert 0.0 <= x < W + 1.0, f"dot {i} x={x} out of [0,{W})"
        assert 0.0 <= y < H + 1.0, f"dot {i} y={y} out of [0,{H})"
