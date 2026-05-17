"""Smoke tests for Arc3ScreenBridge codec launcher.

Tests: §3.4 of CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md

Three GPU tests:
    1. Palette upload + 4×4 frame decode — verify RGBA bytes match palette entries.
    2. Click invert — rect (0,0,64,64), grid 4×4, click (16,16) → cell (1,1).
    3. Action emit — (action_id=6, gx=2, gy=3) round-trips through GPU.

All GPU tests require CUDA and are skipped in no-GPU environments.
Import-level smoke (no GPU) runs unconditionally.
"""

from __future__ import annotations

import ctypes
import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Import-level smoke — must succeed even without CUDA.
# This validates module structure, not GPU execution.
# ---------------------------------------------------------------------------

def test_arc3_screen_bridge_module_importable():
    """Module imports without raising and exposes Arc3ScreenBridge class."""
    mod = importlib.import_module("knowledge3d.cranium.bridges.arc3_screen_bridge")
    assert hasattr(mod, "Arc3ScreenBridge"), (
        "arc3_screen_bridge module must export Arc3ScreenBridge"
    )


# ---------------------------------------------------------------------------
# GPU fixture
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
# Helper: allocate host bytes → device buffer
# ---------------------------------------------------------------------------

def _upload_bytes(data: bytes) -> object:
    """Copy a bytes object to GPU and return CUdeviceptr."""
    from knowledge3d.cranium.sovereign import loader
    n = len(data)
    d_ptr = loader.gpu_malloc(n)
    ByteArray = ctypes.c_uint8 * n
    h_buf = ByteArray(*data)
    loader.memcpy_htod(d_ptr, ctypes.cast(h_buf, ctypes.c_void_p), n)
    return d_ptr


def _download_bytes(d_ptr: object, n: int) -> bytes:
    """Copy n bytes from GPU device pointer back to host."""
    from knowledge3d.cranium.sovereign import loader
    ByteArray = ctypes.c_uint8 * n
    h_buf = ByteArray()
    loader.memcpy_dtoh(ctypes.cast(h_buf, ctypes.c_void_p), d_ptr, n)
    return bytes(h_buf)


# ---------------------------------------------------------------------------
# Test 1: Palette upload + 4×4 frame decode
# ---------------------------------------------------------------------------

@skip_no_gpu
def test_decode_4x4_frame_matches_palette():
    """Upload a 2-color palette, decode a known 4×4 index frame, verify RGBA.

    Palette:
        index 0 → RGBA = (0xFF, 0x00, 0x00, 0xFF)  packed = 0xFF0000FF
        index 1 → RGBA = (0x00, 0xFF, 0x00, 0xFF)  packed = 0xFF00FF00
        indices 2..15 → 0x00000000

    Frame (4×4, row-major):
        0 1 0 1
        1 0 1 0
        0 1 0 1
        1 0 1 0
    """
    import numpy as np  # numpy allowed in test scaffolding
    from knowledge3d.cranium.sovereign import loader
    from knowledge3d.cranium.bridges.arc3_screen_bridge import Arc3ScreenBridge

    bridge = Arc3ScreenBridge()

    # Build 16-entry palette (packed RGBA little-endian: R|G<<8|B<<16|A<<24)
    palette = [0] * 16
    # index 0 → red   (R=0xFF, G=0x00, B=0x00, A=0xFF)
    palette[0] = (0xFF << 0) | (0x00 << 8) | (0x00 << 16) | (0xFF << 24)
    # index 1 → green (R=0x00, G=0xFF, B=0x00, A=0xFF)
    palette[1] = (0x00 << 0) | (0xFF << 8) | (0x00 << 16) | (0xFF << 24)

    bridge.upload_palette(palette)

    # 4×4 checkerboard (palette indices)
    W, H = 4, 4
    frame_indices = bytes([
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
    ])
    d_frame = _upload_bytes(frame_indices)

    try:
        rgba_dev = bridge.decode_frame(d_frame, W, H)
        try:
            rgba_bytes = _download_bytes(rgba_dev, W * H * 4)
        finally:
            loader.gpu_free(rgba_dev)
    finally:
        loader.gpu_free(d_frame)

    # Build expected using numpy (test scaffolding only)
    rgba_arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(H, W, 4)

    # Check a few pixels
    # Pixel (0,0): index 0 → red
    np.testing.assert_array_equal(rgba_arr[0, 0], [0xFF, 0x00, 0x00, 0xFF],
                                  err_msg="pixel (0,0) should be red (index 0)")
    # Pixel (0,1): index 1 → green
    np.testing.assert_array_equal(rgba_arr[0, 1], [0x00, 0xFF, 0x00, 0xFF],
                                  err_msg="pixel (0,1) should be green (index 1)")
    # Pixel (1,0): index 1 → green
    np.testing.assert_array_equal(rgba_arr[1, 0], [0x00, 0xFF, 0x00, 0xFF],
                                  err_msg="pixel (1,0) should be green (index 1)")
    # Pixel (1,1): index 0 → red
    np.testing.assert_array_equal(rgba_arr[1, 1], [0xFF, 0x00, 0x00, 0xFF],
                                  err_msg="pixel (1,1) should be red (index 0)")


# ---------------------------------------------------------------------------
# Test 2: Click invert — (16,16) in rect (0,0,64,64), grid 4×4 → cell (1,1)
# ---------------------------------------------------------------------------

@skip_no_gpu
def test_click_invert_cell_1_1():
    """Screen pixel (16,16) maps to grid cell (1,1) for a 4×4 grid in 64×64 rect.

    Grid cell formula: gx = (lx * grid_w) / rect_w
        lx = sx - rect_x = 16 - 0 = 16
        gx = (16 * 4) / 64 = 1
        gy = (16 * 4) / 64 = 1
    """
    from knowledge3d.cranium.bridges.arc3_screen_bridge import Arc3ScreenBridge

    bridge = Arc3ScreenBridge()

    gx, gy = bridge.invert_click(
        sx=16, sy=16,
        rect=(0, 0, 64, 64),
        grid=(4, 4),
    )

    assert gx == 1, f"Expected gx=1, got gx={gx}"
    assert gy == 1, f"Expected gy=1, got gy={gy}"


@skip_no_gpu
def test_click_invert_out_of_rect():
    """Click outside rect returns (-1, -1)."""
    from knowledge3d.cranium.bridges.arc3_screen_bridge import Arc3ScreenBridge

    bridge = Arc3ScreenBridge()

    gx, gy = bridge.invert_click(
        sx=100, sy=100,
        rect=(0, 0, 64, 64),
        grid=(4, 4),
    )

    assert gx == -1, f"Expected gx=-1 for out-of-rect click, got gx={gx}"
    assert gy == -1, f"Expected gy=-1 for out-of-rect click, got gy={gy}"


# ---------------------------------------------------------------------------
# Test 3: Action emit round-trip
# ---------------------------------------------------------------------------

@skip_no_gpu
def test_action_emit_roundtrip():
    """(action_id=6, gx=2, gy=3) written through GPU kernel and read back intact."""
    from knowledge3d.cranium.bridges.arc3_screen_bridge import Arc3ScreenBridge

    bridge = Arc3ScreenBridge()

    action_id_in = 6   # Click action per ARC3 SDK
    gx_in = 2
    gy_in = 3

    action_id_out, gx_out, gy_out = bridge.emit_action(action_id_in, gx_in, gy_in)

    assert action_id_out == action_id_in, (
        f"action_id mismatch: expected {action_id_in}, got {action_id_out}"
    )
    assert gx_out == gx_in, f"gx mismatch: expected {gx_in}, got {gx_out}"
    assert gy_out == gy_in, f"gy mismatch: expected {gy_in}, got {gy_out}"
