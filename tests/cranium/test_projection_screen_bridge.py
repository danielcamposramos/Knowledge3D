"""Smoke tests for ProjectionScreenBridge codec launcher.

Tests: §3.5 of CLAUDE_DUAL_PATH_INGESTION_AND_DISPATCH_WIRING_04.20.2026.md

GPU test:
    1. Project a 2×2 red viewport into the top-left 2×2 corner of a 4×4
       black screen. Verify the destination rectangle is red and the
       remainder stays black.

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

def test_projection_screen_bridge_module_importable():
    """Module imports without raising and exposes ProjectionScreenBridge class."""
    mod = importlib.import_module(
        "knowledge3d.cranium.bridges.projection_screen_bridge"
    )
    assert hasattr(mod, "ProjectionScreenBridge"), (
        "projection_screen_bridge module must export ProjectionScreenBridge"
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
# Helpers
# ---------------------------------------------------------------------------

def _upload_bytes(data: bytes) -> object:
    """Copy bytes to GPU and return CUdeviceptr."""
    from knowledge3d.cranium.sovereign import loader
    n = len(data)
    d_ptr = loader.gpu_malloc(n)
    ByteArray = ctypes.c_uint8 * n
    h_buf = ByteArray(*data)
    loader.memcpy_htod(d_ptr, ctypes.cast(h_buf, ctypes.c_void_p), n)
    return d_ptr


def _download_bytes(d_ptr: object, n: int) -> bytes:
    """Copy n bytes from GPU device pointer to host."""
    from knowledge3d.cranium.sovereign import loader
    ByteArray = ctypes.c_uint8 * n
    h_buf = ByteArray()
    loader.memcpy_dtoh(ctypes.cast(h_buf, ctypes.c_void_p), d_ptr, n)
    return bytes(h_buf)


# ---------------------------------------------------------------------------
# Test 1: Project 2×2 red viewport into 4×4 black screen at (0,0,2,2)
# ---------------------------------------------------------------------------

@skip_no_gpu
def test_project_viewport_into_screen_corner():
    """2×2 red viewport blitted into top-left 2×2 rect of 4×4 black screen.

    Pixel layout (screen is 4 wide × 4 tall, RGBA row-major):
    After projection:
        row 0: [RED, RED, BLACK, BLACK]
        row 1: [RED, RED, BLACK, BLACK]
        row 2: [BLACK, BLACK, BLACK, BLACK]
        row 3: [BLACK, BLACK, BLACK, BLACK]
    """
    from knowledge3d.cranium.sovereign import loader
    from knowledge3d.cranium.bridges.projection_screen_bridge import ProjectionScreenBridge

    Vw, Vh = 2, 2
    Sw, Sh = 4, 4
    RED   = bytes([0xFF, 0x00, 0x00, 0xFF])
    BLACK = bytes([0x00, 0x00, 0x00, 0xFF])

    # Viewport: all red.
    viewport_bytes = RED * (Vw * Vh)
    # Screen: all black.
    screen_bytes = BLACK * (Sw * Sh)

    d_viewport = _upload_bytes(viewport_bytes)
    d_screen   = _upload_bytes(screen_bytes)
    bridge = ProjectionScreenBridge()

    try:
        bridge.project_to_screen(
            d_viewport, d_screen,
            Vw=Vw, Vh=Vh, Sw=Sw, Sh=Sh,
            rect=(0, 0, 2, 2),
        )
        result = _download_bytes(d_screen, Sw * Sh * 4)
    finally:
        loader.gpu_free(d_viewport)
        loader.gpu_free(d_screen)

    # Parse into list of (R, G, B, A) tuples.
    pixels = [
        (result[i], result[i + 1], result[i + 2], result[i + 3])
        for i in range(0, len(result), 4)
    ]

    # Top-left 2×2 must be red.
    for row in range(2):
        for col in range(2):
            p = pixels[row * Sw + col]
            assert p == (0xFF, 0x00, 0x00, 0xFF), (
                f"pixel ({col},{row}) should be red, got {p}"
            )

    # Pixels outside the rect must remain black.
    for row in range(Sh):
        for col in range(Sw):
            if row < 2 and col < 2:
                continue  # already checked above
            p = pixels[row * Sw + col]
            assert p == (0x00, 0x00, 0x00, 0xFF), (
                f"pixel ({col},{row}) should be black, got {p}"
            )
