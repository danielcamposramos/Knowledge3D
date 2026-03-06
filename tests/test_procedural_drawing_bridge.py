"""
Smoke tests for the procedural drawing bridge.

These lean on the existing procedural_glyph_rasterizer PTX kernel to validate
that RPN drawing programs are parsed → segmented → rasterized on GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_draw_simple_line():
    """Basic MOVE/LINE/STROKE path renders non-empty pixels."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=64)
    result = bridge.execute_rpn_program(" -0.5 -0.5 MOVE 0.5 0.5 LINE STROKE ", width=64, height=64)
    rgba = result.rgba

    assert rgba.shape == (64, 64, 4)
    assert np.any(rgba[..., 0] > 0), "Expected drawn pixels in red channel"


@pytest.mark.cuda
def test_draw_quadratic_curve():
    """Quadratic curve should rasterize to multiple non-zero pixels."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=128)
    result = bridge.execute_rpn_program(" -0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE ", width=96, height=96)
    rgba = result.rgba

    assert rgba.shape == (96, 96, 4)
    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 50, f"Curve should produce visible pixels, got {non_zero}"


@pytest.mark.cuda
def test_glyph_bridge_reuses_persistent_output_buffer():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_glyph_bridge import ProceduralGlyphBridge

    bridge = ProceduralGlyphBridge()
    segments = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    offsets = np.array([0], dtype=np.int32)
    lengths = np.array([1], dtype=np.int32)
    transforms = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    first = bridge.render(segments, offsets, lengths, transforms, batch=1, height=32, width=32)
    first_ptr = int(first.device_ptr.value)
    _ = first.to_numpy()

    second = bridge.render(segments, offsets, lengths, transforms, batch=1, height=32, width=32)
    second_ptr = int(second.device_ptr.value)
    _ = second.to_numpy()

    assert first_ptr != 0
    assert first_ptr == second_ptr
    bridge.close()


@pytest.mark.cuda
def test_drawing_bridge_warmup_is_idempotent():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=64)
    first = bridge.warmup_runtime()
    second = bridge.warmup_runtime()

    assert first["status"] == "ready"
    assert float(first["total_warmup_ms"]) > 0.0
    assert second["status"] == "ready"
    assert first == second
