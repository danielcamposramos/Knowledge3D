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
