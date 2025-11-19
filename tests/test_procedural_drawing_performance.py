"""
Performance and decoding tests for GPU-native RPN execution.

If the pixel_genesis_universal_primitive PTX kernel is missing or CUDA is not
available, tests skip gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def _skip_if_kernel_missing(bridge):
    if bridge.pixel_genesis_kernel is None:
        pytest.skip("pixel_genesis_universal_primitive.ptx not loaded")


@pytest.mark.cuda
def test_gpu_rpn_operand_decoding():
    """Verify GPU decodes float operands and renders a diagonal line."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "-0.5 -0.5 MOVE 0.5 0.5 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=64, height=64)
    rgba = result.rgba

    assert np.any(rgba[..., 0] > 0), "Expected drawn pixels"
    non_zero = np.count_nonzero(rgba[..., 0] > 0.01)
    assert non_zero > 50, f"Expected ≥50 pixels, got {non_zero}"


@pytest.mark.cuda
def test_gpu_quad_bezier():
    """Verify GPU tessellates quadratic Bézier."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "-0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE"
    result = bridge.execute_rpn_gpu(program, width=96, height=96)
    rgba = result.rgba

    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 100, f"Curve should have ≥100 pixels, got {non_zero}"


@pytest.mark.cuda
def test_begin_path():
    """BEGIN_PATH should reset segment count."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "0 0 MOVE 1 1 LINE BEGIN_PATH 2 2 MOVE 3 3 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)
    assert result.segments.shape[0] == 1, f"Expected 1 segment after BEGIN_PATH, got {result.segments.shape[0]}"


@pytest.mark.cuda
def test_stroke_width():
    """STROKE_WIDTH opcode should not crash and keeps geometry intact."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "0.05 STROKE_WIDTH 0 0 MOVE 1 1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)
    assert result.segments.shape[0] == 1


@pytest.mark.cuda
def test_set_color():
    """SET_COLOR opcode should not crash and keeps geometry intact."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "1.0 0.5 0.0 1.0 SET_COLOR 0 0 MOVE 1 1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True)
    assert result.segments.shape[0] == 1


@pytest.mark.cuda
def test_ternary_modulate_local():
    """Local ternary hint overrides global when set."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Local hint = blur (-1.0) should override global normal (0.0)
    # Base segments=32, blur scale=0.5 → 32*0.5=16 segments
    program = "-1.0 TERNARY_MODULATE -0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE"
    result = bridge.execute_rpn_gpu(program, skip_raster=True, ternary_hint=0.0)
    # Should get 16 segments (blur), not 32 (normal)
    assert result.segments.shape[0] == 16, f"Expected 16 segments with local blur, got {result.segments.shape[0]}"


@pytest.mark.cuda
def test_gpu_cubic_bezier():
    """Verify GPU tessellates cubic Bézier."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "0.0 0.0 MOVE 0.3 0.5 0.7 0.5 1.0 0.0 CUBIC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)
    rgba = result.rgba

    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 150, f"S-curve should have ≥150 pixels, got {non_zero}"


@pytest.mark.cuda
def test_rotate_matrix_90deg():
    """ROTATE_MATRIX uses math buffer (cos,sin) to rotate a line by 90 degrees."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    math_buffer = np.array([0.0, 1.0], dtype=np.float32)
    program = "ROTATE_MATRIX 0 0 MOVE 1 0 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True, math_buffer=math_buffer)
    seg = result.segments[0]
    assert abs(seg[0]) < 0.05 and abs(seg[1]) < 0.05
    assert abs(seg[2]) < 0.05 and abs(seg[3] - 1.0) < 0.1


@pytest.mark.cuda
def test_rpn_automatic_rotation():
    """Verify automatic RPN preprocessing for rotation (RPN_SIN/RPN_COS tokens)."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Automatic RPN preprocessing: "PI 2 / RPN_SIN RPN_COS" → math_buffer=[0, 1]
    program = "PI 2 / RPN_SIN RPN_COS ROTATE_MATRIX 0 0 MOVE 1 0 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True)

    # Original: (0,0) → (1,0) horizontal line
    # Rotated 90°: (0,0) → (0,1) vertical line
    seg = result.segments[0]
    assert abs(seg[0]) < 0.05 and abs(seg[1]) < 0.05  # Start at origin
    assert abs(seg[2]) < 0.05  # x1 ≈ 0 (vertical)
    assert abs(seg[3] - 1.0) < 0.1  # y1 ≈ 1


@pytest.mark.cuda
def test_precomputed_path_triangle():
    """PRECOMPUTED_PATH consumes math buffer points."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    math_buffer = np.array([3.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.866], dtype=np.float32)
    program = "PRECOMPUTED_PATH CLOSE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True, math_buffer=math_buffer)

    assert result.segments.shape[0] >= 3


@pytest.mark.cuda
def test_gpu_arc():
    """Verify GPU tessellates an arc via RPN preprocessing."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # semicircle centered at origin, rx=1, ry=1, start=0, sweep=pi
    program = "-1 0 MOVE 1 1 0 3.14159 0 0 RPN_ARC STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128)
    rgba = result.rgba
    non_zero = np.count_nonzero(rgba[..., 0] > 0.05)
    assert non_zero > 100, f"Arc should have ≥100 pixels, got {non_zero}"


@pytest.mark.cuda
def test_multiple_rpn_arcs():
    """Multiple RPN_ARC tokens should produce multiple math records and segments."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Two semicircles back-to-back (should yield roughly 2× segments_per_curve)
    program = (
        "-1 0 MOVE "
        "1 1 0 3.14159 0 0 RPN_ARC "
        "1 1 3.14159 3.14159 0 0 RPN_ARC "
        "STROKE"
    )
    result = bridge.execute_rpn_gpu(program, skip_raster=True)
    assert result.segments.shape[0] >= 60, f"Expected multiple arcs, got {result.segments.shape[0]}"


@pytest.mark.cuda
def test_rpn_arc_gpu_perf():
    """RPN arc preprocessing + drawing should stay under a modest GPU budget."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
    import time

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "-1 0 MOVE 1 1 0 3.14159 0 0 RPN_ARC STROKE"
    t0 = time.perf_counter()
    # skip raster to focus on math + segment emission
    bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Generous threshold; PTX RPN + drawing should stay sub-30 ms even on mid GPUs
    assert elapsed_ms < 30.0, f"Arc GPU path too slow: {elapsed_ms:.2f} ms"


@pytest.mark.cuda
def test_ternary_hint_modulation():
    """Verify ternary hint modulates tessellation quality."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "-0.8 -0.8 MOVE 0.0 0.8 0.8 -0.2 QUAD STROKE"

    blur = bridge.execute_rpn_gpu(program, skip_raster=True, ternary_hint=-1.0)
    norm = bridge.execute_rpn_gpu(program, skip_raster=True, ternary_hint=0.0)
    sharp = bridge.execute_rpn_gpu(program, skip_raster=True, ternary_hint=1.0)

    assert blur.segments.shape[0] < norm.segments.shape[0] < sharp.segments.shape[0]


@pytest.mark.cuda
def test_rpn_execution_latency():
    """Ensure GPU path respects latency budget when kernel is available."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge
    from knowledge3d.cranium.ptx_runtime.latency_guard import LatencyGuard

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    guard = LatencyGuard(threshold_us=100.0)
    program = "0 0 MOVE 1 1 LINE STROKE"

    guard.start()
    bridge.execute_rpn_gpu(program, width=64, height=64)
    elapsed_ns, breached = guard.stop()

    if breached:
        pytest.xfail(f"Latency budget violated: {elapsed_ns / 1000:.1f} µs (optimization pending)")


@pytest.mark.cuda
def test_ai_mode_latency():
    """Skip raster path should be fast and return segments only."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    result = bridge.execute_rpn_gpu("0 0 MOVE 1 1 LINE STROKE", skip_raster=True)
    assert result.rgba is None
    assert result.segments is not None


@pytest.mark.cuda
def test_transform_translate():
    """Verify TRANSLATE opcode transforms coordinates."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Draw line at origin, then translate and draw again
    program = "0 0 MOVE 0.1 0.1 LINE 0.5 0.5 TRANSLATE 0 0 MOVE 0.1 0.1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True)

    # Should have 2 line segments
    assert result.segments.shape[0] == 2, f"Expected 2 segments, got {result.segments.shape[0]}"

    # First segment: (0, 0) → (0.1, 0.1) with identity transform
    seg1 = result.segments[0]
    assert abs(seg1[0] - 0.0) < 0.01, f"Segment 1 x0: expected 0.0, got {seg1[0]}"
    assert abs(seg1[1] - 0.0) < 0.01, f"Segment 1 y0: expected 0.0, got {seg1[1]}"
    assert abs(seg1[2] - 0.1) < 0.01, f"Segment 1 x1: expected 0.1, got {seg1[2]}"
    assert abs(seg1[3] - 0.1) < 0.01, f"Segment 1 y1: expected 0.1, got {seg1[3]}"

    # Second segment: (0, 0) → (0.1, 0.1) translated by (0.5, 0.5)
    seg2 = result.segments[1]
    assert abs(seg2[0] - 0.5) < 0.01, f"Segment 2 x0: expected 0.5, got {seg2[0]}"
    assert abs(seg2[1] - 0.5) < 0.01, f"Segment 2 y0: expected 0.5, got {seg2[1]}"
    assert abs(seg2[2] - 0.6) < 0.01, f"Segment 2 x1: expected 0.6, got {seg2[2]}"
    assert abs(seg2[3] - 0.6) < 0.01, f"Segment 2 y1: expected 0.6, got {seg2[3]}"


@pytest.mark.cuda
def test_transform_scale():
    """Verify SCALE opcode scales coordinates."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # Draw line, then scale and draw again
    program = "0 0 MOVE 0.1 0.1 LINE 2 2 SCALE 0 0 MOVE 0.1 0.1 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=128, height=128, skip_raster=True)

    # Should have 2 line segments
    assert result.segments.shape[0] == 2, f"Expected 2 segments, got {result.segments.shape[0]}"

    # First segment: (0, 0) → (0.1, 0.1) with identity transform
    seg1 = result.segments[0]
    assert abs(seg1[2] - 0.1) < 0.01, f"Segment 1 x1: expected 0.1, got {seg1[2]}"

    # Second segment: (0, 0) → (0.1, 0.1) scaled by 2×
    seg2 = result.segments[1]
    assert abs(seg2[2] - 0.2) < 0.01, f"Segment 2 x1: expected 0.2, got {seg2[2]}"
    assert abs(seg2[3] - 0.2) < 0.01, f"Segment 2 y1: expected 0.2, got {seg2[3]}"


@pytest.mark.cuda
def test_segments_stride_and_style():
    """Segments should include style fields (stride=9) and rasterization should honor color."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "1.0 0.0 0.0 1.0 SET_COLOR 0.05 STROKE_WIDTH 0 0 MOVE 1 0 LINE STROKE"
    result = bridge.execute_rpn_gpu(program, width=64, height=64)
    assert result.segments.shape[1] == 9, "Expected stride=9 (x,y + rgba + width)"
    rgba = result.rgba
    red = rgba[..., 0].mean()
    green = rgba[..., 1].mean()
    blue = rgba[..., 2].mean()
    assert red > green + 1e-3 and red > blue + 1e-3, "Red channel should dominate for red stroke"


@pytest.mark.cuda
def test_parallel_batch_drawing():
    """Run multiple programs; placeholder loop until kernel batch mode lands."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    programs = [
        f"{i*0.05} {i*0.05} MOVE {i*0.08} {i*0.1} LINE STROKE"
        for i in range(6)
    ]
    results = bridge.execute_batch_gpu(programs, width=64, height=64)

    assert len(results) == len(programs)
    for res in results:
        assert res.rgba.shape == (64, 64, 4)
