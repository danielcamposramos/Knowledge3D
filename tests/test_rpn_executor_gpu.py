import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def _skip_if_kernel_missing(bridge):
    if bridge.rpn_executor_kernel is None:
        pytest.skip("rpn_executor.ptx not loaded")


@pytest.mark.cuda
def test_rpn_executor_basic_line():
    """Device-side RPN executor should emit a single line segment."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "0 0 MOVE 1 1 LINE STROKE"
    bytecode = bridge.compile_rpn_to_bytecode(program).tobytes()
    result = bridge.execute_rpn_bytecode_gpu(bytecode, width=64, height=64)
    assert result.segments.shape[0] == 1
    seg = result.segments[0]
    assert abs(seg[0]) < 1e-3 and abs(seg[1]) < 1e-3
    assert abs(seg[2] - 1.0) < 1e-3 and abs(seg[3] - 1.0) < 1e-3


@pytest.mark.cuda
def test_rpn_executor_quad_curve():
    """Quadratic Bézier should emit multiple segments."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    rpn = "0.0 0.0 MOVE 0.5 1.0 1.0 1.0 QUAD STROKE"
    bytecode = bridge.compile_rpn_to_bytecode(rpn).tobytes()
    result = bridge.execute_rpn_bytecode_gpu(bytecode, width=64, height=64)
    assert result.segments.shape[0] >= 10
    assert np.allclose(result.segments[0, :2], [0.0, 0.0], atol=0.05)
    assert np.allclose(result.segments[-1, 2:4], [1.0, 1.0], atol=0.05)


@pytest.mark.cuda
def test_rpn_executor_arc():
    """ARC opcode should emit approximated arc segments."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    # semicircle rx=1, ry=1, start=0, sweep=pi, center=0,0
    rpn = "0.0 0.0 MOVE 1.0 1.0 0.0 3.14159 0.0 0.0 ARC STROKE"
    bytecode = bridge.compile_rpn_to_bytecode(rpn).tobytes()
    result = bridge.execute_rpn_bytecode_gpu(bytecode, width=64, height=64)
    assert result.segments.shape[0] >= 10
    # End near (-1,0)
    end = result.segments[-1]
    assert abs(end[2] + 1.0) < 0.2
    assert abs(end[3]) < 0.2


@pytest.mark.cuda
def test_rpn_executor_ternary_width():
    """Ternary metadata should modulate stroke width."""
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    _skip_if_kernel_missing(bridge)

    program = "1.0 STROKE_WIDTH 0 0 MOVE 1 0 LINE STROKE"
    bc = bridge.compile_rpn_to_bytecode(program).tobytes()
    base = bridge.execute_rpn_bytecode_gpu(bc, width=32, height=32, ternary_meta=None)  # type: ignore[arg-type]
    bold = bridge.execute_rpn_bytecode_gpu(bc, width=32, height=32, ternary_meta=np.array([1], dtype=np.int8))  # type: ignore[arg-type]
    assert bold.segments[0, 8] > base.segments[0, 8]
