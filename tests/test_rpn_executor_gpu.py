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
