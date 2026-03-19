from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fontTools")

from knowledge3d.cranium.procedural_fonts import segments_to_rpn
from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge


def test_segments_to_rpn_roundtrip_host():
    """Segments→RPN→host parser should preserve geometry count and style tokens."""
    # simple triangle
    segments = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.5, 1.0],
            [0.5, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    program = segments_to_rpn(segments, stroke_color=(1.0, 0.0, 0.0, 1.0), stroke_width=0.2)
    # Use host parser to avoid CUDA dependency
    bridge = ProceduralDrawingBridge(matryoshka_dim=512)
    result = bridge.execute_rpn_program(program, width=32, height=32)
    assert result.segments.shape[0] == segments.shape[0]
    assert "SET_COLOR" in program and "STROKE_WIDTH" in program and "STROKE" in program
