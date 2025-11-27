"""Codec opcodes routed through ModularRPNEngine should hit GPU codec kernels."""

import numpy as np

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


def test_rpn_dct_quant():
    """DCT8 + ternary quantisation via RPN should yield ternary grid."""
    engine = ModularRPNEngine()
    block = np.random.randn(8, 8).astype(np.float32)

    program = "DCT8X8_FORWARD 0.2 TERNARY_QUANT"
    result = engine.evaluate(program, data=block.tolist(), return_vector=True)
    arr = np.array(result, dtype=int)

    assert arr.shape == block.shape
    assert set(np.unique(arr)).issubset({-1, 0, 1})


def test_rpn_mdct_batch():
    """Batch MDCT + quantisation via RPN should reshape per frame."""
    engine = ModularRPNEngine()
    frame_size = 1024
    num_frames = 3
    frames = np.random.randn(num_frames, frame_size).astype(np.float32)

    program = f"{frame_size} BATCH_MDCT 0.1 TERNARY_QUANT"
    result = engine.evaluate(program, data=frames.tolist(), return_vector=True)
    arr = np.array(result, dtype=int)

    assert arr.shape == (num_frames, frame_size // 2)
    assert set(np.unique(arr)).issubset({-1, 0, 1})
