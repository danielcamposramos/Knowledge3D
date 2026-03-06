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


def test_rpn_blocks_roundtrip():
    """Block layout reshape ops should round-trip a 2D grid through PTX kernels."""
    engine = ModularRPNEngine()
    grid = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

    result = engine.evaluate("RESHAPE_TO_BLOCKS BLOCKS_TO_GRID", data=grid.tolist(), return_vector=True)
    arr = np.array(result, dtype=np.float32)

    assert arr.shape == grid.shape
    assert np.allclose(arr, grid)


def test_rpn_blocks_dct_roundtrip():
    """Block packet layout should survive DCT/IDCT and return to the original grid."""
    engine = ModularRPNEngine()
    grid = np.random.randn(16, 16).astype(np.float32)

    result = engine.evaluate(
        "RESHAPE_TO_BLOCKS DCT8X8_FORWARD IDCT8X8_INVERSE BLOCKS_TO_GRID",
        data=grid.tolist(),
        return_vector=True,
    )
    arr = np.array(result, dtype=np.float32)

    assert arr.shape == grid.shape
    assert np.allclose(arr, grid, atol=1e-3)
