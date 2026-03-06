from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps


def test_quant_dequant_roundtrip():
    ops = TernaryCodecOps(threshold=0.2)
    vals = [0.1, 0.5, -0.3, 0.0]
    q = ops.quantize(vals)
    dq = ops.dequantize(q)
    assert q == [0, 1, -1, 0]
    assert [int(round(x)) for x in dq] == q


def test_mdct_roundtrip():
    """MDCT forward + inverse should preserve signal shape with high correlation."""
    import numpy as np

    ops = TernaryCodecOps()

    frame_size = 1024
    t = np.linspace(0, 2 * np.pi, frame_size, dtype=np.float32)
    signal = np.sin(440 * t).astype(np.float32)

    window = np.hanning(frame_size).astype(np.float32)
    windowed = signal * window

    coeffs = ops.mdct_forward(windowed.tolist())
    assert len(coeffs) == frame_size // 2

    reconstructed = ops.imdct_inverse(coeffs, frame_size)
    assert len(reconstructed) == frame_size

    correlation = float(np.corrcoef(windowed, np.array(reconstructed, dtype=np.float32))[0, 1])
    assert correlation > 0.95, f"MDCT round-trip correlation too low: {correlation}"


def test_block_layout_roundtrip():
    """Grid-to-block and block-to-grid should be exact for 8x8-divisible float grids."""
    import numpy as np

    ops = TernaryCodecOps()
    grid = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    flat = grid.reshape(-1).tolist()

    blocks = ops.reshape_to_blocks(flat, rows=16, cols=16, block_h=8, block_w=8)
    rebuilt = ops.blocks_to_grid(blocks, rows=16, cols=16, block_h=8, block_w=8)
    arr = np.array(rebuilt, dtype=np.float32).reshape(16, 16)

    assert np.allclose(arr, grid)


def test_dct8_roundtrip():
    """DCT8 forward + inverse should reconstruct contiguous 8x8 blocks accurately."""
    import numpy as np

    ops = TernaryCodecOps()
    blocks = np.random.randn(3, 8, 8).astype(np.float32)
    flat = blocks.reshape(-1).tolist()

    coeffs = ops.dct8_forward(flat)
    rebuilt = ops.dct8_inverse(coeffs)
    arr = np.array(rebuilt, dtype=np.float32).reshape(3, 8, 8)

    assert np.allclose(arr, blocks, atol=1e-3)


def test_numpy_codec_paths_roundtrip():
    import numpy as np

    ops = TernaryCodecOps(threshold=0.2)
    grid = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

    blocks = ops.reshape_to_blocks_numpy(grid.reshape(-1), rows=16, cols=16)
    coeffs = ops.dct8_forward_numpy(blocks)
    quantized = ops.quantize_numpy(coeffs)
    dequantized = ops.dequantize_numpy(quantized)
    rebuilt = ops.dct8_inverse_numpy(dequantized)
    roundtrip = ops.blocks_to_grid_numpy(rebuilt, rows=16, cols=16).reshape(16, 16)

    assert blocks.dtype == np.float32
    assert quantized.dtype == np.int32
    assert roundtrip.shape == (16, 16)
