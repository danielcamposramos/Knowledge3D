import numpy as np

from knowledge3d.cranium.codecs.ternary_quantization import (
    compute_sparsity,
    dequantize_ternary,
    entropy_decode_ternary,
    entropy_encode_ternary,
    quantize_ternary,
)


def test_quantize_ternary_basic():
    coeffs = np.array([0.5, -0.8, 0.05, -0.02, 1.2, -1.5])
    quantized, meta = quantize_ternary(coeffs, threshold=0.1, adaptive=False)

    expected = np.array([1, -1, 0, 0, 1, -1], dtype=np.int8)
    assert np.array_equal(quantized, expected)
    assert meta["sparsity"] == 2 / 6
    assert compute_sparsity(quantized) == meta["sparsity"]


def test_ternary_roundtrip():
    rng = np.random.default_rng(0)
    original = rng.standard_normal(1000)
    quantized, meta = quantize_ternary(original, threshold=0.3, adaptive=False)
    reconstructed = dequantize_ternary(quantized, scale=1.0)

    non_zero_mask = quantized != 0
    assert np.all(np.sign(original[non_zero_mask]) == np.sign(reconstructed[non_zero_mask]))


def test_entropy_coding_roundtrip():
    quantized = np.array([0, 0, 0, 1, -1, 0, 1, 1, 0, 0], dtype=np.int8)
    encoded = entropy_encode_ternary(quantized)
    decoded = entropy_decode_ternary(encoded)

    assert np.array_equal(quantized, decoded)
    assert len(encoded) < len(quantized) * 2  # RLE should compress zero runs
