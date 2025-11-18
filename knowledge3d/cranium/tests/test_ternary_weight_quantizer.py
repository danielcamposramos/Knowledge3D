import numpy as np

from knowledge3d.cranium.tools.ternary_weight_quantizer import quantize_to_ternary


def test_quantize_to_ternary_basic():
    w = np.array([-0.2, -0.01, 0.0, 0.02, 0.5], dtype=np.float32)
    q = quantize_to_ternary(w, threshold=0.05)
    assert q.tolist() == [-1, 0, 0, 0, 1]


def test_quantize_to_ternary_sparsity():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.01, size=1000).astype(np.float32)
    q = quantize_to_ternary(w, threshold=0.05)
    sparsity = np.mean(q == 0)
    assert sparsity > 0.9
