import numpy as np

from knowledge3d.training.rlwhf.train_rlwhf_ternary import ternary_sign


def test_ternary_sign_threshold():
    arr = np.array([-0.002, -0.0001, 0.0, 0.0002, 0.01], dtype=np.float32)
    out = ternary_sign(arr, threshold=0.001)
    assert out.tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]
