import numpy as np
import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import TernaryDepthField
from knowledge3d.cranium.tools.ternary_depth import TernaryDepthComputer

try:
    _probe = TernaryDepthField()  # noqa: F841
except Exception as exc:  # pragma: no cover
    pytest.skip(f"CUDA driver/PTX unavailable: {exc}", allow_module_level=True)


def _unpack_trits(packed, n):
    out = []
    for i in range(n):
        word = packed[i >> 4]
        shift = (i & 0xF) << 1
        bits = (word >> shift) & 0x3
        if bits == 2:
            out.append(1)
        elif bits == 1:
            out.append(0)
        else:
            out.append(-1)
    return out


def test_ternary_depth_field_simple():
    # 4 nodes, dim=2
    embeddings = np.array(
        [
            [1.0, 0.0],   # aligned with query -> +1
            [-1.0, 0.0],  # opposite -> -1
            [0.0, 0.0],   # neutral -> 0
            [0.5, 0.0],   # moderate positive -> +1 if above threshold
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    bridge = TernaryDepthField()
    packed = bridge.compute(embeddings, query, attract_thresh=0.25, repel_thresh=-0.1)
    trits = _unpack_trits(packed, n=4)
    assert trits == [1, -1, 0, 1]


def test_ternary_depth_computer_wrapper():
    embeddings = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    query = np.array([0.0, 1.0], dtype=np.float32)
    comp = TernaryDepthComputer()
    packed = comp.compute(embeddings, query, attract_thresh=0.2, repel_thresh=-0.1)
    trits = _unpack_trits(packed, n=2)
    assert trits == [1, -1]
