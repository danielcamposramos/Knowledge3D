import numpy as np
import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import (
    TritInspectorBridge,
    TritOverlayGenerator,
)
from knowledge3d.cranium.tools.trit_inspector import TritInspector

try:
    _drv_probe = TritOverlayGenerator()  # noqa: F841
except Exception as exc:  # pragma: no cover
    pytest.skip(f"CUDA driver/PTX unavailable: {exc}", allow_module_level=True)


def _pack_trits(trits):
    """Pack list of trits (-1,0,1) into 2-bit encoding."""
    enc = { -1: 0, 0: 1, 1: 2 }
    n_words = (len(trits) + 15) // 16
    buf = np.zeros(n_words, dtype=np.uint32)
    for i, t in enumerate(trits):
        bits = enc[int(t)]
        word = i >> 4
        shift = (i & 0xF) << 1
        buf[word] |= np.uint32(bits << shift)
    return buf


def test_trit_overlay_generator_basic():
    trits = [-1, 0, 1, -1]
    packed = _pack_trits(trits)
    gen = TritOverlayGenerator()
    rgba = gen.generate(packed, grid_shape=(4, 1, 1), field_stride=1)
    rgba = rgba.reshape(-1, 4)
    # Mapping: -1 -> blue, 0 -> transparent, +1 -> red
    assert tuple(rgba[0]) == (0, 0, 255, 96)
    assert tuple(rgba[1]) == (0, 0, 0, 0)
    assert tuple(rgba[2]) == (255, 0, 0, 96)
    assert tuple(rgba[3]) == (0, 0, 255, 96)


def test_trit_inspector_bridge_per_node():
    trits = [-1, 0, 1, -1]
    packed = _pack_trits(trits)
    nodes = np.arange(len(trits), dtype=np.int32)
    inspector = TritInspectorBridge()
    out = inspector.inspect(packed, nodes, field_stride=1)
    assert out.shape[0] == 4
    assert int(out["sum"][0]) == -1
    assert int(out["sum"][1]) == 0
    assert int(out["sum"][2]) == 1
    assert int(out["sum"][3]) == -1
    assert int(out["bottlenecks"][1]) == 1


def test_trit_inspector_high_level_path():
    trits = [-1, 0, 1, 1, -1]
    packed = _pack_trits(trits)
    inspector = TritInspector(field_stride=1)
    summary = inspector.trace_path_trits(packed, path_indices=[0, 1, 2, 3, 4])
    assert summary["path_length"] == 5
    # sum of trits = -1 + 0 + 1 + 1 -1 = 0
    assert summary["sum"] == 0
    assert summary["bottlenecks"] == 1
