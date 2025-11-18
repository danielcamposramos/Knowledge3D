import numpy as np
import pytest

from knowledge3d.cranium.bridges.sovereign_bridges import TernaryPruneDecision

try:
    _probe = TernaryPruneDecision()  # noqa: F841
except Exception as exc:  # pragma: no cover
    pytest.skip(f"CUDA driver/PTX unavailable: {exc}", allow_module_level=True)


def test_ternary_prune_decision_basic():
    scores = np.array([0.8, 0.1, 0.01, -0.2], dtype=np.float32)
    bridge = TernaryPruneDecision()
    out = bridge.decide(scores, keep_thresh=0.5, drop_thresh=0.05)
    assert out.tolist() == [1, 0, -1, -1]
