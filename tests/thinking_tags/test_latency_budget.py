import pytest
import numpy as np
from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge
from knowledge3d.cranium.ptx_runtime.latency_guard import LatencyGuard

@pytest.fixture
def bridge():
    return ThinkingTagBridge()

def test_latency_enforced(bridge):
    """Kimi's hard-fail latency test"""
    lg = LatencyGuard()
    for i in range(1000):
        input_emb = np.random.randn(512).astype(np.float32)
        with lg.measure_scope("inference"):
            result = bridge.inference(input_emb, ['text'])
        assert lg.last_duration <= 35e-6, f"Latency breach {i}: {lg.last_duration*1e6:.2f}µs"
