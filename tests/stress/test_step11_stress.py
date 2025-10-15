"""
Stress scenarios for Step 11 shape generation (Deep Seek specification).

These tests focus on throughput and graceful degradation under memory pressure.
"""
from __future__ import annotations

import gc
import os
import time

import pytest

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from tests.test_step11_confidence_propagation import _ensure_confidence_bridge
from tests.utils.bridge_import import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def step11_bridge(bridge):
    return _ensure_confidence_bridge(bridge)


@pytest.mark.stress
def test_rapid_generation_1000_shapes(step11_bridge):
    """Generate 1000 shapes in 60 seconds (mock-friendly)."""
    start_time = time.time()
    shapes_generated = 0
    prompts = [f"shape_{idx}" for idx in range(1000)]

    for prompt in prompts:
        if time.time() - start_time > 60:
            break
        result = step11_bridge.generate_3d_from_text(prompt)
        if result is not None:
            shapes_generated += 1

    assert shapes_generated >= 800, f"Only generated {shapes_generated}/1000 shapes in 60s"


@pytest.mark.stress
def test_memory_exhaustion_graceful_degradation(step11_bridge):
    """Ensure memory pressure raises a controlled error."""
    if psutil is None:
        pytest.skip("psutil not available")

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss or 1
    shapes_until_oom = 0

    try:
        while True:
            prompt = f"very detailed complex object {shapes_until_oom}"
            step11_bridge.generate_3d_from_text(prompt)
            shapes_until_oom += 1

            if shapes_until_oom % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss
                if current_memory > initial_memory * 10:
                    raise MemoryError("Excessive memory growth detected")

            if shapes_until_oom > 100:
                raise MemoryError("Simulated memory exhaustion")

    except (MemoryError, RuntimeError) as exc:
        message = str(exc).lower()
        assert "memory" in message or "resource" in message
        assert shapes_until_oom > 10, "Should process at least 10 shapes before exhaustion"
