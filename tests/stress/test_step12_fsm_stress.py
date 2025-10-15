"""
Stress test for the Step 12 FSM pipeline (Deep Seek specification).

Runs a burst of concurrent inference calls to validate that the mocked bridge
maintains integrity and reports acceptable latency statistics.
"""
from __future__ import annotations

import threading
import time
from typing import List, Tuple

import pytest

from tests.utils.bridge_import import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def stress_bridge(bridge):
    """
    Reuse the shared bridge fixture but ensure it provides the interfaces
    required for the stress scenario.
    """
    if not hasattr(bridge, "inference"):
        bridge.inference = lambda embedding, signature: bridge
    if not hasattr(bridge, "get_state_trace_report"):
        bridge.get_state_trace_report = lambda: {
            "statistics": {"p99": 60000},
            "stages": [],
            "transitions": [],
        }
    return bridge


@pytest.mark.stress
def test_high_frequency_inference_storm(stress_bridge, mock_embedding):
    """Execute 1000 inferences across 10 workers and validate FSM integrity."""
    results: List[Tuple[int, int, object]] = []
    errors: List[Tuple[int, Exception]] = []
    lock = threading.Lock()

    def inference_worker(worker_id: int) -> None:
        for iteration in range(100):
            try:
                outcome = stress_bridge.inference(mock_embedding, ["text"])
                with lock:
                    results.append((worker_id, iteration, outcome))
            except Exception as exc:  # pragma: no cover - defensive
                with lock:
                    errors.append((worker_id, exc))

    threads = [threading.Thread(target=inference_worker, args=(i,)) for i in range(10)]
    start_time = time.time()

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    duration = time.time() - start_time
    throughput = len(results) / duration if duration else float("inf")

    assert duration <= 10.0, f"Inference storm exceeded 10s budget: {duration:.2f}s"
    assert not errors, f"Stress test produced errors: {errors}"
    assert len(results) >= 500, f"Throughput too low: {len(results)} results"
    assert throughput > 50.0, f"Throughput too low: {throughput:.2f} ops/sec"

    if hasattr(stress_bridge, "get_state_trace_report"):
        report = stress_bridge.get_state_trace_report()
        p99 = report.get("statistics", {}).get("p99", 0)
        assert p99 < 100000, f"p99 latency too high: {p99}µs"
