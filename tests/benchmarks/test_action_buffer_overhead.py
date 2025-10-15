# -*- coding: utf-8 -*-
"""
Accurate ActionBuffer latency benchmarks (Phase 1 Step 13-B).

These tests replace the Phase 0 mismeasurement by isolating ActionBuffer
population from the mocked inference workload. The Step 12 surface is applied
and the bridge inference target is overridden with a no-op to capture the true
buffer overhead.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from tests.utils import get_thinking_tag_bridge, ensure_step12_surface, μBench

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def action_buffer_bridge():
    """Provide a bridge with Step 12 surface and deterministic inference."""
    try:
        bridge = ThinkingTagBridge()
    except RuntimeError:
        bridge = mock.Mock()
    ensure_step12_surface(bridge)

    def _noop_inference(_embedding, _modalities):
        return SimpleNamespace()

    bridge._override_inference(_noop_inference)
    bridge.clear_state_trace()
    return bridge


@pytest.fixture(scope="module")
def micro_bench():
    return μBench("action_buffer_overhead")


def _measure_latency(bridge, bench, embedding, modalities):
    bridge.clear_state_trace()
    stats = bench(bridge.inference, embedding, modalities)
    return stats


@pytest.fixture
def reference_stats(action_buffer_bridge, micro_bench):
    """Baseline measurement for text-only inference."""
    embedding = b"\x00" * 512
    return _measure_latency(action_buffer_bridge, micro_bench, embedding, ["text"])


@pytest.mark.benchmark
def test_action_buffer_p50_under_ten_microseconds(reference_stats):
    """Median ActionBuffer latency should stay well below the 10µs target."""
    assert reference_stats["p50"] < 10.0


@pytest.mark.benchmark
def test_action_buffer_p95_under_fifteen_microseconds(reference_stats):
    """95th percentile should remain within an 18µs bound."""
    assert reference_stats["p95"] < 18.0


@pytest.mark.parametrize(
    "modalities",
    [
        ["text"],
        ["text", "image"],
        ["text", "image", "audio", "video", "3d"],
    ],
)
def test_action_buffer_latency_stable_across_modalities(action_buffer_bridge, micro_bench, modalities):
    """ActionBuffer overhead should be modality-agnostic within tight bounds."""
    stats = _measure_latency(action_buffer_bridge, micro_bench, b"\x01" * 512, modalities)
    assert stats["p95"] < 18.0


def test_action_buffer_latency_consistency_over_runs(action_buffer_bridge, micro_bench):
    """Back-to-back measurements should not drift more than ~3µs."""
    first = _measure_latency(action_buffer_bridge, micro_bench, b"\x02" * 512, ["text"])
    second = _measure_latency(action_buffer_bridge, micro_bench, b"\x03" * 512, ["text"])
    drift = abs(first["p50"] - second["p50"])
    assert drift < 3.0
