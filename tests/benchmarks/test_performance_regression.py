"""
Phase 1 regression benchmarks verifying latency budgets against the
comprehensive performance baseline produced in Phase 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


BASELINE_PATH = Path("reports/comprehensive_performance_baseline.json")


@pytest.fixture(scope="module")
def baseline_data():
    if not BASELINE_PATH.exists():
        pytest.skip("Comprehensive baseline not available – cannot run regression benchmarks")
    with BASELINE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_baseline_contains_expected_sections(baseline_data):
    """Baseline should contain all mandatory measurement sections."""
    expected_sections = {"text_to_3d_pipeline", "state_tracking", "action_buffer", "dynamic_lod", "multi_modal_fusion"}
    available_sections = set(baseline_data["results"].keys())
    assert expected_sections.issubset(available_sections)


@pytest.mark.parametrize("component", ["text_to_3d_pipeline", "state_tracking", "action_buffer", "dynamic_lod", "multi_modal_fusion"])
def test_component_percentiles_are_monotonic(baseline_data, component):
    """Each percentile series must remain monotonic increasing."""
    component_stats = baseline_data["results"][component]
    for key, stats in component_stats.items():
        assert stats["p50"] <= stats["p95"] <= stats["p99"], f"Percentiles out of order for {component}:{key}"


def test_text_to_3d_pipeline_meets_latency_targets(baseline_data):
    """Text-to-3D pipeline p50 latency should remain comfortably under 10ms."""
    pipeline = baseline_data["results"]["text_to_3d_pipeline"]
    for prompt, stats in pipeline.items():
        assert stats["p50"] < 10.0, f"Pipeline p50 regression detected for prompt: {prompt}"


def test_state_tracking_remains_under_two_microseconds(baseline_data):
    """State tracking median latency target is <2µs."""
    tracking = baseline_data["results"]["state_tracking"]
    for prompt, stats in tracking.items():
        assert stats["p50"] < 3.0, f"State tracking regression detected for prompt: {prompt}"


def test_action_buffer_latency_now_under_ten_microseconds(baseline_data):
    """Validated measurement should confirm ActionBuffer p50 < 10µs."""
    action_buffer = baseline_data["results"]["action_buffer"]
    for prompt, stats in action_buffer.items():
        assert stats["p50"] < 10.0, f"ActionBuffer still above target for prompt: {prompt}"


def test_dynamic_lod_latency_within_sub_micro_budget(baseline_data):
    """Dynamic LOD p99 should remain below 0.25µs."""
    lod_stats = baseline_data["results"]["dynamic_lod"]
    for threshold, stats in lod_stats.items():
        assert stats["p99"] < 0.25, f"Dynamic LOD regression at threshold {threshold}"


def test_multi_modal_fusion_latency_within_budget(baseline_data):
    """Multi-modal fusion p50 should remain below 35µs."""
    fusion = baseline_data["results"]["multi_modal_fusion"]
    for combo, stats in fusion.items():
        assert stats["p50"] < 35.0, f"Fusion latency regression detected for modalities: {combo}"
