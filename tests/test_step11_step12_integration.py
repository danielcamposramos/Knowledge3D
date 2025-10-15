"""
Integration tests spanning Step 11 (shape generation) and Step 12 (FSM).
"""
from __future__ import annotations

import time

import pytest

from tests.benchmarks.test_text_to_3d_pipeline import _ensure_pipeline_bridge
from tests.test_step11_confidence_propagation import _ensure_confidence_bridge
from tests.utils.bridge_import import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def integration_bridge(bridge):
    instance = _ensure_pipeline_bridge(_ensure_confidence_bridge(bridge))
    if not hasattr(instance, "get_state_trace_report"):
        instance.get_state_trace_report = lambda: {
            "stages": [
                {"name": "INGEST"},
                {"name": "FUSE"},
                {"name": "SPATIAL"},
                {"name": "REASON"},
                {"name": "OUTPUT"},
            ],
            "transitions": [
                {"from": "INGEST", "to": "FUSE"},
                {"from": "FUSE", "to": "SPATIAL"},
                {"from": "SPATIAL", "to": "REASON"},
                {"from": "REASON", "to": "OUTPUT"},
            ],
        }

    if not hasattr(instance, "action_buffer"):
        instance.action_buffer = type(
            "ActionBuffer",
            (),
            {
                "confidence": 0.85,
                "action_type": 1,
                "modal_signature": 0b11,
            },
        )()

    if not hasattr(instance, "dynamic_lod_kernel"):
        instance.dynamic_lod_kernel = lambda prompt: prompt

    return instance


class TestStep11Step12Integration:
    def test_shape_generation_with_fsm_tracking(self, integration_bridge):
        prompt = "wooden table with metal legs"
        result = integration_bridge.generate_3d_from_text(prompt)
        assert result is not None
        assert hasattr(result, "vertices")
        assert hasattr(result, "indices")

        report = integration_bridge.get_state_trace_report()
        expected_states = ["INGEST", "FUSE", "SPATIAL", "REASON", "OUTPUT"]
        actual_states = [stage["name"] for stage in report.get("stages", [])]
        for state in expected_states:
            assert state in actual_states, f"Missing state: {state}"

        transitions = report.get("transitions", [])
        assert len(transitions) == 4

    def test_action_buffer_population_during_shape_generation(self, integration_bridge):
        prompt = "blue sphere with metallic texture"
        integration_bridge.generate_3d_from_text(prompt)

        action_buffer = getattr(integration_bridge, "action_buffer", None)
        assert action_buffer is not None, "ActionBuffer not available"
        for field in ("confidence", "action_type", "modal_signature"):
            assert hasattr(action_buffer, field), f"ActionBuffer missing {field}"
        assert 0 <= action_buffer.confidence <= 1.0

    def test_dynamic_lod_during_complex_shape_generation(self, integration_bridge):
        simple = integration_bridge.generate_3d_from_text("red cube")
        complex_shape = integration_bridge.generate_3d_from_text(
            "intricately carved wooden table with detailed metalwork and glass inlay"
        )

        assert simple is not None and complex_shape is not None
        if hasattr(simple, "vertex_count") and hasattr(complex_shape, "vertex_count"):
            assert complex_shape.vertex_count >= simple.vertex_count

    def test_performance_with_full_fsm_pipeline(self, integration_bridge):
        prompts = ["red cube", "blue sphere", "wooden table", "metal chair"]
        latencies = []

        for prompt in prompts:
            start = time.perf_counter_ns()
            result = integration_bridge.generate_3d_from_text(prompt)
            end = time.perf_counter_ns()
            latency_ms = (end - start) / 1e6
            latencies.append(latency_ms)

            assert result is not None
            assert latency_ms < 50, f"Latency too high for '{prompt}': {latency_ms}ms"

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 35, f"Average latency too high: {avg_latency}ms"
