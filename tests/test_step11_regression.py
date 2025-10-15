"""
Regression safeguards for Step 11 (GLM specification).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest

from tests.test_step11_confidence_propagation import _ensure_confidence_bridge
from tests.utils.bridge_import import get_thinking_tag_bridge

ThinkingTagBridge = get_thinking_tag_bridge()


@pytest.fixture
def regression_bridge(bridge):
    return _ensure_confidence_bridge(bridge)


@pytest.fixture
def regression_cases() -> Iterable[Dict[str, Any]]:
    data_path = Path("tests/data/regression_cases.json")
    if data_path.exists():
        with data_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return [
        {
            "id": "issue_001",
            "description": "Zero dimension handling",
            "prompt": "cube with zero height",
            "expected": "should not crash",
        },
        {
            "id": "issue_002",
            "description": "Unicode handling",
            "prompt": "椅子 with 中文 characters",
            "expected": "should handle unicode correctly",
        },
        {
            "id": "issue_003",
            "description": "Empty prompt",
            "prompt": "",
            "expected": "should return default primitive",
        },
    ]


def test_regression_cases(regression_bridge, regression_cases):
    """Ensure historical regression cases remain fixed."""
    for case in regression_cases:
        prompt = case["prompt"]
        expected = case["expected"]

        try:
            result = regression_bridge.generate_3d_from_text(prompt)
            exc = None
        except Exception as err:  # pragma: no cover - defensive
            result = None
            exc = err

        if expected == "should not crash":
            assert (result is not None) or (exc is None), f"Regression resurfaced: {case['id']}"
        elif expected == "should handle unicode correctly":
            assert result is not None, f"Unicode handling failed: {case['id']}"
        elif expected == "should return default primitive":
            assert result is not None, f"Default primitive not returned: {case['id']}"
            assert result.primitive_type in {"cube", "fallback"}, "Unexpected primitive for default case"


def test_performance_regression(regression_bridge):
    """Ensure latency does not regress beyond 10% of the baseline."""
    baseline_path = Path("reports/performance_baseline.json")
    if not baseline_path.exists():
        pytest.skip("Performance baseline not available")

    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)

    test_prompts = baseline.get("test_prompts", [])
    if not test_prompts:
        pytest.skip("Baseline file missing test_prompts list")

    for prompt in test_prompts:
        start = time.perf_counter_ns()
        regression_bridge.generate_3d_from_text(prompt)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6

        baseline_latency = baseline["results"].get(prompt)
        if baseline_latency is None:
            continue
        threshold = baseline_latency * 1.1
        assert elapsed_ms < threshold, (
            f"Performance regression detected for '{prompt}': {elapsed_ms:.2f}ms > {threshold:.2f}ms"
        )


def test_api_contract_stability(regression_bridge):
    """Validate that the API structure remains stable."""
    result = regression_bridge.generate_3d_from_text("test cube")
    assert result is not None

    for attribute in ("vertices", "indices", "primitive_type", "confidence"):
        assert hasattr(result, attribute), f"Missing API attribute: {attribute}"

    assert isinstance(result.vertices, (list, tuple))
    assert isinstance(result.indices, (list, tuple))
    assert isinstance(result.primitive_type, str)
    assert isinstance(result.confidence, (int, float))
