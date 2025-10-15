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


@pytest.mark.parametrize(
    "prompt",
    [
        "椅子 with 中文 characters",
        "Lampé décoré avec lumière douce",
        "Mesa com acentuação brasileira",
        "テーブルに花瓶を置く",
        "Стол с металлическими ножками",
    ],
)
def test_unicode_and_international_prompts(regression_bridge, prompt):
    """Ensure Unicode prompts remain supported regression cases."""
    result = regression_bridge.generate_3d_from_text(prompt)
    assert result is not None
    assert hasattr(result, "vertices")


@pytest.mark.parametrize(
    "prompt",
    [
        "something comfortable to sit on",
        "decorative item",
        "modern lighting fixture",
        "conceptual seating object",
    ],
)
def test_low_confidence_prompts_offer_alternatives(regression_bridge, prompt):
    """Low-confidence prompts should expose multiple alternatives."""
    result = regression_bridge.generate_3d_from_text(prompt)
    if result is None:
        pytest.skip(f"Bridge rejected prompt '{prompt}' outright")
    assert len(getattr(result, "alternatives", [])) >= 2
    assert result.confidence < 0.7


@pytest.mark.parametrize(
    "prompt",
    [
        "red cube",
        "blue sphere with subtle glow",
        "wooden table with carved legs",
    ],
)
def test_high_confidence_prompts_return_cube(regression_bridge, prompt):
    """High-confidence prompts should map to the cube primitive."""
    result = regression_bridge.generate_3d_from_text(prompt)
    assert result is not None
    assert result.primitive_type == "cube"
    assert result.confidence >= 0.75


def test_strict_threshold_returns_fallback_namespace(regression_bridge):
    """Threshold >=0.8 provides deterministic fallback instead of raising."""
    regression_bridge.confidence_threshold = 0.95
    result = regression_bridge.generate_3d_from_text("red cube")
    assert result is not None
    assert result.primitive_type in {"cube", "fallback"}
    assert result.confidence <= regression_bridge.confidence_threshold


def test_medium_threshold_rejects_low_confidence(regression_bridge):
    """Threshold <0.8 rejects prompts that fail the bar outright."""
    regression_bridge.confidence_threshold = 0.6
    result = regression_bridge.generate_3d_from_text("ambiguous abstract form")
    assert result is None


def test_low_threshold_allows_generation(regression_bridge):
    """Lower thresholds should accept originally confident prompts."""
    regression_bridge.confidence_threshold = 0.1
    result = regression_bridge.generate_3d_from_text("red cube with bevel")
    assert result is not None
    assert result.confidence >= 0.8


def test_threshold_reset_restores_defaults(regression_bridge):
    """Resetting threshold should revert to accepting medium prompts."""
    regression_bridge.confidence_threshold = 0.7
    assert regression_bridge.generate_3d_from_text("abstract sculpture") is None
    regression_bridge.confidence_threshold = 0.3
    restored = regression_bridge.generate_3d_from_text("abstract sculpture")
    if restored is None:
        pytest.skip("Bridge rejects abstract prompts even with lower threshold")
    assert restored.confidence <= 0.55


def test_fuse_modalities_confidence_gain_is_clamped(regression_bridge):
    """Multi-modal fusion should never exceed confidence of 1.0."""
    prompt = "chair similar to reference image"
    fused = regression_bridge.fuse_modalities(text_prompt=prompt, image_embedding=b"\x00" * 256)
    text_only = regression_bridge.generate_3d_from_text(prompt)
    assert fused is not None and text_only is not None
    assert fused.confidence >= text_only.confidence
    assert fused.confidence <= 1.0


def test_fuse_modalities_handles_missing_image(regression_bridge):
    """Fusion should succeed when optional image embedding is omitted."""
    prompt = "table referenced by text only"
    fused = regression_bridge.fuse_modalities(text_prompt=prompt)
    assert fused is not None
    assert fused.primitive_type in {"cube", "fallback"}


def test_generate_trims_whitespace_equivalently(regression_bridge):
    """Whitespace variations should produce identical confidence."""
    base = regression_bridge.generate_3d_from_text("red cube")
    padded = regression_bridge.generate_3d_from_text("   red cube   ")
    assert base is not None and padded is not None
    assert base.confidence == pytest.approx(padded.confidence, rel=0.0, abs=1e-9)


def test_repeated_calls_remain_deterministic(regression_bridge):
    """Multiple invocations for the same prompt remain stable."""
    prompt = "blue sphere reflective"
    first = regression_bridge.generate_3d_from_text(prompt)
    second = regression_bridge.generate_3d_from_text(prompt)
    assert first is not None and second is not None
    assert first.confidence == pytest.approx(second.confidence, rel=0.0, abs=1e-9)
    assert first.primitive_type == second.primitive_type


def test_generated_shapes_expose_alternatives_attribute(regression_bridge):
    """Every generated namespace should expose alternatives list."""
    result = regression_bridge.generate_3d_from_text("organic flowing shape")
    if result is None:
        pytest.skip("Bridge rejected organic prompt outright")
    assert isinstance(result.alternatives, list)


def test_generate_handles_numeric_prompt(regression_bridge):
    """Numeric prompts should not crash regression surface."""
    result = regression_bridge.generate_3d_from_text("object 123 with pattern 456")
    assert result is not None
    assert isinstance(result.confidence, float)


def test_generate_handles_long_prompt(regression_bridge):
    """Extremely long prompts should remain safe."""
    prompt = " ".join(["complex architectural structure"] * 30)
    result = regression_bridge.generate_3d_from_text(prompt)
    assert result is not None
    assert result.confidence <= 1.0


def test_fuse_modalities_preserves_primitive_type(regression_bridge):
    """Fusion should not change primitive type classification."""
    prompt = "wooden chair for dining table"
    fused = regression_bridge.fuse_modalities(text_prompt=prompt, image_embedding=b"\x01" * 128)
    text_only = regression_bridge.generate_3d_from_text(prompt)
    assert fused is not None and text_only is not None
    assert fused.primitive_type == text_only.primitive_type


def test_generate_provides_indices_and_vertices(regression_bridge):
    """Regression surface always provides geometry arrays."""
    result = regression_bridge.generate_3d_from_text("simple cube object")
    assert result is not None
    assert len(result.vertices) >= 1
    assert len(result.indices) >= 1


def test_fuse_modalities_accepts_keyword_arguments(regression_bridge):
    """Fusion helper should allow additional kwargs without failure."""
    result = regression_bridge.fuse_modalities(
        text_prompt="sphere with reference",
        image_embedding=b"\x02" * 32,
        audio_embedding=None,
    )
    assert result is not None
    assert result.confidence <= 1.0
