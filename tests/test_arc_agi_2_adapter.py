from __future__ import annotations

import pytest

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter


def _sample_task() -> dict:
    return {
        "id": "adapter_test",
        "train": [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}],
        "test": [{"input": [[9, 0], [1, 2]], "output": [[0, 9], [2, 1]]}],
    }


def test_adapter_fallback_when_pipeline_unavailable():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    adapter.pipeline = None

    def fallback(task: dict, use_enriched: bool) -> dict:
        assert task["id"] == "adapter_test"
        assert use_enriched is False
        return {
            "task_id": task["id"],
            "correct": True,
            "exact_match": True,
            "predicted": task["test"][0]["output"],
            "expected": task["test"][0]["output"],
            "reasoning_trace": ["fallback_used"],
            "patterns_used": 1,
            "score": 1.0,
            "fuzzy_score": 1.0,
        }

    result = adapter.solve_task(_sample_task(), fallback_solver=fallback)
    assert result["solver"] == "trm_navigator_fallback"
    assert result["correct"] is True
    assert "fallback_reason" in result


def test_adapter_strict_mode_raises_without_pipeline():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    adapter.pipeline = None
    adapter.strict_legacy = True
    with pytest.raises(RuntimeError, match="Legacy ARC pipeline unavailable"):
        adapter.solve_task(_sample_task())

