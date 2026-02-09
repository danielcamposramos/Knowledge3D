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


def test_describe_visual_transformation_reflection_and_color():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    desc = adapter._describe_visual_transformation(
        [[1, 2], [3, 4]],
        [[2, 1], [4, 3]],
    )
    assert "reflect across vertical axis" in desc

    recolor_desc = adapter._describe_visual_transformation(
        [[1, 1], [2, 2]],
        [[3, 3], [4, 4]],
    )
    assert "color transformation" in recolor_desc or "recolor" in recolor_desc


class _FakeNavigator:
    def generate_from_procedural(self, **kwargs):
        return {
            "id": "gen_rule_1",
            "metadata": {
                "source_galaxy": kwargs.get("source_galaxy", "3DObjects"),
                "confidence": 0.82,
            },
        }

    def navigate_and_compose(self, **_kwargs):
        return {
            "candidates": [
                {"entry": {"id": "cross_modal_rule_1"}, "confidence": 0.76},
                {"entry": {"id": "cross_modal_rule_2"}, "score": 0.71},
            ]
        }


class _FakeKV:
    def __init__(self):
        self.trm_navigator = _FakeNavigator()
        self.events = []

    def log_event(self, event_type: str, event_data: dict):
        self.events.append((event_type, event_data))


def test_discover_patterns_includes_all_sources():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=kv)
    patterns = adapter.discover_patterns(_sample_task()["train"])
    assert patterns
    sources = {pattern.source for pattern in patterns}
    assert "traditional" in sources
    assert "autonomous_generation" in sources
    assert "multi_galaxy_composition" in sources


def test_discover_patterns_contrastive_adds_anti_patterns():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        knowledgeverse=kv,
        enable_contrastive_learning=True,
    )
    patterns = adapter.discover_patterns(_sample_task()["train"])
    sources = {pattern.source for pattern in patterns}
    assert "contrastive_anti" in sources


def test_rank_candidates_prefers_autonomous_and_cross_modal_signals():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=_FakeKV())
    candidates = [
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]],
    ]
    patterns = [
        {
            "pattern_id": "traditional_low",
            "source": "traditional",
            "confidence": 0.55,
            "metadata": {"composition_depth": 1, "reuse_count": 1},
        },
        {
            "pattern_id": "autonomous_high",
            "source": "autonomous_generation",
            "confidence": 0.82,
            "metadata": {
                "composition_depth": 3,
                "reuse_count": 8,
                "source_galaxy": "Drawing+Math+Reality",
                "cross_modal": True,
            },
        },
    ]
    ranked = adapter._rank_candidates(candidates, patterns)
    assert ranked
    assert ranked[0]["pattern"]["pattern_id"] == "autonomous_high"
    assert ranked[0]["score"] > ranked[1]["score"]


class _FakePipelineResult:
    def __init__(self, output_grid):
        self.output_grid = output_grid
        self.correct = True
        self.score = 0.9
        self.fuzzy_score = 0.9
        self.best_program = "GRID 2 2 FILL"
        self.program_type = "test_program"
        self.signature = "sig:test"


class _FakePipeline:
    def process_task(self, **kwargs):
        test_input = kwargs["test_input"]
        # Predict horizontal flip to match _sample_task expected output.
        output = [list(reversed(row)) for row in test_input]
        return _FakePipelineResult(output)


def test_solve_task_emits_oracle_and_ranking_diagnostics():
    kv = _FakeKV()
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False, knowledgeverse=kv)
    adapter.pipeline = _FakePipeline()

    result = adapter.solve_task(_sample_task())

    assert result["correct"] is True
    assert result["legacy_correct"] is True
    assert "oracle_at_3" in result
    assert "oracle_at_10" in result
    assert "oracle_at_all" in result
    assert result["oracle_at_all"] is True
    assert result["correct_rank"] is not None
    assert "ranking_changed_top1" in result
    assert "ranking_score_range" in result
    assert "ranking_score_stddev" in result
    assert isinstance(result["ranking_top_5_scores"], list)
    assert isinstance(result["ranking_top_5_sources"], list)

    event_types = [event_type for event_type, _ in kv.events]
    assert "arc_candidate_ranking" in event_types
    assert "arc_ranking_scores" in event_types


def test_validity_profile_infers_family_and_expected_shape():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
    ]
    profile = adapter._build_validity_profile(train_examples=train, test_input=[[9, 0], [1, 2]])
    assert profile["inferred_family"] in {"spatial", "spatial_or_recolor"}
    assert profile["expected_shape"] == (2, 2)


def test_candidate_validity_rejects_family_mismatch():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    train = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}]
    profile = adapter._build_validity_profile(train_examples=train, test_input=[[9, 0], [1, 2]])
    # Mismatch family: scaling output when family inferred as spatial.
    scaled = [[9, 9, 0, 0], [9, 9, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]]
    ok, reason = adapter._candidate_passes_validity(scaled, profile)
    assert ok is False
    assert reason in {"family", "shape"}


def test_oracle_metrics_include_stratified_fuzzy_keys():
    adapter = ArcAgi2Adapter(use_enriched=False, strict_legacy=False)
    expected = [[1, 0], [0, 1]]
    candidates = [
        {"candidate": [[1, 0], [0, 1]]},
        {"candidate": [[1, 1], [0, 1]]},
    ]
    metrics = adapter._compute_oracle_metrics(candidates, expected, fuzzy_threshold=0.95)
    assert metrics["oracle_at_all"] is True
    assert "oracle_fuzzy_0_80" in metrics
    assert "oracle_fuzzy_0_85" in metrics
    assert "oracle_fuzzy_0_90" in metrics
    assert "oracle_fuzzy_0_95" in metrics
    assert "oracle_exact" in metrics
