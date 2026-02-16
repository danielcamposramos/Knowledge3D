from __future__ import annotations

import pytest

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter, _GeneratedPattern


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


def test_prepare_discovery_examples_adds_negative_form_pairs():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_figure_ground_reversal=True,
    )
    train = [{"input": [[1, 0], [2, 3]], "output": [[0, 1], [3, 2]]}]
    prepared = adapter._prepare_discovery_examples(train)
    assert len(prepared) >= 2
    assert any(
        (row.get("metadata", {}) or {}).get("form_polarity") == "negative"
        for row in prepared
    )


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


def test_forced_navigation_injection_adds_curriculum_patterns():
    adapter = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        enable_forced_navigation_curriculum=True,
        forced_navigation_ratio=1.0,
        forced_navigation_required_galaxies="Math,Reality",
    )
    base = [
        _GeneratedPattern(
            pattern_id="base_0",
            source_galaxy="Drawing",
            target_galaxy="Grammar",
            confidence=0.6,
            query="traditional visual rule: reflect across vertical axis",
            source="traditional",
            pair_index=0,
        )
    ]
    injected = adapter._inject_forced_navigation_patterns(
        train_examples=_sample_task()["train"],
        patterns=base,
    )
    assert len(injected) >= len(base)
    assert any(pattern.source == "curriculum_forced_navigation" for pattern in injected)


def test_forced_navigation_source_expands_galaxy_participation():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_forced_navigation_curriculum=True,
        forced_navigation_ratio=0.5,
        forced_navigation_required_galaxies="Math,Reality",
    )
    galaxies = adapter._extract_pattern_galaxy_set(
        {
            "source": "curriculum_forced_navigation",
            "metadata": {},
        }
    )
    assert "Drawing" in galaxies
    assert "Grammar" in galaxies
    assert "Math" in galaxies
    assert "Reality" in galaxies


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


def test_palette_distribution_score_discriminates_candidates():
    adapter = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    profile = {
        "inferred_family": "spatial_or_recolor",
        "output_palette": [1, 2],
        "output_palette_distribution": {1: 0.75, 2: 0.25},
        "stable_output_palette_size": 2,
    }
    input_grid = [[1, 2], [1, 2]]
    good = [[1, 1], [1, 2]]
    bad = [[1, 2], [1, 2]]
    good_score = adapter._compute_generation_constraint_scores(
        candidate_grid=good,
        input_grid=input_grid,
        profile=profile,
    )["palette_score"]
    bad_score = adapter._compute_generation_constraint_scores(
        candidate_grid=bad,
        input_grid=input_grid,
        profile=profile,
    )["palette_score"]
    assert float(good_score) > float(bad_score)


def test_palette_penalty_weight_increases_penalty_strength():
    components = {
        "family_score": 1.0,
        "shape_score": 1.0,
        "palette_score": 0.4,
        "object_score": 1.0,
    }
    baseline = ArcAgi2Adapter(use_enriched=True, strict_legacy=False)
    palette_heavy = ArcAgi2Adapter(
        use_enriched=True,
        strict_legacy=False,
        palette_penalty_weight=2.0,
    )
    baseline_score = baseline._apply_constraint_penalty(base_score=1.0, components=components)
    palette_heavy_score = palette_heavy._apply_constraint_penalty(base_score=1.0, components=components)
    assert palette_heavy_score < baseline_score


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


def test_full_ptx_validity_path_is_used(monkeypatch):
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_full_ptx=True,
        ptx_validity_strictness="relaxed",
    )
    adapter._full_ptx_available = True

    class _StubPTX:
        def apply_validity_gates_relaxed_ptx(self, *, ranked_candidates, validity_profile, strictness):
            assert strictness == "relaxed"
            return ranked_candidates[:1], {
                "enabled": True,
                "mode": "ptx_validity",
                "strictness": strictness,
                "pre_count": len(ranked_candidates),
                "post_count": 1,
                "filtered_count": max(0, len(ranked_candidates) - 1),
                "fallback_to_ungated": False,
                "family_rejects": 0,
                "shape_rejects": 0,
                "palette_rejects": 0,
                "object_rejects": 0,
                "validity_reject_rate": 0.5,
            }

    monkeypatch.setattr("benchmarks.arc_agi_2_adapter.ARC_PTX_OPS", _StubPTX())
    filtered, report = adapter._apply_validity_gates(
        ranked_candidates=[
            {"candidate": [[1, 0], [0, 1]], "pattern": {"pattern_id": "a"}},
            {"candidate": [[0, 1], [1, 0]], "pattern": {"pattern_id": "b"}},
        ],
        validity_profile={"inferred_family": "spatial"},
    )
    assert len(filtered) == 1
    assert report["mode"] == "ptx_validity"
    assert report["strictness"] == "relaxed"


def test_full_ptx_oracle_path_is_used(monkeypatch):
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_full_ptx=True,
    )
    adapter._full_ptx_available = True

    class _StubPTX:
        def check_oracle_fuzzy_ptx(self, **_kwargs):
            return {
                "oracle_at_3": False,
                "oracle_at_10": True,
                "oracle_at_all": True,
                "correct_rank": 4,
                "oracle_fuzzy_0_80": True,
                "oracle_fuzzy_0_85": True,
                "oracle_fuzzy_0_90": False,
                "oracle_fuzzy_0_95": False,
                "oracle_exact": True,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": True,
                "fuzzy_oracle_at_all": True,
                "fuzzy_best_score": 0.91,
                "fuzzy_best_rank": 4,
            }

    monkeypatch.setattr("benchmarks.arc_agi_2_adapter.ARC_PTX_OPS", _StubPTX())
    metrics = adapter._compute_oracle_metrics(
        ranked_candidates=[{"candidate": [[1, 0], [0, 1]]}],
        expected_output=[[1, 0], [0, 1]],
        fuzzy_threshold=0.95,
    )
    assert metrics["oracle_at_all"] is True
    assert metrics["oracle_at_10"] is True
    assert metrics["ptx_oracle_used"] is True


def test_oracle_rejected_rescue_augments_oracle_metrics_exact():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_oracle_rejected_rescue=True,
        oracle_rejected_rescue_size=4,
        enable_fuzzy_oracle=True,
        fuzzy_oracle_threshold=0.95,
    )
    base_metrics = {
        "oracle_at_3": False,
        "oracle_at_10": False,
        "oracle_at_all": False,
        "correct_rank": None,
        "oracle_fuzzy_0_80": False,
        "oracle_fuzzy_0_85": False,
        "oracle_fuzzy_0_90": False,
        "oracle_fuzzy_0_95": False,
        "oracle_exact": False,
        "fuzzy_oracle_at_3": False,
        "fuzzy_oracle_at_10": False,
        "fuzzy_oracle_at_all": False,
        "fuzzy_best_score": 0.40,
        "fuzzy_best_rank": 0,
        "ptx_oracle_used": False,
    }
    rescue_candidates = [
        {"candidate": [[1, 0], [0, 1]], "score": 0.1, "pattern": {}, "components": {"generation_pass": False}},
    ]
    merged = adapter._augment_oracle_metrics_with_rejected_rescue(
        oracle_metrics=base_metrics,
        rejected_rescue_candidates=rescue_candidates,
        expected_output=[[1, 0], [0, 1]],
        ranked_candidate_count=10,
    )
    assert merged["oracle_rejected_rescue_enabled"] is True
    assert merged["oracle_rejected_rescue_candidate_count"] == 1
    assert merged["oracle_rejected_rescue_exact"] is True
    assert merged["oracle_at_all"] is True
    assert merged["correct_rank"] == 10


def test_build_oracle_rejected_rescue_candidates_skips_existing_signatures():
    adapter = ArcAgi2Adapter(
        use_enriched=False,
        strict_legacy=False,
        enable_oracle_rejected_rescue=True,
        oracle_rejected_rescue_size=8,
    )
    existing_grid = [[1, 1], [0, 0]]
    candidate_map = {
        adapter._grid_signature(existing_grid): (existing_grid, {"pattern_id": "existing"}),
    }
    rejected_reserve = [
        (0.9, existing_grid, {"pattern_id": "dup", "generation_constraint": {"reason": "shape"}}),
        (0.8, [[1, 0], [0, 1]], {"pattern_id": "unique", "generation_constraint": {"reason": "palette"}}),
    ]
    rescue = adapter._build_oracle_rejected_rescue_candidates(
        rejected_reserve=rejected_reserve,
        candidate_map=candidate_map,
    )
    assert len(rescue) == 1
    assert rescue[0]["pattern"]["pattern_id"] == "unique"
    assert rescue[0]["components"]["generation_pass"] is False
