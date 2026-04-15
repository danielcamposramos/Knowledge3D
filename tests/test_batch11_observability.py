from __future__ import annotations

from knowledge3d.ingestion.canonical_curriculum_loader import _target_galaxy_with_subkind
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from scripts.run_headless_tablet_benchmarks import _collapse_attractors, _trace_coverage_report


def _star(*, star_id: str, domain: str, meaning_class: str = "concept") -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class=meaning_class,
        meaning_rpn="GALAXY_LOOKUP TEST RECALL",
        domain=domain,
        taxonomy_refs=[],
        surface_forms={},
        visual_rpn=None,
        visual_refs=[],
        audio_rpn=None,
        audio_refs=[],
        pronunciations={},
        behavior_rpn=None,
        reality_refs=[],
        grammar_refs=[],
        meta_refs=[],
        house_position=[0.0, 0.0, 0.0],
        house_room="",
        galaxy_ref=domain,
        confidence=80,
        polarity=0,
        component_refs=[],
        composite_of=[],
    )


def test_batch11_target_galaxy_routes_new_domains() -> None:
    assert _target_galaxy_with_subkind(_star(star_id="concept.civics.demo", domain="civics")) == "Reality"
    assert _target_galaxy_with_subkind(_star(star_id="concept.economics.demo", domain="economics")) == "Reality"
    assert _target_galaxy_with_subkind(_star(star_id="concept.psychology.demo", domain="psychology")) == "Reality"
    assert _target_galaxy_with_subkind(_star(star_id="concept.arc.demo", domain="arc")) == "Tool"
    assert _target_galaxy_with_subkind(_star(star_id="concept.drawing.demo", domain="drawing")) == "Drawing"
    assert _target_galaxy_with_subkind(_star(star_id="concept.literature.demo", domain="literature")) == "Language"


def test_batch11_trace_coverage_report_detects_gaps_and_collapse() -> None:
    result = {
        "results": [
            {"id": "q1"},
            {"id": "q2"},
            {"id": "q3"},
        ]
    }
    traces = [
        {
            "item_id": "q1",
            "route_family": "MATH",
            "specialist_lane": "math",
            "stars_touched": ["router_a", "fact_a"],
            "stars_recalled": ["fact_a"],
            "opcodes_fired": ["gpu_task_dispatch_sovereign"],
            "normalized_answer": "98",
            "correct": False,
            "program_id": "gpu_task_dispatch_sovereign",
        },
        {
            "item_id": "q2",
            "route_family": "MATH",
            "specialist_lane": "math",
            "stars_touched": ["router_b", "fact_b"],
            "stars_recalled": ["fact_b"],
            "opcodes_fired": ["gpu_task_dispatch_sovereign"],
            "normalized_answer": "98",
            "correct": False,
            "program_id": "gpu_task_dispatch_sovereign",
        },
    ]

    coverage = _trace_coverage_report(suite_name="gsm8k", result=result, traces=traces)

    assert coverage["missing_item_ids"] == ["q3"]
    assert coverage["distinct_stars_touched"] == 4
    assert coverage["distinct_stars_recalled"] == 2
    assert coverage["touched_but_never_recalled"][0]["star_id"] in {"router_a", "router_b"}
    assert coverage["recalled_but_wrong"][0]["star_id"] in {"fact_a", "fact_b"}
    assert coverage["collapse_attractors"][0]["normalized_answer"] == "98"


def test_batch11_collapse_attractors_threshold() -> None:
    traces = [
        {"normalized_answer": "20", "correct": False},
        {"normalized_answer": "20", "correct": False},
        {"normalized_answer": "20", "correct": False},
        {"normalized_answer": "5", "correct": True},
    ]
    attractors = _collapse_attractors(traces)
    assert attractors
    assert attractors[0]["normalized_answer"] == "20"
