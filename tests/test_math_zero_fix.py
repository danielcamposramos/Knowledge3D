from __future__ import annotations

import json

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from scripts.ingest_math_rules import build_math_rule_entries, ingest_math_rules
from scripts.ingest_meaning_layer import (
    build_language_math_bridge_entry,
    ingest_enriched_galaxy,
)


class _NeverEvalEngine:
    def evaluate(self, _program: str) -> float:
        raise AssertionError("math fallback should not attempt direct evaluation here")


def test_answer_math_query_never_returns_name_fallback(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_zero")
    result = kv._answer_math_query(
        task={"type": "MATH_TASK", "question": "Unseen hard problem", "query": "Unseen hard problem"},
        binding={"galaxies": ["Math"], "entry_count": 1},
        reasoning_program={"id": "reasoning_math_top1"},
        route_galaxies=["Math"],
        match={
            "id": "en_service_area",
            "name": "Sum All Values",
            "domain": "math",
            "category": "rule",
            "rpn_program": "",
            "metadata": {},
            "index": 0,
            "confidence": 0.0,
        },
        similarity=0.15,
        engine=_NeverEvalEngine(),
        specialist="math",
        domain_hint="math",
        query_text="Unseen hard problem",
        use_enriched=True,
        query_type="math",
        selection_steps=[],
        best_candidate=None,
    )

    assert result["answer"] == ""
    assert result["result"] == ""
    assert any("unresolved" in step.lower() for step in result["reasoning_trace"])


def test_build_language_math_bridge_entry_keeps_language_and_math_refs() -> None:
    star = MeaningCentricStar(
        star_id="synset_addition",
        meaning_class="concept",
        meaning_rpn="SYNSET N ADDITION DEF the process of adding",
        domain="Foundation/Language",
        surface_forms={
            "en": SurfaceForm(word_ref="addition", char_refs=["char_a"]),
            "es": SurfaceForm(word_ref="adicion", char_refs=["char_es_a"]),
            "pt": SurfaceForm(word_ref="adicao", char_refs=["char_pt_a"]),
        },
    )

    bridge = build_language_math_bridge_entry(star)

    assert bridge["id"] == "math_exec_synset_addition"
    assert bridge["symlink_to"] == "synset_addition"
    assert bridge["metadata"]["language_star_ref"] == "synset_addition"
    assert bridge["metadata"]["direct_eval"] is False
    assert bridge["metadata"]["template_ref"] == "math_template_arithmetic_chain_gpu"


def test_ingest_enriched_galaxy_keeps_math_meaning_in_language_and_math_bridge(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_meaning_ingest")
    star = MeaningCentricStar(
        star_id="synset_addition",
        meaning_class="concept",
        meaning_rpn="SYNSET N ADDITION DEF the process of combining numbers",
        domain="Foundation/Language",
        surface_forms={
            "en": SurfaceForm(word_ref="addition", char_refs=["char_a"]),
            "es": SurfaceForm(word_ref="adicion", char_refs=["char_es_a"]),
            "pt": SurfaceForm(word_ref="adicao", char_refs=["char_pt_a"]),
            "fr": SurfaceForm(word_ref="addition", char_refs=["char_fr_a"]),
            "de": SurfaceForm(word_ref="addition", char_refs=["char_de_a"]),
        },
    )
    meaning_path = tmp_path / "meaning.jsonl"
    with meaning_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(star.to_dict(), ensure_ascii=False) + "\n")

    summary = ingest_enriched_galaxy(
        kv,
        meaning_path=meaning_path,
        mmlu_path=tmp_path / "missing_mmlu.jsonl",
        gsm8k_path=tmp_path / "missing_gsm8k.jsonl",
        full_load=True,
        min_languages=3,
    )

    assert summary["meaning_stars_loaded"] == 1
    language_entries = list(kv.galaxy_manager.get_galaxy("Language").entries)
    math_entries = list(kv.galaxy_manager.get_galaxy("Math").entries)
    language_entry = next(entry for entry in language_entries if entry["id"] == "synset_addition")
    math_entry = next(entry for entry in math_entries if entry["id"] == "math_exec_synset_addition")

    assert language_entry["metadata"]["math_galaxy_ref"] == "math_exec_synset_addition"
    assert math_entry["metadata"]["language_star_ref"] == "synset_addition"
    assert math_entry["symlink_to"] == "synset_addition"


def test_build_math_rule_entries_exceeds_500_and_covers_all_math_types() -> None:
    entries = build_math_rule_entries()

    assert len(entries) >= 500
    math_types = {
        str((entry.get("metadata") or {}).get("math_type", "")).strip()
        for entry in entries
        if isinstance(entry.get("metadata"), dict)
    }
    assert {
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus",
    }.issubset(math_types)


def test_ingest_math_rules_populates_math_galaxy_with_template_programs(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_rules")
    summary = ingest_math_rules(kv)

    assert summary["total_entries"] >= 500
    math_entries = list(kv.galaxy_manager.get_galaxy("Math").entries)
    ids = {str(entry.get("id", "")).strip() for entry in math_entries}
    assert "math_template_permutation_gpu" in ids
    assert "math_template_gcd_gpu" in ids
    assert "math_template_lcm_gpu" in ids


def test_new_math_templates_evaluate_exact_integer_results(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_templates")
    engine = kv.get_gpu_reasoning_engine()

    permutation = kv._evaluate_math_template(
        engine=engine,
        match={"metadata": {"template_ref": "math_template_permutation_gpu"}},
        query_text="How many permutations of 5 objects taken 3 at a time are possible?",
    )
    gcd_value = kv._evaluate_math_template(
        engine=engine,
        match={"metadata": {"template_ref": "math_template_gcd_gpu"}},
        query_text="What is the greatest common divisor of 48 and 18?",
    )
    lcm_value = kv._evaluate_math_template(
        engine=engine,
        match={"metadata": {"template_ref": "math_template_lcm_gpu"}},
        query_text="What is the least common multiple of 12 and 18?",
    )
    remainder = kv._evaluate_math_template(
        engine=engine,
        match={"metadata": {"template_ref": "math_template_remainder_gpu"}},
        query_text="What is the remainder when 29 is divided by 6?",
    )

    assert permutation is not None and permutation[0] == "60"
    assert gcd_value is not None and gcd_value[0] == "6"
    assert lcm_value is not None and lcm_value[0] == "36"
    assert remainder is not None and remainder[0] == "5"
