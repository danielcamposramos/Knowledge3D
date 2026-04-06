from __future__ import annotations

import json
from collections import Counter

from benchmarks.math_competitions import UnifiedMathBenchmark, math_answers_match, normalize_latex_answer
from knowledge3d.tools.benchmark_health_check import evaluate_answer
from knowledge3d.knowledgeverse.foundational_operations_bootstrap import _benchmark_math_entries
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from scripts.ingest_math_rules import build_math_rule_entries, build_rule_catalog, ingest_math_rules
from scripts.ingest_meaning_layer import (
    build_language_math_bridge_entry,
    build_math_language_symlink_entries,
    ingest_enriched_galaxy,
)


class _NeverEvalEngine:
    def evaluate(self, _program: str) -> float:
        raise AssertionError("math fallback should not attempt direct evaluation here")


def test_answer_math_query_never_returns_name_fallback(tmp_path) -> None:
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_math_zero",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
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
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_meaning_ingest",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
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
    language_entry = next(entry for entry in language_entries if entry["id"] == "language_synset_addition")
    math_entry = next(entry for entry in math_entries if entry["id"] == "math_exec_synset_addition")

    assert language_entry["metadata"]["meaning_star_id"] == "synset_addition"
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


def test_rule_catalog_expands_high_roi_math_families() -> None:
    specs = build_rule_catalog()
    ids = {str(spec["id"]).strip() for spec in specs}
    counts = Counter(str(spec["math_type"]).strip() for spec in specs)

    assert counts["Prealgebra"] >= 15
    assert {
        "prealgebra_arithmetic_operations",
        "prealgebra_percentage_conversion",
        "number_theory_gcd_euclidean",
        "geometry_coordinate_distance",
        "algebra_linear_equation_one_var",
        "counting_conditional_probability_family",
        "intermediate_algebra_exponential_growth_family",
        "precalculus_matrix_determinant_family",
    }.issubset(ids)


def test_math_language_symlinks_cover_expanded_rule_catalog() -> None:
    symlinks = build_math_language_symlink_entries()
    ids = {str(entry["id"]).strip() for entry in symlinks}
    lookup = {str(entry["id"]).strip(): entry for entry in symlinks}

    assert len(symlinks) >= len(build_rule_catalog())
    assert "lang_math_symlink_prealgebra_arithmetic_operations" in ids
    assert "lang_math_symlink_geometry_coordinate_distance" in ids

    entry = lookup["lang_math_symlink_prealgebra_arithmetic_operations"]
    assert entry["metadata"]["symlink_target"] == "math_anchor_prealgebra_arithmetic_operations"
    assert entry["metadata"]["symlink_galaxy"] == "Math"
    assert "arithmetic" in entry["metadata"]["query_anchor"]


def test_ingest_math_rules_populates_math_galaxy_with_template_programs(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_rules")
    summary = ingest_math_rules(kv)

    assert summary["total_entries"] >= 500
    math_entries = list(kv.galaxy_manager.get_galaxy("Math").entries)
    ids = {str(entry.get("id", "")).strip() for entry in math_entries}
    assert "math_template_permutation_gpu" in ids
    assert "math_template_gcd_gpu" in ids
    assert "math_template_lcm_gpu" in ids
    assert "math_template_triangle_area_gpu" in ids
    assert "math_template_circle_area_gpu" in ids
    assert "math_template_rate_scaling_gpu" in ids
    assert len([entry for entry in math_entries if entry.get("category") == "template_program"]) >= 60


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


def test_generic_high_roi_templates_execute_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_high_roi_templates")
    engine = kv.get_gpu_reasoning_engine()
    entries = {str(entry["id"]).strip(): entry for entry in build_math_rule_entries()}

    triangle = kv._evaluate_math_template(
        engine=engine,
        match=entries["math_template_triangle_area_gpu"],
        query_text="A triangle has base 10 and height 4. Find its area.",
    )
    pythagorean = kv._evaluate_math_template(
        engine=engine,
        match=entries["math_template_pythagorean_hypotenuse_gpu"],
        query_text="A right triangle has legs 3 and 4. What is the hypotenuse?",
    )
    determinant = kv._evaluate_math_template(
        engine=engine,
        match=entries["math_template_determinant_2x2_gpu"],
        query_text="Find the determinant of [[2, 3], [1, 4]].",
    )

    assert triangle is not None and triangle[0] == "20"
    assert pythagorean is not None and pythagorean[0] == "5"
    assert determinant is not None and determinant[0] == "5"


def test_latex_normalization_handles_common_numeric_forms() -> None:
    assert math_answers_match("0.75", r"\frac{3}{4}")
    assert math_answers_match("0.5", r"\dfrac{1}{2}")
    assert math_answers_match("2", r"\sqrt[3]{8}")
    assert math_answers_match(str(2 * 3**0.5), r"2\sqrt{3}")
    assert math_answers_match(str(2 * 3.141592653589793), r"2\pi")
    assert math_answers_match(str((1 + 5**0.5) / 2), r"\frac{1+\sqrt{5}}{2}")
    assert math_answers_match("12345", r"12\,345")
    assert normalize_latex_answer(r"\left(\frac{3}{4}\right)").replace(" ", "") in {"(((3)/(4)))", "((((3)/(4))))"}
    assert math_answers_match("monday", r"\text{Monday}")


def test_health_check_math_evaluator_accepts_latex_expected_answers() -> None:
    assert evaluate_answer("math", "0.75", r"\frac{3}{4}") is True
    assert evaluate_answer("gsm8k", "18", "18") is True


def test_unified_math_benchmark_loads_both_math_and_math_sources(tmp_path) -> None:
    dataset_dir = tmp_path / "datasets"
    (dataset_dir / "data").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "gsm8k").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data" / "train.jsonl").write_text(
        json.dumps(
            {
                "problem": r"Solve for x: 2x = 3.",
                "solution": r"The answer is \boxed{\frac{3}{2}}.",
                "type": "Algebra",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gsm_path = dataset_dir / "gsm8k" / "test.jsonl"
    gsm_path.write_text(
        json.dumps(
            {
                "question": "If you have 10 apples and eat 4, how many remain?",
                "answer": "They remain 6. #### 6",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bench = UnifiedMathBenchmark(
        knowledgeverse=Knowledgeverse(storage_root=tmp_path / "kv_unified_math"),
        dataset_path=dataset_dir,
        gsm8k_dataset_path=gsm_path,
        max_problems=1,
        max_math_questions=1,
        source_filter=["math", "gsm8k"],
    )

    assert len(bench.problems) == 2
    assert {str(problem["suite"]) for problem in bench.problems} == {"math", "gsm8k"}
    assert any(problem["answer"] == r"\frac{3}{2}" for problem in bench.problems if problem["suite"] == "math")
    assert any(problem["answer"] == "6" for problem in bench.problems if problem["suite"] == "gsm8k")


def test_math_benchmark_question_anchor_executes_without_answer_leak(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_benchmark_anchor")
    engine = kv.get_gpu_reasoning_engine()
    entry = next(entry for entry in _benchmark_math_entries() if entry["id"] == "benchmark_math_math_0_direct")
    metadata = dict(entry["metadata"])
    metadata.pop("template_ref", None)
    metadata.pop("template_params", None)
    match = {**entry, "metadata": metadata}
    task = {
        "type": "MATH_TASK",
        "task_id": "math_0",
        "competition": "MATH:Algebra",
        "question": entry["content"],
        "query": entry["content"],
        "expected_answer": "0",
    }

    assert kv._is_answer_bearing_benchmark_shortcut(
        entry=match,
        task=task,
        query_text=entry["content"],
    ) is False
    assert kv._benchmark_navigation_entry_allowed(
        entry=match,
        task_type="MATH_TASK",
        task=task,
        query_text=entry["content"],
    ) is True

    result = kv._answer_math_query(
        task=task,
        binding={"galaxies": ["Math"], "entry_count": 1},
        reasoning_program={"id": "reasoning_math_template_match_top1"},
        route_galaxies=["Math"],
        match=match,
        similarity=1.0,
        engine=engine,
        specialist="math",
        domain_hint="algebra",
        query_text=entry["content"],
        use_enriched=True,
        query_type="math",
        selection_steps=[],
        best_candidate=None,
    )

    assert result["answer"] == "0"
