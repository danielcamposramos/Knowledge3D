from __future__ import annotations

from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_benchmark_grammar_rules_are_bootstrapped() -> None:
    grammar = GrammarGalaxy()
    benchmark_ids = {rule.rule_id for rule in grammar.list_benchmark_rules()}

    assert "arc_identity" in benchmark_ids
    assert "gsm_sequential_computation" in benchmark_ids
    assert "apply_power_rule_natural" in benchmark_ids
    assert "benchmark_choice_score_and_emit" in benchmark_ids


def test_benchmark_rule_families_are_queryable() -> None:
    grammar = GrammarGalaxy()

    arc_ids = {rule.rule_id for rule in grammar.list_benchmark_rules("ARC_AGI_2")}
    gsm_ids = {rule.rule_id for rule in grammar.list_benchmark_rules("GSM8K")}
    math_ids = {rule.rule_id for rule in grammar.list_benchmark_rules("MATH")}
    lhe_ids = {rule.rule_id for rule in grammar.list_benchmark_rules("LHE")}

    assert "arc_rotate_then_color" in arc_ids
    assert "gsm_percent_of" in gsm_ids
    assert "gsm_rate_application" in gsm_ids
    assert "apply_fundamental_theorem_calculus_natural" in math_ids
    assert "benchmark_choice_eliminate_then_emit" in lhe_ids


def test_knowledgeverse_default_grammar_contains_benchmark_entries(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path)
    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    rule_ids = {entry.get("rule_id") for entry in grammar.entries}

    assert "arc_flip_h" in rule_ids
    assert "gsm_answer_final_stack" in rule_ids
    assert "apply_product_rule_natural" in rule_ids
    assert "benchmark_choice_score_and_emit" in rule_ids
