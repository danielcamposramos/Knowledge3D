from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from scripts.populate_reasoning_strategies import (
    build_reasoning_grammar_rules,
    build_reasoning_meaning_entries,
    build_reasoning_meta_rules,
    build_reasoning_reality_entries,
    populate_reasoning_strategies,
)


def test_populate_reasoning_strategies_is_idempotent(tmp_path: Path) -> None:
    house_dir = tmp_path / "house"
    first = populate_reasoning_strategies(house_dir=house_dir)
    second = populate_reasoning_strategies(house_dir=house_dir)

    assert first["reasoning_strategies.jsonl"]["after"] >= 5
    assert first["Reality.jsonl"]["after"] >= 4
    assert first["Grammar.jsonl"]["after"] >= 8
    assert first["Tool.jsonl"]["after"] >= 4

    assert second["reasoning_strategies.jsonl"]["appended"] == 0
    assert second["Reality.jsonl"]["appended"] == 0
    assert second["Grammar.jsonl"]["appended"] == 0
    assert second["Tool.jsonl"]["appended"] == 0

    rows = [
        json.loads(line)
        for line in (house_dir / "reasoning_strategies.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = {row["id"] for row in rows}
    assert "forward_entity_extraction" in ids
    assert "backward_goal_tracing" in ids
    assert "operation_chain_construction" in ids
    assert "result_normalization_validation" in ids
    assert "word_problem_multi_step_reasoning" in ids
    assert "question_subject_domain_routing" in ids
    assert "question_option_elimination" in ids
    assert "question_factual_lookup_validation" in ids
    assert "reasoning_quantity_role_initial" in ids
    assert "reasoning_quantity_role_delta" in ids
    assert "anti_pattern_router_final_answer" in ids
    assert "anti_pattern_isolated_template_halt" in ids
    forward_row = next(row for row in rows if row["id"] == "forward_entity_extraction")
    assert forward_row["selection_role"] == "unknown"
    assert forward_row["layer_id"] == 0
    assert forward_row["answer_eligible"] is False
    assert forward_row["sovereign_route_exempt"] is True

    grammar_rows = [
        json.loads(line)
        for line in (house_dir / "Grammar.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grammar_ids = {row["id"] for row in grammar_rows}
    assert "grammar_forward_entity_extraction" in grammar_ids
    assert "grammar_backward_goal_tracing" in grammar_ids
    assert "grammar_operation_chain_construction" in grammar_ids
    assert "grammar_recursive_subtask_decomposition" in grammar_ids
    assert "grammar_validate_units_and_magnitude" in grammar_ids
    assert "grammar_subject_domain_alignment" in grammar_ids
    assert "grammar_option_elimination" in grammar_ids
    assert "grammar_factual_lookup" in grammar_ids
    assert "grammar_option_verification" in grammar_ids
    grammar_forward = next(row for row in grammar_rows if row["id"] == "grammar_forward_entity_extraction")
    assert grammar_forward["route_family"] == "GRAMMAR"
    assert grammar_forward["selection_role"] == "executor"
    assert grammar_forward["layer_id"] == 3
    assert grammar_forward["answer_eligible"] is False
    assert grammar_forward["route_policy"] == {
        "requires_validator": True,
        "answer_gate": False,
        "branch_topk": 2,
    }
    assert grammar_forward["validator_refs"] == [
        "grammar_normalization_validator",
        "grammar_answer_validator",
    ]

    tool_rows = [
        json.loads(line)
        for line in (house_dir / "Tool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_ids = {row["id"] for row in tool_rows}
    assert "meta_four_way_reading_strategy" in tool_ids
    assert "meta_decompose_multi_step_word_problem" in tool_ids
    assert "meta_apply_backward_trace_before_emit" in tool_ids
    assert "meta_validate_units_before_answer" in tool_ids
    assert "meta_verify_option_before_emit" in tool_ids
    assert "meta_avoid_router_as_final_answer" in tool_ids
    assert "meta_avoid_isolated_template_halt" in tool_ids


def test_reasoning_strategies_are_visible_through_house_loader(tmp_path: Path) -> None:
    root = tmp_path
    (root / "galaxies").mkdir(parents=True)
    (root / "house").mkdir(parents=True)
    (root / "galaxies" / "Reality.jsonl").write_text("", encoding="utf-8")
    (root / "galaxies" / "Grammar.jsonl").write_text("", encoding="utf-8")
    (root / "galaxies" / "Tool.jsonl").write_text("", encoding="utf-8")
    populate_reasoning_strategies(house_dir=root / "house")

    manager = GalaxyManager(storage_root=root / "galaxies", extra_storage_roots=[root / "house"])
    reasoning_entries = manager._read_entries_from_disk("reasoning_strategies")
    reasoning_ids = {entry["id"] for entry in reasoning_entries}
    assert "forward_entity_extraction" in reasoning_ids
    assert "word_problem_multi_step_reasoning" in reasoning_ids
    assert "question_subject_domain_routing" in reasoning_ids
    assert "anti_pattern_router_final_answer" in reasoning_ids
    assert "reasoning_quantity_role_initial" in reasoning_ids

    grammar_entries = manager._read_entries_from_disk("Grammar")
    grammar_ids = {entry["id"] for entry in grammar_entries}
    assert "grammar_dependency_dag_build" in grammar_ids
    assert "grammar_result_normalization" in grammar_ids
    assert "grammar_subject_domain_alignment" in grammar_ids
    assert "grammar_option_verification" in grammar_ids

    tool_entries = manager._read_entries_from_disk("Tool")
    tool_ids = {entry["id"] for entry in tool_entries}
    assert "meta_four_way_reading_strategy" in tool_ids
    assert "meta_verify_option_before_emit" in tool_ids

    reality_entries = manager._read_entries_from_disk("Reality")
    reality_ids = {entry["id"] for entry in reality_entries}
    assert "reality_dependency_dag" in reality_ids
    assert "reality_unit_consistency" in reality_ids

    stars = load_all_galaxies_from_disk(root)
    star_ids = {str(star.get("_id") or star.get("id") or "") for star in stars}
    assert "forward_entity_extraction" in star_ids
    assert "grammar_operation_chain_construction" in star_ids
    assert "meta_validate_units_before_answer" in star_ids
    assert "question_factual_lookup_validation" in star_ids
    assert "anti_pattern_isolated_template_halt" in star_ids

    assert len(build_reasoning_meaning_entries()) >= 5
    assert len(build_reasoning_reality_entries()) >= 4
    assert len(build_reasoning_grammar_rules()) >= 8
    assert len(build_reasoning_meta_rules()) >= 4
