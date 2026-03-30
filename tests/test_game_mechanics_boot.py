from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from scripts.populate_game_mechanics import (
    build_game_grammar_rules,
    build_game_mechanics_entries,
    build_game_meta_rules,
    build_game_reality_entries,
    populate_game_knowledge,
)


def test_populate_game_mechanics_is_idempotent(tmp_path: Path) -> None:
    house_dir = tmp_path / "house"
    first = populate_game_knowledge(house_dir=house_dir)
    second = populate_game_knowledge(house_dir=house_dir)

    assert first["game_mechanics.jsonl"]["after"] >= 100
    assert first["Reality.jsonl"]["after"] >= 20
    assert first["Grammar.jsonl"]["after"] >= 20
    assert first["Tool.jsonl"]["after"] >= 15

    assert second["game_mechanics.jsonl"]["appended"] == 0
    assert second["Reality.jsonl"]["appended"] == 0
    assert second["Grammar.jsonl"]["appended"] == 0
    assert second["Tool.jsonl"]["appended"] == 0

    path = house_dir / "game_mechanics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row["id"] for row in rows}
    assert "switch_actuator" in ids
    assert "movement_recharge_block" in ids
    assert "no_instruction_discovery" in ids
    assert "movement_wall_jump" in ids
    assert "object_pushable_crate" in ids
    assert "puzzle_sokoban_alignment" in ids
    assert "input_action_click" in ids
    assert "visual_flashing_temporary" in ids
    assert "screen_transition_uniform_color" in ids
    assert "level_design_hub_spoke" in ids
    assert "state_machine_start_prompt" in ids
    assert "screen_transition_dismiss" in ids
    assert "post_transition_new_context" in ids
    assert "movement_budget_visual_bar" in ids
    assert "movement_budget_depletion_penalty" in ids
    assert "movement_budget_conservation" in ids
    assert "lives_system" in ids
    assert "lives_visual_indicator" in ids
    assert "strategic_reset" in ids
    assert "budget_sufficiency_check" in ids
    assert "screen_flash_failure" in ids
    assert "screen_flash_color_semantics" in ids
    assert "reference_box_current_state" in ids
    assert "level_count_indicator" in ids
    assert rows[0]["meaning_rpn"]
    assert rows[0]["behavior_rpn"]
    assert rows[0]["meta_refs"]
    categories = {
        str((row.get("metadata") or {}).get("properties", {}).get("knowledge_category", "seed"))
        for row in rows
    }
    assert {
        "movement",
        "objects",
        "puzzles",
        "controls",
        "visual_encoding",
        "level_design",
        "state_machine",
    }.issubset(categories)

    grammar_rows = [
        json.loads(line)
        for line in (house_dir / "Grammar.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grammar_ids = {row["id"] for row in grammar_rows}
    assert "grammar_unlock_door_on_pattern_match" in grammar_ids
    assert "grammar_infer_rule_from_state_delta" in grammar_ids
    assert "grammar_apply_gravity" in grammar_ids
    assert "grammar_transition_game_state" in grammar_ids
    assert "grammar_detect_transition_screen" in grammar_ids
    assert "grammar_reperceive_after_transition" in grammar_ids
    assert "grammar_decode_movement_budget_bar" in grammar_ids
    assert "grammar_decode_lives_indicator" in grammar_ids
    assert "grammar_trigger_strategic_reset" in grammar_ids
    assert "grammar_classify_failure_flash" in grammar_ids
    assert "grammar_compare_reference_box_to_target" in grammar_ids

    tool_rows = [
        json.loads(line)
        for line in (house_dir / "Tool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_ids = {row["id"] for row in tool_rows}
    assert "meta_route_to_switch_when_locked" in tool_ids
    assert "meta_click_to_advance_after_completion" in tool_ids
    assert "meta_start_from_title_state" in tool_ids
    assert "meta_read_visual_affordance_before_action" in tool_ids
    assert "meta_dismiss_transition_before_navigation" in tool_ids
    assert "meta_reset_context_after_level_completion" in tool_ids
    assert "meta_compare_budget_to_route_before_commit" in tool_ids
    assert "meta_reset_before_budget_depletion" in tool_ids
    assert "meta_reperceive_after_failure_flash" in tool_ids


def test_galaxy_manager_reads_house_jsonl_entries(tmp_path: Path) -> None:
    galaxies_dir = tmp_path / "galaxies"
    house_dir = tmp_path / "house"
    galaxies_dir.mkdir(parents=True)
    house_dir.mkdir(parents=True)
    (galaxies_dir / "Reality.jsonl").write_text("", encoding="utf-8")
    (galaxies_dir / "Grammar.jsonl").write_text("", encoding="utf-8")
    (galaxies_dir / "Tool.jsonl").write_text("", encoding="utf-8")
    populate_game_knowledge(house_dir=house_dir)

    manager = GalaxyManager(storage_root=galaxies_dir, extra_storage_roots=[house_dir])
    paths = manager.iter_storage_jsonl_paths()
    stems = [path.stem for path in paths]

    assert "Reality" in stems
    assert "Grammar" in stems
    assert "Tool" in stems
    assert "game_mechanics" in stems

    entries = manager._read_entries_from_disk("game_mechanics")
    entry_ids = {entry["id"] for entry in entries}
    assert "lock_key_pattern_match" in entry_ids
    assert "shape_transform_block" in entry_ids
    assert "movement_teleporter_pair" in entry_ids
    assert "level_design_rising_difficulty" in entry_ids

    grammar_entries = manager._read_entries_from_disk("Grammar")
    grammar_ids = {entry["id"] for entry in grammar_entries}
    assert "grammar_route_agent_on_walkable_grid" in grammar_ids
    assert "grammar_complete_level_after_goal_cross" in grammar_ids
    assert "grammar_apply_swim_motion" in grammar_ids
    assert "grammar_decode_visual_signal" in grammar_ids

    tool_entries = manager._read_entries_from_disk("Tool")
    tool_ids = {entry["id"] for entry in tool_entries}
    assert "meta_match_key_before_door" in tool_ids
    assert "meta_learn_from_visual_transition" in tool_ids
    assert "meta_backtrack_when_dependency_unsatisfied" in tool_ids

    reality_entries = manager._read_entries_from_disk("Reality")
    reality_ids = {entry["id"] for entry in reality_entries}
    assert "reality_walkable_terrain" in reality_ids
    assert "reality_level_completion_signal" in reality_ids
    assert "reality_ui_game_state" in reality_ids
    assert "reality_gravity_field" in reality_ids
    assert "reality_transition_screen_state" in reality_ids
    assert "reality_post_transition_context" in reality_ids
    assert "reality_movement_budget_visual_bar" in reality_ids
    assert "reality_movement_budget_depletion_penalty" in reality_ids
    assert "reality_lives_counter" in reality_ids
    assert "reality_failure_flash_state" in reality_ids
    assert "reality_reference_box_state" in reality_ids
    assert "reality_level_count_progress" in reality_ids
    assert "reality_strategic_reset_affordance" in reality_ids


def test_disk_loader_includes_house_game_mechanics(tmp_path: Path) -> None:
    root = tmp_path
    (root / "galaxies").mkdir(parents=True)
    (root / "house").mkdir(parents=True)
    (root / "galaxies" / "Reality.jsonl").write_text("", encoding="utf-8")
    (root / "galaxies" / "Grammar.jsonl").write_text("", encoding="utf-8")
    (root / "galaxies" / "Tool.jsonl").write_text("", encoding="utf-8")
    populate_game_knowledge(house_dir=root / "house")

    stars = load_all_galaxies_from_disk(root)
    star_ids = {str(star.get("_id") or star.get("id") or "") for star in stars}

    assert "switch_actuator" in star_ids
    assert "multi_step_state_transform" in star_ids
    assert "reality_key_pattern_state" in star_ids
    assert "grammar_unlock_door_on_pattern_match" in star_ids
    assert "meta_probe_transform_blocks_on_mismatch" in star_ids
    assert "input_action_undo" in star_ids
    assert "visual_shape_encodes_lock_class" in star_ids
    assert "screen_transition_uniform_color" in star_ids
    assert "screen_transition_dismiss" in star_ids
    assert "post_transition_new_context" in star_ids
    assert "movement_budget_visual_bar" in star_ids
    assert "lives_system" in star_ids
    assert "strategic_reset" in star_ids
    assert "screen_flash_failure" in star_ids
    assert "reference_box_current_state" in star_ids
    assert "level_count_indicator" in star_ids
    assert "reality_level_topology" in star_ids
    assert "reality_transition_screen_state" in star_ids
    assert "reality_movement_budget_visual_bar" in star_ids
    assert "reality_lives_counter" in star_ids
    assert "reality_failure_flash_state" in star_ids
    assert "grammar_transition_game_state" in star_ids
    assert "grammar_detect_transition_screen" in star_ids
    assert "grammar_decode_movement_budget_bar" in star_ids
    assert "grammar_trigger_strategic_reset" in star_ids
    assert "meta_start_from_title_state" in star_ids
    assert "meta_dismiss_transition_before_navigation" in star_ids
    assert "meta_reset_before_budget_depletion" in star_ids
    assert len(build_game_mechanics_entries()) >= 100
    assert len(build_game_reality_entries()) >= 20
    assert len(build_game_grammar_rules()) >= 20
    assert len(build_game_meta_rules()) >= 15
