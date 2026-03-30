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
    populate_game_mechanics,
)


def test_populate_game_mechanics_is_idempotent(tmp_path: Path) -> None:
    house_dir = tmp_path / "house"
    first = populate_game_knowledge(house_dir=house_dir)
    second = populate_game_knowledge(house_dir=house_dir)

    assert first["game_mechanics.jsonl"]["after"] >= 10
    assert first["Reality.jsonl"]["after"] >= 8
    assert first["Grammar.jsonl"]["after"] >= 8
    assert first["Tool.jsonl"]["after"] >= 6

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
    assert rows[0]["meaning_rpn"]
    assert rows[0]["behavior_rpn"]
    assert rows[0]["meta_refs"]

    grammar_rows = [
        json.loads(line)
        for line in (house_dir / "Grammar.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grammar_ids = {row["id"] for row in grammar_rows}
    assert "grammar_unlock_door_on_pattern_match" in grammar_ids
    assert "grammar_infer_rule_from_state_delta" in grammar_ids

    tool_rows = [
        json.loads(line)
        for line in (house_dir / "Tool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_ids = {row["id"] for row in tool_rows}
    assert "meta_route_to_switch_when_locked" in tool_ids
    assert "meta_click_to_advance_after_completion" in tool_ids


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

    grammar_entries = manager._read_entries_from_disk("Grammar")
    grammar_ids = {entry["id"] for entry in grammar_entries}
    assert "grammar_route_agent_on_walkable_grid" in grammar_ids
    assert "grammar_complete_level_after_goal_cross" in grammar_ids

    tool_entries = manager._read_entries_from_disk("Tool")
    tool_ids = {entry["id"] for entry in tool_entries}
    assert "meta_match_key_before_door" in tool_ids
    assert "meta_learn_from_visual_transition" in tool_ids

    reality_entries = manager._read_entries_from_disk("Reality")
    reality_ids = {entry["id"] for entry in reality_entries}
    assert "reality_walkable_terrain" in reality_ids
    assert "reality_level_completion_signal" in reality_ids


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
    assert len(build_game_mechanics_entries()) >= 10
    assert len(build_game_reality_entries()) >= 8
    assert len(build_game_grammar_rules()) >= 8
    assert len(build_game_meta_rules()) >= 6
