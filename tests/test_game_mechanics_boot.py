from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from scripts.populate_game_mechanics import build_game_mechanics_entries, populate_game_mechanics


def test_populate_game_mechanics_is_idempotent(tmp_path: Path) -> None:
    house_dir = tmp_path / "house"
    first = populate_game_mechanics(house_dir=house_dir)
    second = populate_game_mechanics(house_dir=house_dir)

    assert first["after"] >= 10
    assert second["appended"] == 0
    assert second["replaced"] >= 10

    path = house_dir / "game_mechanics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row["id"] for row in rows}
    assert "switch_actuator" in ids
    assert "movement_recharge_block" in ids
    assert "no_instruction_discovery" in ids


def test_galaxy_manager_reads_house_jsonl_entries(tmp_path: Path) -> None:
    galaxies_dir = tmp_path / "galaxies"
    house_dir = tmp_path / "house"
    galaxies_dir.mkdir(parents=True)
    house_dir.mkdir(parents=True)
    (galaxies_dir / "Reality.jsonl").write_text("", encoding="utf-8")
    populate_game_mechanics(house_dir=house_dir)

    manager = GalaxyManager(storage_root=galaxies_dir, extra_storage_roots=[house_dir])
    paths = manager.iter_storage_jsonl_paths()
    stems = [path.stem for path in paths]

    assert "Reality" in stems
    assert "game_mechanics" in stems

    entries = manager._read_entries_from_disk("game_mechanics")
    entry_ids = {entry["id"] for entry in entries}
    assert "lock_key_pattern_match" in entry_ids
    assert "shape_transform_block" in entry_ids


def test_disk_loader_includes_house_game_mechanics(tmp_path: Path) -> None:
    root = tmp_path
    (root / "galaxies").mkdir(parents=True)
    (root / "house").mkdir(parents=True)
    (root / "galaxies" / "Reality.jsonl").write_text("", encoding="utf-8")
    populate_game_mechanics(house_dir=root / "house")

    stars = load_all_galaxies_from_disk(root)
    star_ids = {str(star.get("_id") or star.get("id") or "") for star in stars}

    assert "switch_actuator" in star_ids
    assert "multi_step_state_transform" in star_ids
    assert len(build_game_mechanics_entries()) >= 10
