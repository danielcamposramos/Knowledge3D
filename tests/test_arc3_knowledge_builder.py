from __future__ import annotations

import json

from knowledge3d.knowledgeverse.arc3_knowledge_builder import (
    ARC3_GRAMMAR_RULES,
    ARC3_META_RULES,
    ARC3_REALITY_ENTRIES,
    build_arc3_galaxy_knowledge,
)


def test_arc3_knowledge_builder_is_additive_and_idempotent(tmp_path):
    galaxy_dir = tmp_path / "galaxies"

    first = build_arc3_galaxy_knowledge(galaxy_dir=galaxy_dir)
    second = build_arc3_galaxy_knowledge(galaxy_dir=galaxy_dir)

    assert first["Grammar.jsonl"]["appended"] == len(ARC3_GRAMMAR_RULES)
    assert first["Reality.jsonl"]["appended"] == len(ARC3_REALITY_ENTRIES)
    assert first["Tool.jsonl"]["appended"] == len(ARC3_META_RULES)
    assert first["Word.jsonl"]["appended"] > 0
    assert second["Grammar.jsonl"]["appended"] == 0
    assert second["Reality.jsonl"]["appended"] == 0
    assert second["Tool.jsonl"]["appended"] == 0
    assert second["Word.jsonl"]["appended"] == 0


def test_arc3_knowledge_builder_rewrites_existing_ids_in_place(tmp_path):
    galaxy_dir = tmp_path / "galaxies"
    galaxy_dir.mkdir(parents=True, exist_ok=True)
    grammar_path = galaxy_dir / "Grammar.jsonl"
    grammar_path.write_text(
        json.dumps({"id": "arc3_nav_move_up", "embedding": [1.0] * 32}, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    result = build_arc3_galaxy_knowledge(galaxy_dir=galaxy_dir)
    rows = [
        json.loads(line)
        for line in grammar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if row.get("id") == "grammar_spatial_move_toward_above"]
    legacy = [row for row in rows if row.get("id") == "arc3_nav_move_up"]

    assert result["Grammar.jsonl"]["removed"] >= 1
    assert len(matches) == 1
    assert not legacy
    assert "embedding" not in matches[0]
    assert matches[0]["metadata"]["query_anchor"] == "object above center top north navigate game frame move up direction spatial"


def test_arc3_knowledge_builder_writes_ids_without_stored_embeddings(tmp_path):
    galaxy_dir = tmp_path / "galaxies"
    build_arc3_galaxy_knowledge(galaxy_dir=galaxy_dir)

    grammar_rows = [
        json.loads(line)
        for line in (galaxy_dir / "Grammar.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reality_rows = [
        json.loads(line)
        for line in (galaxy_dir / "Reality.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tool_rows = [
        json.loads(line)
        for line in (galaxy_dir / "Tool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    word_rows = [
        json.loads(line)
        for line in (galaxy_dir / "Word.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = grammar_rows + reality_rows + tool_rows + word_rows
    ids = {row["id"] for row in rows}

    assert "grammar_spatial_move_toward_below" in ids
    assert "reality_arc3_available_actions" in ids
    assert "meta_spatial_respect_available_actions" in ids
    assert "reality_action_move_up" in ids
    assert "word_direction_above" in ids
    assert "arc3_nav_move_down" not in ids
    assert "word_arc3_navigate_above" not in ids
    assert all("embedding" not in row for row in rows)
    query_anchors = {
        row["id"]: row.get("metadata", {}).get("query_anchor")
        for row in rows
        if row["id"].startswith("grammar_") or row["id"].startswith("word_") or row["id"].startswith("reality_action_")
    }
    assert query_anchors["grammar_spatial_move_toward_below"] == "object below center bottom south navigate game frame move down direction spatial"
    assert query_anchors["grammar_spatial_keyboard_game"] == "keyboard navigation move directional spatial action availability"
    assert query_anchors["word_direction_above"] == "above upward north up direction spatial movement"
    assert query_anchors["reality_action_move_up"] == "move up upward above north spatial translation direction"
