from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_manager import normalize_disk_entry
from scripts.normalize_meaning_stars import normalize_meaning_stars_file


def test_normalize_legacy_meaning_layer_entry_to_canonical_schema() -> None:
    legacy = {
        "star_id": "synset_00001740_a",
        "meaning_rpn": "SYNSET A ABLE DEF having the necessary means",
        "meaning_class": "adjective",
        "surface_forms": "{'en': {'word_ref': 'able', 'char_refs': ['a', 'b', 'l', 'e']}}",
        "confidence": "1",
        "polarity": "0",
        "behavior_rpn": "None",
        "visual_rpn": "None",
        "taxonomy_refs": "['wordnet/adjective']",
        "component_refs": "['meaning_component_1']",
        "grammar_refs": "['grammar_rule_1']",
        "reality_refs": "['reality_ref_1']",
        "house_position": "[1, 2, 3]",
        "house_room": "House/Library",
    }

    normalized = normalize_disk_entry("meaning_layer_stars", legacy)

    assert normalized["id"] == "synset_00001740_a"
    assert normalized["galaxy"] == "meaning_layer_stars"
    assert normalized["category"] == "adjective"
    assert normalized["layer"] == 2
    assert normalized["content"] == legacy["meaning_rpn"]
    assert normalized["rpn_program"] == legacy["meaning_rpn"]
    assert normalized["metadata"]["meaning_star_id"] == "synset_00001740_a"
    assert normalized["metadata"]["surface_form_languages"] == ["en"]
    assert normalized["metadata"]["surface_forms"]["en"]["word_ref"] == "able"
    assert normalized["metadata"]["house_position"] == [1.0, 2.0, 3.0]
    assert normalized["metadata"]["confidence"] == 1
    assert normalized["metadata"]["component_refs"] == ["meaning_component_1"]
    assert normalized["metadata"]["grammar_refs"] == ["grammar_rule_1"]
    assert normalized["metadata"]["reality_refs"] == ["reality_ref_1"]
    assert "behavior_rpn" not in normalized["metadata"]


def test_normalize_meaning_stars_file_rewrites_in_place(tmp_path) -> None:
    path = tmp_path / "meaning_layer_stars.jsonl"
    legacy = {
        "star_id": "synset_00001740_a",
        "meaning_rpn": "SYNSET A ABLE DEF having the necessary means",
        "meaning_class": "adjective",
        "surface_forms": "{'en': {'word_ref': 'able', 'char_refs': ['a', 'b', 'l', 'e']}}",
        "confidence": "1",
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    summary = normalize_meaning_stars_file(path)

    assert summary["total"] == 1
    assert summary["written"] == 1
    assert summary["normalized"] == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["id"] == "synset_00001740_a"
    assert rows[0]["galaxy"] == "meaning_layer_stars"
    assert rows[0]["metadata"]["surface_forms"]["en"]["word_ref"] == "able"
