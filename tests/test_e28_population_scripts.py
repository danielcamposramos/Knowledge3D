from __future__ import annotations

import json

from scripts.populate_grammar_rules import (
    build_grammar_rule_entries,
    build_word_anchor_entries,
    populate_grammar_rules,
)
from scripts.populate_math_symbols import (
    build_math_symbol_entries,
    populate_math_symbols,
)
from scripts.populate_reality_systems import (
    build_reality_system_entries,
    populate_reality_systems,
)


def _load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_e28_reality_population_is_idempotent_and_symlinked(tmp_path):
    galaxy_dir = tmp_path / "galaxies"

    first = populate_reality_systems(galaxy_dir=galaxy_dir)
    second = populate_reality_systems(galaxy_dir=galaxy_dir)
    rows = _load_rows(galaxy_dir / "Reality.jsonl")
    ids = {row["id"] for row in rows}
    system = next(row for row in rows if row["id"] == "reality_system_constant_acceleration_1d")

    assert len(build_reality_system_entries()) == 23
    assert first["Reality.jsonl"]["appended"] == 23
    assert second["Reality.jsonl"]["appended"] == 0
    assert second["Reality.jsonl"]["replaced"] == 23
    assert "reality_atom_position_1d" in ids
    assert "reality_system_heat_2d" in ids
    assert system["component_refs"] == [
        "reality_atom_position_1d",
        "reality_atom_velocity_1d",
        "reality_atom_acceleration_1d",
        "reality_atom_timestep",
    ]
    assert "behavior_rpn" in system
    assert "law_rpn" in system
    assert "visual_rpn" in system


def test_e28_grammar_population_writes_foundational_rules_and_word_anchors(tmp_path):
    galaxy_dir = tmp_path / "galaxies"

    first = populate_grammar_rules(galaxy_dir=galaxy_dir)
    second = populate_grammar_rules(galaxy_dir=galaxy_dir)
    grammar_rows = _load_rows(galaxy_dir / "Grammar.jsonl")
    word_rows = _load_rows(galaxy_dir / "Word.jsonl")
    grammar_ids = {row["id"] for row in grammar_rows}
    word_ids = {row["id"] for row in word_rows}
    power_rule = next(row for row in grammar_rows if row["id"] == "grammar_math_power_rule")

    assert len(build_grammar_rule_entries()) == 100
    assert len(build_word_anchor_entries()) > 20
    assert first["Grammar.jsonl"]["appended"] == 100
    assert first["Word.jsonl"]["appended"] == len(build_word_anchor_entries())
    assert second["Grammar.jsonl"]["appended"] == 0
    assert second["Grammar.jsonl"]["replaced"] == 100
    assert second["Word.jsonl"]["appended"] == 0
    assert "grammar_logic_modus_ponens" in grammar_ids
    assert "grammar_unit_celsius_to_fahrenheit" in grammar_ids
    assert "grammar_trig_pythagorean" in grammar_ids
    assert "word_derivative" in word_ids
    assert "word_celsius" in word_ids
    assert power_rule["symbol_refs"] == [
        "char_math_partial",
        "char_op_power",
        "char_op_multiply",
    ]
    assert power_rule["word_refs"] == ["word_derivative", "word_power"]


def test_e28_character_population_writes_35_procedural_entries(tmp_path):
    galaxy_dir = tmp_path / "galaxies"

    first = populate_math_symbols(galaxy_dir=galaxy_dir)
    second = populate_math_symbols(galaxy_dir=galaxy_dir)
    rows = _load_rows(galaxy_dir / "Character.jsonl")
    ids = {row["id"] for row in rows}
    summation = next(row for row in rows if row["id"] == "char_math_summation")
    digit = next(row for row in rows if row["id"] == "char_digit_7")
    op = next(row for row in rows if row["id"] == "char_op_power")

    assert len(build_math_symbol_entries()) == 35
    assert first["Character.jsonl"]["appended"] == 35
    assert second["Character.jsonl"]["appended"] == 0
    assert second["Character.jsonl"]["replaced"] == 35
    assert "char_math_integral" in ids
    assert "char_digit_0" in ids
    assert "char_op_equals" in ids
    assert summation["metadata"]["glyph"] == "∑"
    assert summation["word_refs"] == ["word_sum"]
    assert digit["word_refs"] == ["word_seven"]
    assert op["word_refs"] == ["word_power"]
    assert "visual_rpn" in summation
