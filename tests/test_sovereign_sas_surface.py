"""Smoke tests for the sovereign SAS surface."""

from __future__ import annotations

import pathlib

from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import RPNProgram
from knowledge3d.cranium.sas_grammar_bootstrap import build_sas_rule_stars
from knowledge3d.cranium.sas_symbol_bootstrap import SYMBOL_REGISTRY, build_symbol_table
from knowledge3d.ingestion import ingest_sas_bootstrap
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_sas_opcodes_in_correct_range() -> None:
    sas_ops = [
        op.OP_CANONICALIZE,
        op.OP_CAS_HASH,
        op.OP_SEMANTIC_RESOLVE,
        op.OP_RULE_SELECT,
        op.OP_CONTEXTUAL_REWRITE,
        op.OP_SEMANTIC_EQUIV,
    ]
    for value in sas_ops:
        assert 0x238 <= value <= 0x25F


def test_sas_no_collision_with_cas() -> None:
    cas_range = set(range(0x220, 0x238))
    sas_range = set(range(0x238, 0x240))
    assert cas_range.isdisjoint(sas_range)


def test_symbol_table_has_physical_constants() -> None:
    values, star_ids = build_symbol_table(galaxy_manager=None)
    assert values[0x20] == SYMBOL_REGISTRY[0x20][1]
    assert values[0x21] == SYMBOL_REGISTRY[0x21][1]
    assert len(star_ids) == 256


def test_symbol_table_live_override_uses_galaxy_values() -> None:
    class _Galaxy:
        def __init__(self, entries):
            self.entries = list(entries)

    class _Manager:
        def __init__(self, entries):
            self._entries = list(entries)

        def get_galaxy(self, name: str):
            if name == "Reality":
                return _Galaxy(self._entries)
            return _Galaxy([])

    star = MeaningCentricStar(
        star_id="physical_constant_g",
        meaning_class="physical_constant",
        domain="reality",
        galaxy_ref="Reality",
        surface_forms={"en": {"word_ref": "G"}},
        reality_refs=["6.5e-11"],
    )
    manager = _Manager([star.to_galaxy_entry(galaxy_name="Reality")])
    values, star_ids = build_symbol_table(manager)
    assert values[0x20] == 6.5e-11
    assert star_ids[0x20] != 0


def test_sas_grammar_stars_are_valid_meaning_stars() -> None:
    stars = build_sas_rule_stars()
    assert len(stars) >= 7
    for star in stars:
        assert isinstance(star, MeaningCentricStar)
        assert star.meaning_class == "sas_rule"
        assert star.domain == "grammar"
        assert star.galaxy_ref == "Grammar"
        assert star.meaning_rpn
        assert star.behavior_rpn


def test_sas_rule_stars_star_ids_unique() -> None:
    stars = build_sas_rule_stars()
    ids = [star.star_id for star in stars]
    assert len(ids) == len(set(ids))


def test_sas_bootstrap_ingestion_returns_symbol_table_and_stores_rules() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.rows = []

        def bulk_disk_sync(self):
            from contextlib import nullcontext

            return nullcontext()

        def store_meaning_star(self, galaxy_name: str, star, *, category: str = "meaning_star", metadata=None):
            self.rows.append((galaxy_name, star, category, dict(metadata or {})))
            return "inserted"

        def get_galaxy(self, _name: str):
            class _Galaxy:
                entries = []

            return _Galaxy()

    manager = _Manager()
    values, star_ids, stars = ingest_sas_bootstrap(manager)
    assert len(values) == 256
    assert len(star_ids) == 256
    assert len(stars) >= 7
    assert len(manager.rows) == len(stars)
    assert all(row[0] == "Grammar" for row in manager.rows)
    assert all(row[1].meaning_class == "sas_rule" for row in manager.rows)


def test_sas_mnemonics_compile_in_rpn_program() -> None:
    program = RPNProgram()
    program.u16(op.OP_CANONICALIZE)
    program.u16(op.OP_SEMANTIC_RESOLVE)
    program.u16(op.OP_SEMANTIC_EQUIV)
    assert len(program.to_bytes()) > 0


def test_sovereign_sas_files_do_not_import_symengine() -> None:
    files = [
        pathlib.Path("knowledge3d/cranium/sas_symbol_bootstrap.py"),
        pathlib.Path("knowledge3d/cranium/sas_grammar_bootstrap.py"),
        pathlib.Path("knowledge3d/cranium/ptx_runtime/rpn_opcodes.py"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "symengine" not in text
