"""Smoke tests for the sovereign CAS surface."""

from __future__ import annotations

import pathlib

import pytest

from knowledge3d.cranium.cas_grammar_bootstrap import DIFF_RULES
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.ingestion import ingest_cas_grammar
from knowledge3d.ingestion.cas_ingestion import expression_to_rpn
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_cas_opcodes_in_dedicated_range() -> None:
    cas_ops = [
        op.OP_POLY_COEFF,
        op.OP_POLY_BUILD,
        op.OP_POLY_ADD,
        op.OP_POLY_MUL,
        op.OP_SIMPLIFY,
        op.OP_SUBSTITUTE,
        op.OP_SOLVE_LINEAR,
        op.OP_RULE_APPLY,
    ]
    for value in cas_ops:
        assert 0x220 <= value <= 0x25F, f"CAS op {hex(value)} outside dedicated range 0x220-0x25F"


def test_cas_opcode_no_overlap_with_drawing() -> None:
    drawing = set(range(0x200, 0x220))
    cas = set(range(0x220, 0x260))
    assert not drawing & cas


def test_grammar_rules_are_meaning_centric_stars() -> None:
    for rule in DIFF_RULES:
        assert isinstance(rule, MeaningCentricStar)
        assert rule.meaning_class == "cas_rule"
        assert rule.domain == "grammar"
        assert rule.meaning_rpn is not None
        assert rule.behavior_rpn is not None


def test_star_node_header_exists() -> None:
    header = pathlib.Path("knowledge3d/cranium/kernels/cas_star_node.h")
    assert header.exists(), f"Missing: {header}"


def test_cas_grammar_bootstrap_ingestion() -> None:
    class _Manager:
        def __init__(self) -> None:
            self.rows = []

        def bulk_disk_sync(self):
            from contextlib import nullcontext

            return nullcontext()

        def store_meaning_star(self, galaxy_name: str, star, *, category: str = "meaning_star", metadata=None):
            self.rows.append((galaxy_name, star, category, dict(metadata or {})))
            return "inserted"

    manager = _Manager()
    count = ingest_cas_grammar(manager)
    assert count == len(DIFF_RULES)
    assert all(row[0] == "Grammar" for row in manager.rows)
    assert all(row[1].meaning_class == "cas_rule" for row in manager.rows)


def test_cas_tokens_registered_in_modular_engine() -> None:
    assert ModularRPNEngine.OPCODES["poly_build"] == op.OP_POLY_BUILD
    assert ModularRPNEngine.OPCODES["simplify"] == op.OP_SIMPLIFY
    assert ModularRPNEngine.OPCODES["cas_eval"] == op.OP_CAS_EVAL


def test_cas_ingestion_uses_symengine_only_when_available() -> None:
    try:
        import symengine  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            expression_to_rpn("x**2 + sin(x)")
        return

    rpn = expression_to_rpn("x**2 + sin(x)")
    tokens = rpn.split()
    assert tokens[-1] == "OP_ADD"
    assert "OP_POWER" in tokens
    assert "OP_SIN" in tokens
    assert tokens.count("OP_VAR_X") >= 2


def test_cas_ingestion_not_imported_in_hot_path() -> None:
    hot_path_dirs = [
        pathlib.Path("knowledge3d/cranium/ptx_runtime"),
        pathlib.Path("knowledge3d/cranium/kernels"),
        pathlib.Path("knowledge3d/cranium/bridges"),
        pathlib.Path("knowledge3d/knowledgeverse"),
    ]
    for root in hot_path_dirs:
        for py_path in root.rglob("*.py"):
            text = py_path.read_text(encoding="utf-8", errors="ignore")
            if "symengine" in text and py_path.name != "cas_ingestion.py":
                raise AssertionError(f"symengine imported in hot path: {py_path}")
