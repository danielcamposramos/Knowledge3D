from __future__ import annotations

import pytest

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op


def _make_engine() -> TieredRPNEngine:
    try:
        return TieredRPNEngine()
    except RuntimeError as exc:
        if "invalid device context" in str(exc).lower():
            pytest.skip("CUDA context unavailable for tiered CAS/SAS dispatch tests")
        raise


def test_cas_hash_selects_tier1() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_CAS_HASH]) == 1


def test_semantic_resolve_selects_tier2() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_SEMANTIC_RESOLVE]) == 2


def test_canonicalize_selects_tier3() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_CANONICALIZE]) == 3


def test_contextual_rewrite_selects_tier3() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_CONTEXTUAL_REWRITE]) == 3


def test_mixed_sas_program_uses_highest_tier() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_CAS_HASH, op.OP_CONTEXTUAL_REWRITE]) == 3


def test_existing_cas_block_remains_tier2() -> None:
    engine = _make_engine()
    assert engine.select_tier([op.OP_POLY_BUILD, op.OP_SIMPLIFY]) == 2
