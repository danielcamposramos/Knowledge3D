from __future__ import annotations

import pytest

from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as op
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.sas_symbol_bootstrap import build_symbol_table


def _make_engines() -> tuple[ModularRPNEngine, TieredRPNEngine]:
    try:
        modular = ModularRPNEngine()
    except RuntimeError as exc:
        if "invalid device context" in str(exc).lower():
            pytest.skip("CUDA context unavailable for swarm CAS integration tests")
        raise
    return modular, modular._sovereign_engine


def test_game2d_worker_program_compiles_to_cas_sas_tier3() -> None:
    modular, tiered = _make_engines()
    expression = "1 canonicalize 2 3 rule_select 4 contextual_rewrite"
    tokens = modular.tokenize_rpn(expression)
    op_codes, _scalars, _vectors = modular.compile_tokens(tokens, 0)
    assert op.OP_CANONICALIZE in op_codes
    assert op.OP_RULE_SELECT in op_codes
    assert op.OP_CONTEXTUAL_REWRITE in op_codes
    assert tiered.select_tier(op_codes) == 3


def test_math_worker_program_compiles_to_semantic_cas_surface() -> None:
    modular, tiered = _make_engines()
    expression = "1 canonicalize 32 semantic_resolve 2 semantic_equiv"
    tokens = modular.tokenize_rpn(expression)
    op_codes, _scalars, _vectors = modular.compile_tokens(tokens, 0)
    assert op.OP_CANONICALIZE in op_codes
    assert op.OP_SEMANTIC_RESOLVE in op_codes
    assert op.OP_SEMANTIC_EQUIV in op_codes
    assert tiered.select_tier(op_codes) == 3


def test_lightweight_semantic_lookup_program_stays_tier2_without_rewrite() -> None:
    modular, tiered = _make_engines()
    expression = "32 semantic_resolve 33 semantic_resolve semantic_equiv"
    tokens = modular.tokenize_rpn(expression)
    op_codes, _scalars, _vectors = modular.compile_tokens(tokens, 0)
    assert op.OP_SEMANTIC_RESOLVE in op_codes
    assert op.OP_SEMANTIC_EQUIV in op_codes
    assert tiered.select_tier(op_codes) == 2


def test_rule_select_and_contextual_rewrite_execute_inside_modular_kernel() -> None:
    modular, _tiered = _make_engines()
    modular._sovereign_engine.bind_cas_pool()
    expression = (
        "101 cas_push_sym "
        "0 cas_push_const "
        "10 2 cas_build "
        "101 cas_push_sym "
        "1 cas_push_sym "
        "0 cas_push_const "
        "10 2 cas_build "
        "canonicalize "
        "2 1 rule_select "
        "contextual_rewrite"
    )
    result, stack = modular.evaluate_with_stack(expression)
    assert result == pytest.approx(4.0)
    assert stack[-1] == pytest.approx(4.0)


def test_canonicalize_and_semantic_equiv_execute_on_gpu() -> None:
    modular, _tiered = _make_engines()
    modular._sovereign_engine.bind_cas_pool()
    expression = (
        "1 cas_push_sym "
        "1 cas_push_const "
        "10 2 cas_build "
        "1 cas_push_const "
        "1 cas_push_sym "
        "10 2 cas_build "
        "canonicalize "
        "2 canonicalize "
        "semantic_equiv"
    )
    result = modular.evaluate(expression)
    assert result == pytest.approx(1.0)


def test_semantic_resolve_reads_bound_symbol_table_inside_kernel() -> None:
    modular, _tiered = _make_engines()
    values, star_ids = build_symbol_table(galaxy_manager=None)
    modular._sovereign_engine.bind_sas_symbol_table(values, star_ids)
    result = modular.evaluate("32 semantic_resolve")
    assert result == pytest.approx(values[32])
