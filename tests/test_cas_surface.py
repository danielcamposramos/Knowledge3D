from __future__ import annotations

import pytest

from knowledge3d.cranium.bridges.cas_integration_bridge import CASExpression, SovereignRPNCAS


@pytest.fixture
def cas() -> SovereignRPNCAS:
    engine = SovereignRPNCAS()
    try:
        yield engine
    finally:
        engine.cleanup()


def test_cas_basic_arithmetic_on_gpu(cas: SovereignRPNCAS) -> None:
    expr = CASExpression("2 + 3 * 4", [], "evaluate")
    result = cas.evaluate_expression(expr)
    assert result == pytest.approx(14.0, rel=1.0e-5)


def test_cas_constraints_drive_variable_values(cas: SovereignRPNCAS) -> None:
    expr = CASExpression("x**2 + 2*x + 1", ["x"], "evaluate", constraints=["x=2.5"])
    result = cas.evaluate_expression(expr)
    assert result == pytest.approx(12.25, rel=1.0e-5)


def test_cas_ternary_subset_compiles_to_live_surface(cas: SovereignRPNCAS) -> None:
    expr = CASExpression("1 and (0 or not 0)", [], "ternary_logic")
    result = cas.evaluate_expression(expr)
    assert result == pytest.approx(1.0, rel=1.0e-5)


def test_cas_matrix_literal_path_is_real_code_not_notimplemented(cas: SovereignRPNCAS) -> None:
    expr = CASExpression("[[1, 2], [3, 4]]", [], "matrix")
    result = cas.evaluate_matrix(expr, (2, 2))
    assert result == [1.0, 2.0, 3.0, 4.0]


def test_cas_stats_are_truthful(cas: SovereignRPNCAS) -> None:
    cas.evaluate_expression(CASExpression("2 + 2", [], "evaluate"))
    stats = cas.get_performance_stats()
    assert stats["cas_executions"] >= 1
    assert stats["gpu_execution_mode"] == "gpu_first_rpn_ptx"
    assert stats["sovereign_gpu_execution"] is True
