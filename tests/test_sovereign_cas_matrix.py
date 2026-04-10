from __future__ import annotations

from knowledge3d.cranium.bridges.cas_integration_bridge import CASExpression, SovereignRPNCAS


def test_compile_matrix_identity_literal() -> None:
    cas = SovereignRPNCAS()
    try:
        values = cas.evaluate_matrix(CASExpression("identity", [], "evaluate"), (2, 2))
        assert values == [1.0, 0.0, 0.0, 1.0]
    finally:
        cas.cleanup()


def test_compile_matrix_scalar_fill() -> None:
    cas = SovereignRPNCAS()
    try:
        values = cas.evaluate_matrix(CASExpression("2 + 3", [], "evaluate"), (2, 3))
        assert values == [5.0] * 6
    finally:
        cas.cleanup()
