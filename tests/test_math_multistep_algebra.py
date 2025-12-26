def _skip_if_no_gpu() -> None:
    try:
        from knowledge3d.cranium.sovereign import loader as sovereign_loader

        ptr = sovereign_loader.gpu_malloc(4)
        sovereign_loader.gpu_free(ptr)
    except Exception:
        import pytest

        pytest.skip("GPU not available for sovereign RPN execution")


def test_store_recall_opcodes_string_engine() -> None:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

    _skip_if_no_gpu()
    engine = ModularRPNEngine()

    result = engine.evaluate("5 STORE_A 3 STORE_B RECALL_A RECALL_B *")
    assert result == 15.0

    result = engine.evaluate("5 2 pow")
    assert result == 25.0

    result = engine.evaluate("1 STORE_A 5 STORE_B 6 STORE_C RECALL_B DUP * RECALL_A RECALL_C * 4 * -")
    assert result == 1.0


def test_quadratic_rule_solves_one_root() -> None:
    from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import ALGEBRA_RULES
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    _skip_if_no_gpu()
    engine = ModularRPNEngine()
    nav = TRMMathNavigator(rule_bank=ALGEBRA_RULES, math_galaxy=MATH_GALAXY, rpn_engine=engine)

    result, meta = nav.solve("Solve x^2 - 5x + 6 = 0")
    assert meta["rule_used"] == "quadratic_standard_form"
    assert float(result) in (2.0, 3.0)
