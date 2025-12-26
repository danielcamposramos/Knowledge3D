from __future__ import annotations


class _EchoEngine:
    def evaluate(self, expression: str, *_args, **_kwargs):
        return expression


def test_math_symbol_galaxy_populated():
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

    assert len(MATH_GALAXY) >= 50
    assert MATH_GALAXY.lookup("\\frac") is not None


def test_math_grammar_rules_populated():
    from knowledge3d.training.arc_agi import math_grammar_rules

    rule_lists = [name for name in dir(math_grammar_rules) if name.endswith("_RULES")]
    total = sum(len(getattr(math_grammar_rules, name)) for name in rule_lists)
    assert total >= 100


def test_trm_math_navigator_routes_and_composes():
    from knowledge3d.training.arc_agi import math_grammar_rules
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    nav = TRMMathNavigator(
        rule_bank=math_grammar_rules.SOVEREIGN_MATH_RULES,
        math_galaxy=MATH_GALAXY,
        rpn_engine=_EchoEngine(),
    )

    result, meta = nav.solve("\\frac{24}{4}")
    assert result == "24 4 /"
    assert meta["rule_used"] == "latex_frac"

    result2, meta2 = nav.solve("gcd(12, 8)")
    assert result2 == "12 8 gcd"
    assert meta2["rule_used"] == "latex_gcd_parens"


def test_math_knowledge_loader_can_populate_math_galaxy_without_crashing():
    from knowledge3d.training.math_benchmarks.math_knowledge_loader import MathKnowledgeLoader

    loader = MathKnowledgeLoader()
    stats = loader.load_all()
    assert "formulas" in stats and "rules" in stats and "rpn_patterns" in stats
    added = loader.populate_math_galaxy(max_symbols=10)
    assert isinstance(added, int)

