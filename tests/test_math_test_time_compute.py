from __future__ import annotations


class _EchoEngine:
    def evaluate(self, expression: str, *_args, **_kwargs):
        stack: list[float] = []
        for tok in (expression or "").split():
            if tok in {"+", "-", "*", "/"}:
                if len(stack) < 2:
                    raise ValueError("stack underflow")
                b = float(stack.pop())
                a = float(stack.pop())
                if tok == "+":
                    stack.append(a + b)
                elif tok == "-":
                    stack.append(a - b)
                elif tok == "*":
                    stack.append(a * b)
                else:
                    stack.append(a / b)
                continue
            stack.append(float(tok))
        if not stack:
            return None
        return float(stack[-1])


def test_generic_equation_galaxy_basics():
    from knowledge3d.cranium.generic_equations import GENERIC_EQUATION_GALAXY

    assert len(GENERIC_EQUATION_GALAXY) >= 7
    eq = GENERIC_EQUATION_GALAXY.get("fair_share")
    assert eq is not None
    assert "total / count" in eq.formula or "/" in eq.rpn


def test_test_time_compute_solves_rate_time_without_rules(tmp_path):
    from knowledge3d.cranium.generic_equations import GENERIC_EQUATION_GALAXY
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        generic_equations_galaxy=GENERIC_EQUATION_GALAXY,
        rule_bank=[],
        shadow_copy=None,
        thinking_budget=4,
    )

    # No word_sequence rules: must rely on test-time compute candidates from numbers.
    text = "A car travels at 60 miles per hour for 2 hours. How far does it travel?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=1)
    assert result == 120.0
    assert meta.get("template_used") in {"test_time_compute", "rate_duration", "extract_operate_aggregate"}


def test_test_time_compute_solves_simple_conversion(tmp_path):
    from knowledge3d.cranium.generic_equations import GENERIC_EQUATION_GALAXY
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        generic_equations_galaxy=GENERIC_EQUATION_GALAXY,
        rule_bank=[],
        shadow_copy=None,
        thinking_budget=4,
    )

    text = "She has 300 cents. How many dollars is that?"
    result, meta = reader.solve(problem_text=text, rpn_engine=_EchoEngine(), max_attempts=1)
    assert result == 3.0
    assert meta.get("rpn_program")

