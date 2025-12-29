"""Calculus grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

CALCULUS_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_derivative",
        language="math",
        domain="math",
        pattern="calculus",
        rpn_program="d/d VAR RECALL ( FUNC RECALL ) = RESULT RECALL",
        examples=[
            {"VAR": "x", "FUNC": "x²", "RESULT": "2x"},
            {"VAR": "x", "FUNC": "sin(x)", "RESULT": "cos(x)"},
        ],
        description="Derivative: d/dx f(x)",
    ),
    GrammarRule(
        rule_id="math_integral",
        language="math",
        domain="math",
        pattern="calculus",
        rpn_program="∫ FUNC RECALL d VAR RECALL = RESULT RECALL + C",
        examples=[
            {"FUNC": "x", "VAR": "x", "RESULT": "x²/2"},
            {"FUNC": "cos(x)", "VAR": "x", "RESULT": "sin(x)"},
        ],
        description="Integral: ∫ f(x) dx",
    ),
    GrammarRule(
        rule_id="math_limit",
        language="math",
        domain="math",
        pattern="calculus",
        rpn_program="lim VAR RECALL → LIMIT_VAL RECALL FUNC RECALL = RESULT RECALL",
        examples=[{"VAR": "x", "LIMIT_VAL": "0", "FUNC": "sin(x)/x", "RESULT": "1"}],
        description="Limit: lim x→a f(x)",
    ),
]

__all__ = ["CALCULUS_RULES"]
