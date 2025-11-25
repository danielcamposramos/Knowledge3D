"""Algebra grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

ALGEBRA_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_quadratic",
        language="math",
        domain="math",
        pattern="equation",
        rpn_program="A RECALL x² B RECALL x C RECALL + + = 0",
        examples=[{"A": "1", "B": "2", "C": "1"}],
        description="Quadratic equation: ax² + bx + c = 0",
    ),
    GrammarRule(
        rule_id="math_linear_equation",
        language="math",
        domain="math",
        pattern="equation",
        rpn_program="A RECALL x B RECALL + = C RECALL",
        examples=[{"A": "2", "B": "3", "C": "7"}],
        description="Linear equation ax + b = c",
    ),
    GrammarRule(
        rule_id="math_inequality",
        language="math",
        domain="math",
        pattern="inequality",
        rpn_program="EXPR1 RECALL ≤ EXPR2 RECALL",
        examples=[{"EXPR1": "x+2", "EXPR2": "5"}],
        description="Inequality expression",
    ),
]

__all__ = ["ALGEBRA_RULES"]
