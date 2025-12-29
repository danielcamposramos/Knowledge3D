"""Bezier and curve grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

CURVE_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="draw_quadratic_bezier",
        language="drawing",
        domain="drawing",
        pattern="curve",
        rpn_program="X1 RECALL Y1 RECALL MOVE CX RECALL CY RECALL X2 RECALL Y2 RECALL QUAD STROKE",
        examples=[{"X1": "0", "Y1": "0", "CX": "5", "CY": "10", "X2": "10", "Y2": "0"}],
        description="Quadratic Bezier curve",
    ),
    GrammarRule(
        rule_id="draw_cubic_bezier",
        language="drawing",
        domain="drawing",
        pattern="curve",
        rpn_program="X1 RECALL Y1 RECALL MOVE CX1 RECALL CY1 RECALL CX2 RECALL CY2 RECALL X2 RECALL Y2 RECALL CUBIC STROKE",
        examples=[{"X1": "0", "Y1": "0", "CX1": "3", "CY1": "10", "CX2": "7", "CY2": "10", "X2": "10", "Y2": "0"}],
        description="Cubic Bezier curve",
    ),
]

__all__ = ["CURVE_RULES"]
