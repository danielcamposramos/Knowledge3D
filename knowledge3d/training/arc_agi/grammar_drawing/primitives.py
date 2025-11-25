"""Drawing primitive grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

PRIMITIVE_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="draw_line",
        language="drawing",
        domain="drawing",
        pattern="primitive",
        rpn_program="X1 RECALL Y1 RECALL MOVE X2 RECALL Y2 RECALL LINE STROKE",
        examples=[{"X1": "0", "Y1": "0", "X2": "10", "Y2": "10"}],
        description="Draw line from (x1,y1) to (x2,y2)",
    ),
    GrammarRule(
        rule_id="draw_rectangle",
        language="drawing",
        domain="drawing",
        pattern="primitive",
        rpn_program="X RECALL Y RECALL MOVE W RECALL 0 LINE 0 H RECALL LINE -W RECALL 0 LINE CLOSE FILL",
        examples=[{"X": "0", "Y": "0", "W": "5", "H": "5"}],
        description="Draw filled rectangle",
    ),
    GrammarRule(
        rule_id="draw_circle",
        language="drawing",
        domain="drawing",
        pattern="primitive",
        rpn_program="CX RECALL CY RECALL R RECALL CIRCLE FILL",
        examples=[{"CX": "5", "CY": "5", "R": "3"}],
        description="Draw filled circle",
    ),
]

__all__ = ["PRIMITIVE_RULES"]
