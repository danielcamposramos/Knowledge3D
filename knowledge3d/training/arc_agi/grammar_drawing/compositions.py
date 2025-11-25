"""Drawing composition grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

COMPOSITION_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="draw_rotated_square",
        language="drawing",
        domain="drawing",
        pattern="composition",
        rpn_program="45 ROTATE X RECALL Y RECALL MOVE W RECALL 0 LINE 0 H RECALL LINE -W RECALL 0 LINE CLOSE FILL",
        examples=[{"X": "5", "Y": "5", "W": "3", "H": "3"}],
        description="Draw square rotated 45 degrees",
    ),
    GrammarRule(
        rule_id="draw_pattern_repeat",
        language="drawing",
        domain="drawing",
        pattern="composition",
        rpn_program="SHAPE RECALL DX RECALL DY RECALL N RECALL REPEAT_PATTERN",
        examples=[{"SHAPE": "circle", "DX": "1", "DY": "0", "N": "5"}],
        description="Repeat shape n times with offset (dx, dy)",
    ),
]

__all__ = ["COMPOSITION_RULES"]
