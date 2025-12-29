"""Drawing transform grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

TRANSFORM_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="draw_rotate",
        language="drawing",
        domain="drawing",
        pattern="transform",
        rpn_program="ANGLE RECALL ROTATE",
        examples=[{"ANGLE": "45"}],
        description="Rotate drawing by angle",
    ),
    GrammarRule(
        rule_id="draw_translate",
        language="drawing",
        domain="drawing",
        pattern="transform",
        rpn_program="DX RECALL DY RECALL TRANSLATE",
        examples=[{"DX": "1", "DY": "1"}],
        description="Translate drawing",
    ),
    GrammarRule(
        rule_id="draw_scale",
        language="drawing",
        domain="drawing",
        pattern="transform",
        rpn_program="SX RECALL SY RECALL SCALE",
        examples=[{"SX": "2.0", "SY": "1.5"}],
        description="Scale drawing",
    ),
]

__all__ = ["TRANSFORM_RULES"]
