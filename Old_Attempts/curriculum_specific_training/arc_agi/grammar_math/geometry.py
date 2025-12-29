"""Geometry grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

GEOMETRY_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_area_circle",
        language="math",
        domain="math",
        pattern="formula",
        rpn_program="A = π R RECALL²",
        examples=[{"R": "5"}],
        description="Circle area: A = πr²",
    ),
    GrammarRule(
        rule_id="math_pythagorean",
        language="math",
        domain="math",
        pattern="theorem",
        rpn_program="A RECALL² + B RECALL² = C RECALL²",
        examples=[{"A": "3", "B": "4", "C": "5"}],
        description="Pythagorean theorem",
    ),
]

__all__ = ["GEOMETRY_RULES"]
