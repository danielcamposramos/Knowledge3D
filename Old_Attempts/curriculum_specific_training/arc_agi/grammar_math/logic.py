"""Logic and set theory grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

LOGIC_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_set_union",
        language="math",
        domain="math",
        pattern="set_op",
        rpn_program="SET1 RECALL ∪ SET2 RECALL = RESULT_SET RECALL",
        examples=[{"SET1": "A", "SET2": "B", "RESULT_SET": "A ∪ B"}],
        description="Set union",
    ),
    GrammarRule(
        rule_id="math_set_intersection",
        language="math",
        domain="math",
        pattern="set_op",
        rpn_program="SET1 RECALL ∩ SET2 RECALL = RESULT_SET RECALL",
        examples=[{"SET1": "A", "SET2": "B", "RESULT_SET": "A ∩ B"}],
        description="Set intersection",
    ),
    GrammarRule(
        rule_id="math_implication",
        language="math",
        domain="math",
        pattern="logic",
        rpn_program="PREMISE RECALL → CONCLUSION RECALL",
        examples=[{"PREMISE": "P", "CONCLUSION": "Q"}],
        description="Logical implication",
    ),
]

__all__ = ["LOGIC_RULES"]
