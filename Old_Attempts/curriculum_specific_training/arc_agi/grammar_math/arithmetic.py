"""Arithmetic grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

ARITHMETIC_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_addition",
        language="math",
        domain="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL + OPERAND2 RECALL = RESULT RECALL",
        examples=[
            {"operand1": "2", "operand2": "3", "result": "5"},
            {"operand1": "7", "operand2": "8", "result": "15"},
        ],
        description="Addition: a + b = c",
    ),
    GrammarRule(
        rule_id="math_subtraction",
        language="math",
        domain="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL - OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "5", "operand2": "3", "result": "2"}],
        description="Subtraction: a - b = c",
    ),
    GrammarRule(
        rule_id="math_multiplication",
        language="math",
        domain="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL × OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "3", "operand2": "4", "result": "12"}],
        description="Multiplication: a × b = c",
    ),
    GrammarRule(
        rule_id="math_division",
        language="math",
        domain="math",
        pattern="binary_op",
        rpn_program="OPERAND1 RECALL ÷ OPERAND2 RECALL = RESULT RECALL",
        examples=[{"operand1": "12", "operand2": "3", "result": "4"}],
        description="Division: a ÷ b = c",
    ),
]

__all__ = ["ARITHMETIC_RULES"]
