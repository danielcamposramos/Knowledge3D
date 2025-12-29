"""Linear algebra grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

LINEAR_ALGEBRA_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_matrix_mult",
        language="math",
        domain="math",
        pattern="matrix_op",
        rpn_program="MATRIX1 RECALL × MATRIX2 RECALL = RESULT_MATRIX RECALL",
        examples=[{"MATRIX1": "A", "MATRIX2": "B", "RESULT_MATRIX": "AB"}],
        description="Matrix multiplication",
    ),
    GrammarRule(
        rule_id="math_determinant",
        language="math",
        domain="math",
        pattern="matrix_property",
        rpn_program="det( MATRIX RECALL ) = RESULT RECALL",
        examples=[{"MATRIX": "A", "RESULT": "det(A)"}],
        description="Matrix determinant",
    ),
    GrammarRule(
        rule_id="math_eigenvalue",
        language="math",
        domain="math",
        pattern="eigenproblem",
        rpn_program="MATRIX RECALL VECTOR RECALL = LAMBDA RECALL VECTOR RECALL",
        examples=[{"MATRIX": "A", "VECTOR": "v", "LAMBDA": "λ"}],
        description="Eigenvalue equation Av = λv",
    ),
]

__all__ = ["LINEAR_ALGEBRA_RULES"]
