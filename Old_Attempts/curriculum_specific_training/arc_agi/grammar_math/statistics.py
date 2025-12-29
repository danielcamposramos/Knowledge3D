"""Statistics and probability grammar rules."""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule

STATISTICS_RULES: List[GrammarRule] = [
    GrammarRule(
        rule_id="math_mean",
        language="math",
        domain="math",
        pattern="statistics",
        rpn_program="μ = Σ DATA RECALL / N RECALL",
        examples=[{"DATA": "x₁..xₙ", "N": "n"}],
        description="Mean: μ = Σx/n",
    ),
    GrammarRule(
        rule_id="math_variance",
        language="math",
        domain="math",
        pattern="statistics",
        rpn_program="σ² = Σ ( DATA RECALL - μ RECALL )² / N RECALL",
        examples=[{"DATA": "x", "μ": "μ", "N": "n"}],
        description="Variance: σ² = Σ(x - μ)²/n",
    ),
    GrammarRule(
        rule_id="math_probability_union",
        language="math",
        domain="math",
        pattern="probability",
        rpn_program="P RECALL (A ∪ B) = P RECALL (A) + P RECALL (B) - P RECALL (A ∩ B)",
        examples=[{"A": "A", "B": "B"}],
        description="Probability of union of events",
    ),
]

__all__ = ["STATISTICS_RULES"]
