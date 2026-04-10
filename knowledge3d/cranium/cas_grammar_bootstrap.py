"""Bootstrap Grammar Galaxy with sovereign CAS transformation rules."""

from __future__ import annotations

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


DIFF_RULES = [
    MeaningCentricStar(
        star_id="cas_rule:diff_power",
        meaning_class="cas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_CONST n OP_POWER OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_CONST n OP_VAR_X OP_CONST n_minus_1 OP_POWER OP_MUL",
        taxonomy_refs=["calculus", "differentiation", "power_rule"],
        grammar_refs=["derivative_operator", "power_rule"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_sin",
        meaning_class="cas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_SIN OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_COS",
        taxonomy_refs=["calculus", "differentiation", "trigonometry"],
        grammar_refs=["derivative_operator", "sine_rule"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_cos",
        meaning_class="cas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_COS OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_SIN OP_NEGATE",
        taxonomy_refs=["calculus", "differentiation", "trigonometry"],
        grammar_refs=["derivative_operator", "cosine_rule"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:diff_exp",
        meaning_class="cas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_EXP OP_SYMBOLIC_DIFF",
        behavior_rpn="OP_VAR_X OP_EXP",
        taxonomy_refs=["calculus", "differentiation", "exponential"],
        grammar_refs=["derivative_operator", "exp_rule"],
        confidence=1,
        polarity=1,
    ),
    MeaningCentricStar(
        star_id="cas_rule:trig_pythagorean",
        meaning_class="cas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn="OP_VAR_X OP_SIN OP_CONST 2 OP_POWER OP_VAR_X OP_COS OP_CONST 2 OP_POWER OP_ADD",
        behavior_rpn="OP_CONST 1.0",
        taxonomy_refs=["trigonometry", "identity", "simplification"],
        grammar_refs=["pythagorean_identity"],
        confidence=1,
        polarity=1,
    ),
]


def build_cas_rule_stars() -> list[MeaningCentricStar]:
    """Return the foundational sovereign CAS transformation rules."""
    return [MeaningCentricStar.from_dict(rule.to_dict()) for rule in DIFF_RULES]


__all__ = ["DIFF_RULES", "build_cas_rule_stars"]
