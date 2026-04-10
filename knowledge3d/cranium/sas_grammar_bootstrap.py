"""Bootstrap Grammar Galaxy with SAS canonicalization and equivalence rules."""

from __future__ import annotations

from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


DEFEASIBLE_METADATA = {
    "rule_strength": 1,
    "trust_weight": 1.0,
    "superior_to": [],
}

_SAS_META_REFS = [
    "rule_strength:1",
    "trust_weight:1.0",
]


def _sas_rule(
    star_id: str,
    meaning_rpn: str,
    behavior_rpn: str,
    taxonomy_refs: list[str],
    *,
    grammar_refs: list[str] | None = None,
) -> MeaningCentricStar:
    return MeaningCentricStar(
        star_id=star_id,
        meaning_class="sas_rule",
        domain="grammar",
        galaxy_ref="Grammar",
        meaning_rpn=meaning_rpn,
        behavior_rpn=behavior_rpn,
        taxonomy_refs=list(taxonomy_refs),
        grammar_refs=list(grammar_refs or ["canonical_form", "identity_removal"]),
        meta_refs=_SAS_META_REFS,
        confidence=1,
        polarity=1,
    )


CANONICALIZATION_RULES = [
    _sas_rule(
        "sas_rule:commute_add",
        "OP_VAR_X OP_VAR_Y OP_ADD",
        "OP_VAR_Y OP_VAR_X OP_ADD",
        ["algebra", "commutativity", "addition"],
        grammar_refs=["canonical_form", "commutative_sort"],
    ),
    _sas_rule(
        "sas_rule:commute_mul",
        "OP_VAR_X OP_VAR_Y OP_MUL",
        "OP_VAR_Y OP_VAR_X OP_MUL",
        ["algebra", "commutativity", "multiplication"],
        grammar_refs=["canonical_form", "commutative_sort"],
    ),
    _sas_rule(
        "sas_rule:identity_add_zero",
        "OP_VAR_X OP_CONST 0.0 OP_ADD",
        "OP_VAR_X",
        ["algebra", "identity", "addition"],
    ),
    _sas_rule(
        "sas_rule:identity_mul_one",
        "OP_VAR_X OP_CONST 1.0 OP_MUL",
        "OP_VAR_X",
        ["algebra", "identity", "multiplication"],
    ),
    _sas_rule(
        "sas_rule:annihilate_mul_zero",
        "OP_VAR_X OP_CONST 0.0 OP_MUL",
        "OP_CONST 0.0",
        ["algebra", "annihilator", "multiplication"],
    ),
    _sas_rule(
        "sas_rule:power_zero",
        "OP_VAR_X OP_CONST 0.0 OP_POWER",
        "OP_CONST 1.0",
        ["algebra", "exponentiation", "identity"],
    ),
    _sas_rule(
        "sas_rule:power_one",
        "OP_VAR_X OP_CONST 1.0 OP_POWER",
        "OP_VAR_X",
        ["algebra", "exponentiation", "identity"],
    ),
    _sas_rule(
        "sas_rule:division_inverse",
        "OP_VAR_X OP_VAR_Y OP_DIV",
        "OP_VAR_X OP_CONST 1.0 OP_VAR_Y OP_DIV OP_MUL",
        ["algebra", "division", "inverse"],
        grammar_refs=["canonical_form", "operator_rewrite"],
    ),
    _sas_rule(
        "sas_rule:subtraction_as_addition",
        "OP_VAR_X OP_VAR_Y OP_SUB",
        "OP_VAR_X OP_CONST -1.0 OP_VAR_Y OP_MUL OP_ADD",
        ["algebra", "subtraction", "addition"],
        grammar_refs=["canonical_form", "operator_rewrite"],
    ),
    _sas_rule(
        "sas_rule:distributive_mul",
        "OP_VAR_X OP_VAR_Y OP_VAR_Z OP_ADD OP_MUL",
        "OP_VAR_X OP_VAR_Y OP_MUL OP_VAR_X OP_VAR_Z OP_MUL OP_ADD",
        ["algebra", "distribution", "multiplication"],
        grammar_refs=["canonical_form", "expansion"],
    ),
    _sas_rule(
        "sas_rule:distributive_div",
        "OP_VAR_X OP_VAR_Y OP_ADD OP_VAR_Z OP_DIV",
        "OP_VAR_X OP_VAR_Z OP_DIV OP_VAR_Y OP_VAR_Z OP_DIV OP_ADD",
        ["algebra", "distribution", "division"],
        grammar_refs=["canonical_form", "expansion"],
    ),
    _sas_rule(
        "sas_rule:fraction_simplify",
        "OP_VAR_X OP_VAR_Z OP_MUL OP_VAR_Y OP_VAR_Z OP_MUL OP_DIV",
        "OP_VAR_X OP_VAR_Y OP_DIV",
        ["algebra", "fraction", "simplification"],
        grammar_refs=["canonical_form", "fraction_simplify"],
    ),
    _sas_rule(
        "sas_rule:fraction_add_same_denom",
        "OP_VAR_X OP_VAR_Z OP_DIV OP_VAR_Y OP_VAR_Z OP_DIV OP_ADD",
        "OP_VAR_X OP_VAR_Y OP_ADD OP_VAR_Z OP_DIV",
        ["algebra", "fraction", "addition"],
        grammar_refs=["canonical_form", "fraction_addition"],
    ),
    _sas_rule(
        "sas_rule:power_product",
        "OP_VAR_X OP_VAR_Y OP_POWER OP_VAR_X OP_VAR_Z OP_POWER OP_MUL",
        "OP_VAR_X OP_VAR_Y OP_VAR_Z OP_ADD OP_POWER",
        ["algebra", "exponentiation", "product_rule"],
        grammar_refs=["canonical_form", "power_simplify"],
    ),
    _sas_rule(
        "sas_rule:power_quotient",
        "OP_VAR_X OP_VAR_Y OP_POWER OP_VAR_X OP_VAR_Z OP_POWER OP_DIV",
        "OP_VAR_X OP_VAR_Y OP_VAR_Z OP_SUB OP_POWER",
        ["algebra", "exponentiation", "quotient_rule"],
        grammar_refs=["canonical_form", "power_simplify"],
    ),
    _sas_rule(
        "sas_rule:power_of_power",
        "OP_VAR_X OP_VAR_Y OP_POWER OP_VAR_Z OP_POWER",
        "OP_VAR_X OP_VAR_Y OP_VAR_Z OP_MUL OP_POWER",
        ["algebra", "exponentiation", "power_rule"],
        grammar_refs=["canonical_form", "power_simplify"],
    ),
    _sas_rule(
        "sas_rule:double_negation",
        "OP_VAR_X OP_NEGATE OP_NEGATE",
        "OP_VAR_X",
        ["algebra", "negation", "identity"],
    ),
    _sas_rule(
        "sas_rule:square_sqrt",
        "OP_VAR_X OP_CONST 2.0 OP_POWER OP_SQRT",
        "OP_VAR_X",
        ["algebra", "sqrt", "power"],
        grammar_refs=["canonical_form", "radical_simplify"],
    ),
]


def build_sas_rule_stars() -> list[MeaningCentricStar]:
    """Return the foundational SAS canonicalization and identity rules."""
    return [MeaningCentricStar.from_dict(rule.to_dict()) for rule in CANONICALIZATION_RULES]


__all__ = ["CANONICALIZATION_RULES", "DEFEASIBLE_METADATA", "build_sas_rule_stars"]
