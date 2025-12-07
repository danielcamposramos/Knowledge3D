"""
Example math grammar rules leveraging Math Galaxy symlinks.

Rules reference canonical symbols via symbol_refs (codepoints) without duplicating
glyph data. Enables cross-domain discovery and automatic generalization.
"""

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


# Calculus rules using ∑ and ∫
CALCULUS_RULES = [
    GrammarRule(
        rule_id="calc_riemann_sum",
        language="math",
        pattern="∑[i=a..b] f(i)",
        rpn_program="a RECALL b RECALL 1 - swap - 1 + { i STORE f RECALL i RECALL swap CALL } swap times +",
        domain="math_calculus",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "∑[i=1..5] i", "output": "15"}],
    ),
    GrammarRule(
        rule_id="calc_definite_integral",
        language="math",
        pattern="∫[a..b] f(x) dx",
        rpn_program="a RECALL b RECALL f RECALL TRAPEZOIDAL_INTEGRATE",
        domain="math_calculus",
        symbol_refs=[8747],  # ∫
        examples=[{"input": "∫[0..1] x² dx", "output": "0.333"}],
    ),
    GrammarRule(
        rule_id="calc_partial_derivative",
        language="math",
        pattern="∂f/∂x",
        rpn_program="f RECALL x RECALL PARTIAL_DIFF",
        domain="math_calculus",
        symbol_refs=[8706],  # ∂
        examples=[{"input": "∂(x²y)/∂x", "output": "2xy"}],
    ),
]


# Set theory rules using ∈, ∪, ∩
SET_THEORY_RULES = [
    GrammarRule(
        rule_id="set_membership",
        language="math",
        pattern="x ∈ S",
        rpn_program="x RECALL S RECALL CONTAINS",
        domain="math_set",
        symbol_refs=[8712],  # ∈
        examples=[{"input": "3 ∈ {1,2,3}", "output": "true"}],
    ),
    GrammarRule(
        rule_id="set_union",
        language="math",
        pattern="A ∪ B",
        rpn_program="A RECALL B RECALL UNION",
        domain="math_set",
        symbol_refs=[8746],  # ∪
        examples=[{"input": "{1,2} ∪ {2,3}", "output": "{1,2,3}"}],
    ),
    GrammarRule(
        rule_id="set_intersection",
        language="math",
        pattern="A ∩ B",
        rpn_program="A RECALL B RECALL INTERSECT",
        domain="math_set",
        symbol_refs=[8745],  # ∩
        examples=[{"input": "{1,2} ∩ {2,3}", "output": "{2}"}],
    ),
]


# Logic rules using ∀, ∃, ⇒
LOGIC_RULES = [
    GrammarRule(
        rule_id="logic_forall",
        language="math",
        pattern="∀x P(x)",
        rpn_program="DOMAIN RECALL { x STORE P RECALL x RECALL swap CALL } ALL",
        domain="math_logic",
        symbol_refs=[8704],  # ∀
        examples=[{"input": "∀x∈ℕ: x≥0", "output": "true"}],
    ),
    GrammarRule(
        rule_id="logic_exists",
        language="math",
        pattern="∃x P(x)",
        rpn_program="DOMAIN RECALL { x STORE P RECALL x RECALL swap CALL } ANY",
        domain="math_logic",
        symbol_refs=[8707],  # ∃
        examples=[{"input": "∃x∈ℕ: x>10", "output": "true"}],
    ),
    GrammarRule(
        rule_id="logic_implies",
        language="math",
        pattern="P ⇒ Q",
        rpn_program="P RECALL neg Q RECALL or",
        domain="math_logic",
        symbol_refs=[8658],  # ⇒
        examples=[{"input": "rain ⇒ wet", "output": "¬rain ∨ wet"}],
    ),
]


# Statistics rules using ∑ (cross-domain)
STATISTICS_RULES = [
    GrammarRule(
        rule_id="stat_expected_value",
        language="math",
        pattern="E[X] = ∑ xᵢP(xᵢ)",
        rpn_program="VALUES RECALL PROBS RECALL { * } zipwith +",
        domain="math_statistics",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "E[dice]", "output": "3.5"}],
    ),
    GrammarRule(
        rule_id="stat_variance",
        language="math",
        pattern="Var[X] = ∑ (xᵢ-μ)²P(xᵢ)",
        rpn_program="VALUES RECALL MU RECALL { swap - 2 pow } map PROBS RECALL { * } zipwith +",
        domain="math_statistics",
        symbol_refs=[8721, 956],  # ∑, μ
        examples=[{"input": "Var[dice]", "output": "2.917"}],
    ),
]


# Finance rules using ∑ (cross-domain)
FINANCE_RULES = [
    GrammarRule(
        rule_id="fin_npv",
        language="math",
        pattern="NPV = ∑ CFₜ/(1+r)ᵗ",
        rpn_program="CASHFLOWS RECALL RATE RECALL { t STORE 1 RATE + t pow / } mapi +",
        domain="math_finance",
        symbol_refs=[8721],  # ∑
        examples=[{"input": "NPV([100,100,100], 0.1)", "output": "248.69"}],
    ),
]


def get_all_math_rules() -> list:
    """Get all math grammar rules."""
    return (
        CALCULUS_RULES
        + SET_THEORY_RULES
        + LOGIC_RULES
        + STATISTICS_RULES
        + FINANCE_RULES
    )


def register_with_discovery_layer():
    """Register all math rules with the Discovery Layer."""
    from knowledge3d.cranium.discovery_layer import DiscoveryLayer

    discovery = DiscoveryLayer()

    for rule in get_all_math_rules():
        discovery.register_rule(
            rule_id=rule.rule_id, domain=rule.domain, symbol_refs=rule.symbol_refs
        )

    return discovery

