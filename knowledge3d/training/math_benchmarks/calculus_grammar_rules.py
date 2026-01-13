"""
Calculus Grammar Rules (numeric RPN).

These rules are bootstraps for routing targets and numeric execution.
They focus on explicit numeric patterns (e.g., derivatives evaluated at x=a).
"""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


def get_calculus_rules() -> List[GrammarRule]:
    """Return numeric calculus grammar rules with polyglot notation variants."""
    rules: List[GrammarRule] = []

    def add(rule_id: str, pattern: str, rpn_program: str, domain: str = "calculus", description: str = "") -> None:
        rules.append(
            GrammarRule(
                rule_id=rule_id,
                language="math",
                pattern=pattern,
                rpn_program=rpn_program,
                domain=domain,
                description=description,
            )
        )

    # === 1. Power Rule: d/dx(x^n) at x=a ===
    add(
        "apply_power_rule_natural",
        r"(?:derivative|diff)\s*(?:of)?\s*x\^(\d+\.?\d*)\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {1} {0} 1 - ^ *",
        description="d/dx(x^n) at x=a -> n*a^(n-1)",
    )
    add(
        "apply_power_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*x\s*\^\s*(\d+\.?\d*)",
        "{1} {0} {1} 1 - ^ *",
        description="f'(a) where f(x)=x^n",
    )
    add(
        "apply_power_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?x\s*\^\s*(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {1} {0} 1 - ^ *",
        description="d/dx[x^n]|_{x=a}",
    )

    # === 2. Constant Multiple Rule: d/dx(k*x^n) at x=a ===
    add(
        "apply_constant_multiple_rule_natural",
        r"(?:derivative|diff).*?(\d+\.?\d*)\s*\*?\s*x\^(\d+\.?\d*)\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {1} * {2} {1} 1 - ^ *",
    )
    add(
        "apply_constant_multiple_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*(\d+\.?\d*)\s*\*?\s*x\^(\d+\.?\d*)",
        "{1} {2} * {0} {2} 1 - ^ *",
    )
    add(
        "apply_constant_multiple_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?(\d+\.?\d*)\s*\*?\s*x\^(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {1} * {2} {1} 1 - ^ *",
    )

    # === 3. Sum Rule: d/dx(x^n + x^m) at x=a ===
    add(
        "apply_sum_rule_natural",
        r"(?:derivative|diff).*?x\^(\d+\.?\d*)\s*\+\s*x\^(\d+\.?\d*)\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {2} {0} 1 - ^ * {1} {2} {1} 1 - ^ * +",
    )
    add(
        "apply_sum_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*x\^(\d+\.?\d*)\s*\+\s*x\^(\d+\.?\d*)",
        "{1} {0} {1} 1 - ^ * {2} {0} {2} 1 - ^ * +",
    )
    add(
        "apply_sum_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?x\^(\d+\.?\d*)\s*\+\s*x\^(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {2} {0} 1 - ^ * {1} {2} {1} 1 - ^ * +",
    )

    # === 4. Product Rule: d/dx(x^n * x^m) at x=a ===
    add(
        "apply_product_rule_natural",
        r"(?:derivative|diff).*?\(?x\^(\d+\.?\d*)\)?\s*(?:\*|\\cdot|\\times)\s*\(?x\^(\d+\.?\d*)\)?\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {2} {0} 1 - ^ * {2} {1} ^ * {2} {0} ^ {1} {2} {1} 1 - ^ * * +",
    )
    add(
        "apply_product_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*x\^(\d+\.?\d*)\s*(?:\*|\\cdot|\\times)\s*x\^(\d+\.?\d*)",
        "{1} {0} {1} 1 - ^ * {0} {2} ^ * {0} {1} ^ {2} {0} {2} 1 - ^ * * +",
    )
    add(
        "apply_product_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?x\^(\d+\.?\d*)\s*(?:\*|\\cdot|\\times)\s*x\^(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {2} {0} 1 - ^ * {2} {1} ^ * {2} {0} ^ {1} {2} {1} 1 - ^ * * +",
    )

    # === 5. Quotient Rule: d/dx(x^n / x^m) at x=a ===
    add(
        "apply_quotient_rule_natural",
        r"(?:derivative|diff).*?\(?x\^(\d+\.?\d*)\)?\s*/\s*\(?x\^(\d+\.?\d*)\)?\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {2} {0} 1 - ^ * {2} {1} ^ * {2} {0} ^ {1} {2} {1} 1 - ^ * * - {2} {1} ^ 2 ^ /",
    )
    add(
        "apply_quotient_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*x\^(\d+\.?\d*)\s*/\s*x\^(\d+\.?\d*)",
        "{1} {0} {1} 1 - ^ * {0} {2} ^ * {0} {1} ^ {2} {0} {2} 1 - ^ * * - {0} {2} ^ 2 ^ /",
    )
    add(
        "apply_quotient_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?x\^(\d+\.?\d*)\s*/\s*x\^(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {2} {0} 1 - ^ * {2} {1} ^ * {2} {0} ^ {1} {2} {1} 1 - ^ * * - {2} {1} ^ 2 ^ /",
    )

    # === 6. Chain Rule: d/dx((x^n)^m) at x=a ===
    add(
        "apply_chain_rule_natural",
        r"(?:derivative|diff).*?\(?x\^(\d+\.?\d*)\)?\s*\^\s*(\d+\.?\d*)\s*at\s*x\s*=\s*(\d+\.?\d*)",
        "{0} {1} * {2} {0} {1} * 1 - ^ *",
    )
    add(
        "apply_chain_rule_prime",
        r"f'\(\s*(\d+\.?\d*)\s*\).*?f\(x\)\s*=\s*\(?x\^(\d+\.?\d*)\)?\s*\^\s*(\d+\.?\d*)",
        "{1} {2} * {0} {1} {2} * 1 - ^ *",
    )
    add(
        "apply_chain_rule_leibniz",
        r"(?:\\frac\{d\}\{dx\}|d/dx).*?\(?x\^(\d+\.?\d*)\)?\s*\^\s*(\d+\.?\d*).*?\|_\{x\s*=\s*(\d+\.?\d*)\}",
        "{0} {1} * {2} {0} {1} * 1 - ^ *",
    )

    # === 7. Integration by Parts: ∫ x e^x from a to b ===
    add(
        "apply_integration_by_parts_natural",
        r"(?:integral(?:\s+of)?)\s*x\s*e\^x\s*from\s*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)",
        "{1} 1 - {1} exp * {0} 1 - {0} exp * -",
    )
    add(
        "apply_integration_by_parts_latex",
        r"\\int_\{?(\d+\.?\d*)\}?\^\{?(\d+\.?\d*)\}?\s*x\s*e\^x\s*dx?",
        "{1} 1 - {1} exp * {0} 1 - {0} exp * -",
    )

    # === 8. Fundamental Theorem: ∫ x^n from a to b ===
    add(
        "apply_fundamental_theorem_calculus_natural",
        r"(?:integral(?:\s+of)?)\s*x\^(\d+\.?\d*)\s*from\s*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)",
        "{2} {0} 1 + ^ {1} {0} 1 + ^ - {0} 1 + /",
    )
    add(
        "apply_fundamental_theorem_calculus_latex",
        r"\\int_\{?(\d+\.?\d*)\}?\^\{?(\d+\.?\d*)\}?\s*x\^(\d+\.?\d*)\s*dx?",
        "{1} {2} 1 + ^ {0} {2} 1 + ^ - {2} 1 + /",
    )

    # === 9. Pythagorean Identity ===
    add(
        "apply_pythagorean_identity_natural",
        r"sin\^2\([^)]*\)\s*\+\s*cos\^2\([^)]*\)",
        "1",
        domain="geometry",
    )
    add(
        "apply_pythagorean_identity_latex",
        r"\\sin\^2\([^)]*\)\s*\+\s*\\cos\^2\([^)]*\)",
        "1",
        domain="geometry",
    )

    return rules


__all__ = ["get_calculus_rules"]
