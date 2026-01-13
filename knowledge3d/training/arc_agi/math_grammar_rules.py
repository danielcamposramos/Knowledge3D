"""Compatibility wrapper for arc_agi.math_grammar_rules."""

from __future__ import annotations

import importlib


_module = importlib.import_module("math_grammar_rules")

WORD_PROBLEM_RULES = _module.WORD_PROBLEM_RULES
ALGEBRA_RULES = _module.ALGEBRA_RULES
GSM8K_TEMPLATES = _module.GSM8K_TEMPLATES
COMPETITION_MATH_RULES = _module.COMPETITION_MATH_RULES
CALCULUS_RULES = _module.CALCULUS_RULES
LINEAR_ALGEBRA_RULES = _module.LINEAR_ALGEBRA_RULES
SET_THEORY_RULES = _module.SET_THEORY_RULES
LOGIC_RULES = _module.LOGIC_RULES
STATISTICS_RULES = _module.STATISTICS_RULES
FINANCE_RULES = _module.FINANCE_RULES
SYMBOLIC_RULES = _module.SYMBOLIC_RULES
SOVEREIGN_MATH_RULES = _module.SOVEREIGN_MATH_RULES
GALAXY_AWARE_RULES = _module.GALAXY_AWARE_RULES
get_all_math_rules = _module.get_all_math_rules

__all__ = [
    "WORD_PROBLEM_RULES",
    "ALGEBRA_RULES",
    "GSM8K_TEMPLATES",
    "COMPETITION_MATH_RULES",
    "CALCULUS_RULES",
    "LINEAR_ALGEBRA_RULES",
    "SET_THEORY_RULES",
    "LOGIC_RULES",
    "STATISTICS_RULES",
    "FINANCE_RULES",
    "SYMBOLIC_RULES",
    "SOVEREIGN_MATH_RULES",
    "GALAXY_AWARE_RULES",
    "get_all_math_rules",
]
