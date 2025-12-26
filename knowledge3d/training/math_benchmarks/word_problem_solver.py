"""
Word problem solver using grammar-style pattern matching.

Generates RPN programs for basic arithmetic word problems (GSM8K-style)
without relying on CPU fallbacks or external libraries.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from knowledge3d.training.arc_agi.math_grammar_rules import (
    WORD_PROBLEM_RULES,
    COMPETITION_MATH_RULES,
)


class WordProblemSolver:
    """
    Solve word problems by matching text against arithmetic grammar rules
    and synthesizing an RPN program that can be executed by the sovereign
    ModularRPNEngine.
    """

    def __init__(self) -> None:
        # Limit to word + competition rules to avoid over-greedy symbolic regex (e.g., |x|)
        self.rules = list(WORD_PROBLEM_RULES + COMPETITION_MATH_RULES)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for fast matching."""
        self.number_re = re.compile(r"(\d+(?:\.\d+)?)")
        self.phrase_patterns = [
            (re.compile(r"(\d+(?:\.\d+)?)\s*more\s*than", re.IGNORECASE), "wp_addition_more"),
            (re.compile(r"half\s*(?:of|as\s*many)", re.IGNORECASE), "wp_division_half"),
            (re.compile(r"twice\s*(?:as\s*many|the)", re.IGNORECASE), "wp_multiplication_twice"),
            (re.compile(r"double\s*(?:the)?", re.IGNORECASE), "wp_multiplication_double"),
            (re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of", re.IGNORECASE), "wp_percentage_of"),
            (re.compile(r"total|altogether|combined", re.IGNORECASE), "wp_addition_total"),
            (re.compile(r"(\d+(?:\.\d+)?)\s*(?:times|x)\s*(\d+(?:\.\d+)?)", re.IGNORECASE), "wp_multiplication_times"),
            (re.compile(r"split|divided\s*(?:by|among)", re.IGNORECASE), "wp_division_split"),
            (re.compile(r"per\s*(?:hour|minute|day|week)", re.IGNORECASE), "wp_multiplication_per"),
            (re.compile(r"remaining|left\s*over", re.IGNORECASE), "wp_subtraction_remaining"),
            (re.compile(r"less\s*than", re.IGNORECASE), "wp_subtraction_less"),
        ]

    def _score_rule_match(self, pattern: re.Pattern, match: re.Match) -> float:
        """Score a rule match by pattern length and capture count."""
        specificity = len(pattern.pattern)
        captures = len(match.groups())
        return specificity * 0.1 + captures * 0.5

    def solve(self, problem_text: str) -> Dict[str, object]:
        """
        Generate an RPN program from a word problem.

        Returns a dictionary with numbers, matched rule ids, and rpn_program.
        """
        numbers = [float(n) for n in self.number_re.findall(problem_text)]
        scored_rules: List[Tuple[str, float]] = []
        text_lower = problem_text.lower()

        for pattern, rule_id in self.phrase_patterns:
            m = pattern.search(text_lower)
            if m:
                scored_rules.append((rule_id, self._score_rule_match(pattern, m)))

        # Order rules by score (higher first) to encourage more specific matches
        matched_rules = [rid for rid, _ in sorted(scored_rules, key=lambda x: x[1], reverse=True)]

        # Try regex-based grammar match for richer competition patterns
        rule_match = self.match_rule(problem_text)
        rpn_program = ""
        if rule_match:
            rule, variables = rule_match
            matched_rules.append(rule.rule_id)
            rpn_program = self._render_rpn(rule.rpn_program, variables)
        else:
            # Direct LaTeX binomial parsing
            latex_binom = re.search(r"\\binom\{(\d+)\}\{(\d+)\}", problem_text)
            if latex_binom:
                n, k = latex_binom.groups()
                rpn_program = f"{n} {k} binomial"
                matched_rules.append("comp_latex_binom")
            else:
                rpn_program = self._generate_rpn(numbers, matched_rules)

        return {
            "numbers": numbers,
            "matched_rules": matched_rules,
            "rpn_program": rpn_program,
        }

    def _generate_rpn(self, numbers: List[float], rules: List[str]) -> str:
        """
        Generate a compositional RPN program from numbers and matched rules.

        Rules can chain; we avoid early returns where operations should combine.
        """
        if not numbers:
            return ""

        base = numbers[0]

        # Half/Double with total aggregation
        if "wp_division_half" in rules:
            if len(numbers) != 1:
                return ""
            if "wp_addition_total" in rules or "wp_addition_more" in rules:
                return f"{base} DUP 2 / +"
            return f"{base} 2 /"

        if "wp_multiplication_twice" in rules or "wp_multiplication_double" in rules:
            if len(numbers) != 1:
                return ""
            if "wp_addition_total" in rules or "wp_addition_more" in rules:
                return f"{base} DUP 2 * +"
            return f"{base} 2 *"

        if "wp_percentage_of" in rules and len(numbers) >= 2:
            pct, base_val = numbers[0], numbers[1]
            if "wp_subtraction_remaining" in rules:
                return f"{base_val} DUP {pct} * 100 / -"
            return f"{base_val} {pct} * 100 /"

        if "wp_addition_total" in rules or "wp_addition_more" in rules:
            # Summing all extracted numbers is often wrong for GSM8K-style prompts.
            # Only attempt small, clearly-aggregative cases.
            if not (2 <= len(numbers) <= 3):
                return ""
            rpn_parts: List[str] = [str(numbers[0])]
            for n in numbers[1:]:
                rpn_parts.append(str(n))
                rpn_parts.append("+")
            return " ".join(rpn_parts)

        if "wp_subtraction_less" in rules and len(numbers) >= 2:
            if len(numbers) != 2:
                return ""
            return f"{numbers[1]} {numbers[0]} -"

        if "wp_subtraction_remaining" in rules and len(numbers) >= 2:
            if len(numbers) != 2:
                return ""
            return f"{numbers[0]} {numbers[1]} -"

        if "wp_multiplication_times" in rules and len(numbers) >= 2:
            if len(numbers) != 2:
                return ""
            return f"{numbers[0]} {numbers[1]} *"

        if "wp_division_split" in rules and len(numbers) >= 2:
            if len(numbers) != 2:
                return ""
            return f"{numbers[0]} {numbers[1]} /"

        # Default: no confident operation.
        return ""

    def match_rule(self, problem_text: str) -> Optional[Tuple[object, Dict[str, object]]]:
        """Match text against grammar rules, extracting captured variables."""
        text_lower = problem_text.lower()
        for rule in self.rules:
            if "\\\\binom" in rule.pattern or "\\\\frac" in rule.pattern or "\\\\sqrt" in rule.pattern:
                pattern = rule.pattern
            else:
                pattern = rule.pattern.replace("\\\\", "\\")
            try:
                m = re.search(pattern, problem_text, re.IGNORECASE)
                if not m:
                    continue
                groups = m.groups()
                variables: Dict[str, object] = {}
                var_names = ["n", "k", "a", "b", "p", "r", "x", "num", "denom", "m"]
                for i, val in enumerate(groups):
                    key = var_names[i] if i < len(var_names) else f"g{i}"
                    try:
                        variables[key] = float(val)
                    except (ValueError, TypeError):
                        variables[key] = val
                    variables[f"g{i}"] = variables[key]
                return rule, variables
            except re.error:
                if rule.pattern.lower() in text_lower:
                    return rule, {}
        return None

    def _render_rpn(self, template: str, variables: Dict[str, object]) -> str:
        """Substitute placeholders like {n}, {k} in RPN template."""
        rpn = template
        for key, val in variables.items():
            rpn = rpn.replace("{" + key + "}", str(val))
        return rpn


__all__ = ["WordProblemSolver"]
