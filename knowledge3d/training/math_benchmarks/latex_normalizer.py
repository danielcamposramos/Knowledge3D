"""
LaTeX normalization for MATH benchmark preprocessing.

This runs during dataset loading (not in the hot path) to map common LaTeX
notation into natural-language phrases that strict grammar regexes can match.
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple


LATEX_NORMALIZATION_RULES: Tuple[Tuple[str, str], ...] = (
    # Roots.
    (r"\\sqrt\[(\d+)\]\{([^}]+)\}", r"(\2)^(1/\1)"),
    (r"\\sqrt\{([^}]+)\}", r"(\1)^(1/2)"),
    # Context-aware f'(a) normalization (Fallback for legacy rules)
    (r"f'\(\s*(\d+\.?\d*)\s*\)", r"derivative of f at x=\1"),

    # Leibniz evaluation bar: d/dx[...]|_{x=a}
    (r"\\frac\{d\}\{dx\}\s*\[([^\]]+)\]\s*\|_\{x\s*=\s*([^}]+)\}", r"derivative of \1 at x=\2"),
    (r"\\frac\{d\}\{dx\}\s*\(([^\)]+)\)\s*\|_\{x\s*=\s*([^}]+)\}", r"derivative of \1 at x=\2"),
    (r"\|_\{x\s*=\s*([^}]+)\}", r" at x=\1"),
    
    # Derivatives (specific before generic fractions).
    (r"\\dfrac\{d\}\{dx\}", "derivative of"),
    (r"\\frac\{d\}\{dx\}", "derivative of"),
    (r"\\dfrac\{d\}\{d([a-zA-Z])\}", r"derivative with respect to \1 of"),
    (r"\\frac\{d\}\{d([a-zA-Z])\}", r"derivative with respect to \1 of"),
    (r"([A-Za-z])'\(([^)]+)\)", r"derivative of \1(\2)"),
    # Integrals.
    (r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r"integral from \1 to \2 of"),
    (r"\\int_([^\s\^]+)\^\{([^}]+)\}", r"integral from \1 to \2 of"),
    (r"\\int_([^\s\^]+)\^([^\s]+)", r"integral from \1 to \2 of"),
    (r"\\int", "integral of"),
    # Fractions (quotients).
    (r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"(\1) divided by (\2)"),
    (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1) divided by (\2)"),
    # Multiplication.
    (r"\\cdot", "*"),
    # Exponents.
    (r"\^\{([^}]+)\}", r"^(\1)"),
    (r"\^([A-Za-z0-9]+)", r"^(\1)"),
)


_STRIP_RULES: Tuple[Tuple[str, str], ...] = (
    (r"\$([^$]+)\$", r"\1"),  # Inline math delimiters.
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\!", ""),
    (r"\\\s+", " "),
)


def _apply_rules(text: str, rules: Iterable[Tuple[str, str]]) -> str:
    normalized = text
    for pattern, replacement in rules:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def normalize_latex_to_natural(text: str) -> str:
    """
    Normalize LaTeX notation to natural language for regex matching.
    """
    if not text:
        return text
    normalized = _apply_rules(text, LATEX_NORMALIZATION_RULES)
    normalized = _apply_rules(normalized, _STRIP_RULES)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


__all__ = ["normalize_latex_to_natural", "LATEX_NORMALIZATION_RULES"]
