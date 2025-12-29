"""Stub executor for math grammar programs."""

from __future__ import annotations

from typing import Dict

import math


class MathGrammarExecutor:
    """Evaluate very small arithmetic expressions from grammar rules (placeholder)."""

    def evaluate_binary(self, op: str, a: float, b: float) -> float:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op in ("*", "×"):
            return a * b
        if op in ("/", "÷"):
            return a / b if b != 0 else math.inf
        raise ValueError(f"Unsupported op: {op}")

    def evaluate_quadratic(self, a: float, b: float, c: float) -> float:
        """Return discriminant as quick sanity metric."""
        return b * b - 4 * a * c


__all__ = ["MathGrammarExecutor"]
