"""
Quality scoring for RPN programs.

Favors compound/novel programs and penalizes trivial duplicates.
"""

from __future__ import annotations

import re
from typing import Set


class QualityScorer:
    """Score programs by combining execution score, complexity, and novelty."""

    SIMPLE_PATTERNS = {
        r"^\d+ rotate$",
        r"^\d+ flip$",
        r"^\d+ \d+ RECOLOR$",
        r"^IDENTITY$",
    }

    COMPOUND_INDICATORS = ["compose", "COMPOSE", "then", "THEN", "and", "AND"]

    def __init__(self) -> None:
        self.seen_signatures: Set[str] = set()

    def score_quality(self, program: str, execution_score: float) -> float:
        """
        Compute quality score.

        Starts from execution score and adjusts for simplicity, composition, and novelty.
        """
        quality = execution_score

        is_simple = any(re.match(pattern, program) for pattern in self.SIMPLE_PATTERNS)
        if is_simple:
            quality *= 0.5

        is_compound = any(indicator in program for indicator in self.COMPOUND_INDICATORS)
        if is_compound:
            quality = min(quality * 1.3, 1.0)

        signature = self._pattern_signature(program)
        if signature not in self.seen_signatures:
            quality = min(quality * 1.2, 1.0)
            self.seen_signatures.add(signature)

        return float(quality)

    def get_complexity_level(self, program: str) -> str:
        """Classify program complexity."""
        is_simple = any(re.match(pattern, program) for pattern in self.SIMPLE_PATTERNS)
        is_compound = any(indicator in program for indicator in self.COMPOUND_INDICATORS)
        if is_compound:
            return "compound"
        if is_simple:
            return "simple"
        return "intermediate"

    @staticmethod
    def _pattern_signature(program: str) -> str:
        """Normalize numbers to 'N' for a structural signature."""
        tokens = program.split()
        signature = []
        for tok in tokens:
            signature.append("N" if tok.isdigit() else tok)
        return " ".join(signature)


__all__ = ["QualityScorer"]
