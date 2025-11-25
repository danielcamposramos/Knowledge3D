"""
ProgramComposer: combines Drawing + Grammar candidates and classifies outcomes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


class ProgramComposer:
    """Compose visual programs with grammar transforms (lightweight, sovereign)."""

    def __init__(self) -> None:
        pass

    def compose(self, drawing_program: str, grammar_rules: List[GrammarRule]) -> List[Tuple[str, str]]:
        """
        Compose drawing program with grammar rules.

        Returns:
            List of (program, program_type) where program_type ∈ {'visual','transformation','hybrid'}
        """
        compositions: List[Tuple[str, str]] = []

        for rule in grammar_rules:
            domain = getattr(rule, "domain", "")
            if domain == "drawing":
                compositions.append((f"{drawing_program} {rule.rpn_program}", "visual"))
            elif domain in {"spatial", "geometry"} or rule.pattern in {"transform", "primitive"}:
                compositions.append((f"{drawing_program} {rule.rpn_program}", "transformation"))
            else:
                compositions.append((f"{drawing_program} {rule.rpn_program}", "hybrid"))

        return compositions

    def classify(self, program: str) -> str:
        lowered = program.lower()
        if "cell" in lowered or "grid" in lowered:
            return "visual"
        if "rotate" in lowered or "flip" in lowered or "translate" in lowered:
            return "transformation"
        return "hybrid"


__all__ = ["ProgramComposer"]
