"""
Unified Galaxy Loader - Load ALL galaxies for benchmark evaluation.

This is the sovereign approach: the model "knows" by loading galaxies.
No external preprocessing - knowledge lives in Galaxy storage.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class UnifiedGalaxyLoader:
    """
    Load all available galaxies.

    Per FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md:
    - Layer 1: Character/Math Symbol galaxies (glyphs, symbols)
    - Layer 3: Grammar galaxies (transformation rules)
    - Layer 4: Meta-rules (when/why to apply)
    """

    def __init__(self) -> None:
        self.galaxies: Dict[str, Any] = {}
        self._load_all_galaxies()

    def _load_all_galaxies(self) -> None:
        """Load all available galaxies."""
        # Layer 1: Math Symbol Galaxy
        from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
        self.galaxies["math_symbols"] = MATH_GALAXY

        # Cross-domain fundamentals (generic equations used for test-time compute).
        from knowledge3d.cranium.generic_equations import GENERIC_EQUATION_GALAXY
        self.galaxies["generic_equations"] = GENERIC_EQUATION_GALAXY

        # Layer 1: Character Galaxy (procedural fonts)
        try:
            from knowledge3d.cranium.procedural_fonts import CharacterGalaxy
            self.galaxies["characters"] = CharacterGalaxy()
        except Exception:
            # CharacterGalaxy not present in this build; drawing galaxy covers glyph base.
            pass

        # Layer 1: Drawing Galaxy (glyph/raster base)
        from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
        self.galaxies["drawing"] = DrawingGalaxy()

        # Layer 1/Reality: Physics/Chem/Bio stacked reality nodes (sovereign GPU RPN systems).
        try:
            from knowledge3d.cranium.reality_galaxy import RealityGalaxy

            reality = RealityGalaxy()
            reality.load_galaxy()
            self.galaxies["reality"] = reality
        except Exception:
            # Optional in some installs; keep benchmarks runnable without it.
            pass

        # Layer 2: Word Galaxy (semantics)
        from knowledge3d.cranium.word_galaxy import get_word_galaxy
        self.galaxies["word_galaxy"] = get_word_galaxy()

        # Layer 3: Grammar Galaxy
        from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
        self.galaxies["grammar"] = GrammarGalaxy()

        # Layer 3: Math Grammar Rules
        from knowledge3d.training.arc_agi.math_grammar_rules import (
            WORD_PROBLEM_RULES,
            ALGEBRA_RULES,
            GSM8K_TEMPLATES,
            COMPETITION_MATH_RULES,
            CALCULUS_RULES,
            LINEAR_ALGEBRA_RULES,
            SET_THEORY_RULES,
            LOGIC_RULES,
            STATISTICS_RULES,
            FINANCE_RULES,
            SYMBOLIC_RULES,
            SOVEREIGN_MATH_RULES,
        )

        self.galaxies["word_rules"] = WORD_PROBLEM_RULES
        self.galaxies["algebra_rules"] = ALGEBRA_RULES
        self.galaxies["gsm8k_templates"] = GSM8K_TEMPLATES
        self.galaxies["competition_rules"] = COMPETITION_MATH_RULES
        self.galaxies["calculus_rules"] = CALCULUS_RULES
        self.galaxies["linear_algebra_rules"] = LINEAR_ALGEBRA_RULES
        self.galaxies["set_rules"] = SET_THEORY_RULES
        self.galaxies["logic_rules"] = LOGIC_RULES
        self.galaxies["statistics_rules"] = STATISTICS_RULES
        self.galaxies["finance_rules"] = FINANCE_RULES
        self.galaxies["symbolic_rules"] = SYMBOLIC_RULES
        self.galaxies["sovereign_rules"] = SOVEREIGN_MATH_RULES

        # Hard guard: all critical galaxies must be present
        required = ("math_symbols", "drawing", "word_galaxy", "grammar", "generic_equations")
        missing = [r for r in required if r not in self.galaxies]
        if missing:
            raise RuntimeError(f"Missing required galaxies: {missing}")

    def lookup_symbol(self, symbol: str) -> Optional[Any]:
        """Look up a symbol across all loaded galaxies."""
        if "math_symbols" in self.galaxies:
            entry = self.galaxies["math_symbols"].lookup(symbol)
            if entry:
                return entry

        if "characters" in self.galaxies:
            entry = self.galaxies["characters"].lookup(symbol)
            if entry:
                return entry

        return None

    def compose_rpn(self, symbol: str, *args: object) -> str:
        """Compose RPN from symbol using galaxy templates."""
        if "math_symbols" in self.galaxies:
            return self.galaxies["math_symbols"].compose_rpn(symbol, *args)
        return ""

    def get_grammar_rules(self):
        """Get all grammar rules for pattern matching."""
        rules = []
        for key in (
            "word_rules",
            "gsm8k_templates",
            "algebra_rules",
            "competition_rules",
            "calculus_rules",
            "linear_algebra_rules",
            "set_rules",
            "logic_rules",
            "statistics_rules",
            "finance_rules",
            "symbolic_rules",
            "sovereign_rules",
        ):
            if key in self.galaxies:
                rules.extend(self.galaxies[key])
        return rules

    def report(self) -> str:
        """Report loaded galaxy statistics."""
        lines = ["=== GALAXY LOADER STATUS ==="]
        for name, galaxy in self.galaxies.items():
            if hasattr(galaxy, "__len__"):
                try:
                    lines.append(f"  {name}: {len(galaxy)} entries")
                    continue
                except Exception:
                    pass
            if hasattr(galaxy, "all_symbols"):
                try:
                    lines.append(f"  {name}: {len(galaxy.all_symbols())} symbols")
                    continue
                except Exception:
                    pass
            lines.append(f"  {name}: loaded")
        return "\n".join(lines)


# Global instance
UNIFIED_GALAXY = UnifiedGalaxyLoader()
