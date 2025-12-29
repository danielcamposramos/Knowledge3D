"""
Placeholder language grammar rules (Layer 3).

Extend this file with ~300+ language rules extracted from PDFs.
Rules should use:
  - symbol_refs: references to Math/Character Galaxy codepoints (Layer 1)
  - word_refs: references to Word Galaxy IDs (Layer 2)
No duplication of strings/glyphs.
"""

from __future__ import annotations

from typing import List

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


def get_language_rules() -> List[GrammarRule]:
    """
    Return language grammar rules. Currently empty placeholder.
    Populate with rules extracted from PDFs (Layer 3).
    """
    return []
