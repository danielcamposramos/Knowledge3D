"""Grammar normalizer: map variant/slang/typo tokens to canonical forms."""

from __future__ import annotations

from typing import List

from .grammar_galaxy import GrammarGalaxy


class GrammarNormalizer:
    """Normalizes tokens using grammar galaxy variant mappings."""

    def __init__(self, galaxy: GrammarGalaxy | None = None):
        self.galaxy = galaxy or GrammarGalaxy()

    def normalize_tokens(self, tokens: List[str], language: str) -> List[str]:
        return self.galaxy.normalize_tokens(tokens, language)

    def normalize_text(self, text: str, language: str) -> str:
        tokens = text.strip().split()
        norm = self.normalize_tokens(tokens, language)
        return " ".join(norm)


__all__ = ["GrammarNormalizer"]
