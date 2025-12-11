"""
Word Galaxy — Layer 2 semantic definitions referencing Layer 1 characters.

Follows the symlink pattern:
- NO glyph duplication
- char_sequence stores Unicode codepoints (ASCII allowed, others must exist in Math Galaxy)
- referenced later by grammar rules via word_refs (Layer 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class WordDefinition:
    """A word/term definition referencing Layer 1 characters."""

    word_id: str
    char_sequence: List[int]  # Unicode codepoints; Layer 1 symlinks
    definition: str
    domain: str
    rpn_context: Optional[str] = None  # Optional RPN usage context
    related_symbols: List[int] = field(default_factory=list)  # Math Galaxy refs
    examples: List[str] = field(default_factory=list)

    def validate_char_sequence(self) -> bool:
        """Ensure all characters exist (ASCII allowed, others must be in Math Galaxy)."""
        from knowledge3d.cranium.math_galaxy import get_math_galaxy

        math_galaxy = get_math_galaxy()
        for cp in self.char_sequence:
            if cp > 127 and math_galaxy.get(cp) is None:
                return False
        return True


class WordGalaxy:
    """Layer 2 storage for words/terms (semantic layer)."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("/K3D/Knowledge3D.local/galaxies/words")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._words: Dict[str, WordDefinition] = {}
        self._load()

    def _words_file(self) -> Path:
        return self.storage_path / "words.json"

    def _load(self) -> None:
        """Load words from storage if present."""
        words_file = self._words_file()
        if words_file.exists():
            data = json.loads(words_file.read_text(encoding="utf-8"))
            for word_data in data:
                word = WordDefinition(**word_data)
                self._words[word.word_id] = word

    def _save(self) -> None:
        """Persist words to storage."""
        words_file = self._words_file()
        data = [asdict(w) for w in self._words.values()]
        words_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_word(self, word: WordDefinition) -> bool:
        """Add a word after validating its character sequence."""
        if not word.validate_char_sequence():
            raise ValueError(f"Invalid char_sequence for {word.word_id}")
        self._words[word.word_id] = word
        self._save()
        return True

    def get(self, word_id: str) -> Optional[WordDefinition]:
        return self._words.get(word_id)

    def search_by_domain(self, domain: str) -> List[WordDefinition]:
        return [w for w in self._words.values() if w.domain == domain]

    def compose_from_text(self, text: str) -> List[int]:
        """Convert plain text to a char_sequence (symlink pattern)."""
        return [ord(c) for c in text]

    def stats(self) -> Dict[str, object]:
        total = len(self._words)
        domains = sorted(set(w.domain for w in self._words.values()))
        avg_len = (
            sum(len(w.char_sequence) for w in self._words.values()) / total
            if total
            else 0.0
        )
        return {
            "total_words": total,
            "domains": domains,
            "avg_char_sequence_len": avg_len,
        }

    def all_words(self) -> List[WordDefinition]:
        """Return all words (for ingestion/reporting)."""
        return list(self._words.values())


# Singleton accessor
_word_galaxy: Optional[WordGalaxy] = None


def get_word_galaxy() -> WordGalaxy:
    global _word_galaxy
    if _word_galaxy is None:
        _word_galaxy = WordGalaxy()
    return _word_galaxy
