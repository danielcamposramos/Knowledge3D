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
from typing import Dict, List, Optional, Sequence
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

    # ------------------------------------------------------------------ #
    # Galaxy-based tokenization ("reading")
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List["WordEntry"]:
        """
        Tokenize raw text into WordEntry objects.

        This is a Galaxy-aligned "reading" primitive used by TRM and
        higher-level solvers to avoid external preprocessing shortcuts.
        """
        tokens = list(_segment_text(text))
        return _tokenize_with_number_words(tokens)


@dataclass(frozen=True)
class WordEntry:
    token: str
    normalized: str
    category: str
    value: Optional[float] = None
    rpn_literal: Optional[str] = None
    role: Optional[str] = None


def _segment_text(text: str) -> Sequence[str]:
    """
    Segment text into tokens without regex.

    Groups contiguous digits (including '.') into one token, contiguous letters
    into one token, and treats punctuation as standalone tokens.
    """
    tokens: List[str] = []
    buf: List[str] = []
    buf_kind: Optional[str] = None  # "alpha" | "num" | "latex"

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf:
            tokens.append("".join(buf))
        buf = []
        buf_kind = None

    for idx, ch in enumerate(text):
        if ch.isspace():
            flush()
            continue

        # LaTeX command: \frac, \binom ...
        if ch == "\\":
            flush()
            buf.append(ch)
            buf_kind = "latex"
            continue

        if buf_kind == "latex":
            if ch.isalpha():
                buf.append(ch)
                continue
            flush()
            # fallthrough to handle current char as its own token

        if ch.isdigit() or ch == ".":
            if buf_kind in (None, "num"):
                buf.append(ch)
                buf_kind = "num"
                continue
            flush()
            buf.append(ch)
            buf_kind = "num"
            continue

        # Thousands separator inside numeric token ("3,000" -> "3000").
        if ch == "," and buf_kind == "num":
            nxt = text[idx + 1] if idx + 1 < len(text) else ""
            if nxt.isdigit():
                continue

        if ch.isalpha():
            if buf_kind in (None, "alpha"):
                buf.append(ch)
                buf_kind = "alpha"
                continue
            flush()
            buf.append(ch)
            buf_kind = "alpha"
            continue

        # Punctuation / symbol
        flush()
        tokens.append(ch)

    flush()
    return tokens


_VERBS = {
    "sold",
    "sell",
    "gives",
    "gave",
    "give",
    "buys",
    "bought",
    "gets",
    "got",
    "earned",
    "spends",
    "spent",
    "uses",
    "used",
}
_AGGREGATORS = {"altogether", "total", "sum", "combined", "in", "all"}
_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "as",
    "many",
    "much",
    "did",
    "does",
    "do",
    "how",
    "what",
    "when",
    "where",
    "then",
}

_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALE_WORDS: Dict[str, int] = {"hundred": 100, "thousand": 1000}


def _parse_number_words(words: List[str]) -> int | None:
    """
    Parse a short English number phrase into an int.

    Supports: one..nineteen, tens, hundred/thousand compositions.
    Examples:
      ["ten"] -> 10
      ["twenty", "five"] -> 25
      ["one", "hundred", "twenty", "three"] -> 123
      ["two", "thousand", "ten"] -> 2010
    """
    if not words:
        return None
    total = 0
    current = 0
    used_any = False
    for w in words:
        lw = w.lower()
        if lw in _NUMBER_WORDS:
            current += _NUMBER_WORDS[lw]
            used_any = True
            continue
        if lw in _SCALE_WORDS:
            scale = _SCALE_WORDS[lw]
            if current == 0:
                current = 1
            current *= scale
            used_any = True
            if scale >= 1000:
                total += current
                current = 0
            continue
        return None
    if not used_any:
        return None
    return total + current


def _tokenize_with_number_words(tokens: List[str]) -> List["WordEntry"]:
    """
    Tokenize with number-word fusion.

    This keeps number understanding inside WordGalaxy (Galaxy-aligned),
    reducing no_rule_match for GSM8K prompts that spell numbers ("ten").
    """
    entries: List[WordEntry] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok:
            i += 1
            continue
        if tok.isalpha():
            j = i
            phrase: List[str] = []
            while j < len(tokens) and tokens[j] and tokens[j].isalpha():
                lw = tokens[j].lower()
                if lw in _NUMBER_WORDS or lw in _SCALE_WORDS:
                    phrase.append(tokens[j])
                    j += 1
                    continue
                break
            if phrase:
                val = _parse_number_words(phrase)
                if val is not None:
                    lit = str(int(val))
                    raw = " ".join(phrase)
                    entries.append(
                        WordEntry(
                            token=raw,
                            normalized=lit,
                            category="number",
                            value=float(val),
                            rpn_literal=lit,
                        )
                    )
                    i = j
                    continue
        entries.append(_infer_word_entry(tok))
        i += 1
    return entries


def _infer_word_entry(token: str) -> WordEntry:
    raw = token
    normalized = token.strip()
    lower = normalized.lower()

    # Numbers (integer/float)
    is_num = False
    if normalized and (normalized[0].isdigit() or normalized[0] == "."):
        dot_count = 0
        ok = True
        for ch in normalized:
            if ch.isdigit():
                continue
            if ch == ".":
                dot_count += 1
                if dot_count > 1:
                    ok = False
                    break
                continue
            ok = False
            break
        is_num = ok and any(ch.isdigit() for ch in normalized)

    if is_num:
        try:
            val = float(normalized)
        except Exception:
            val = None
        return WordEntry(
            token=raw,
            normalized=lower,
            category="number",
            value=val,
            rpn_literal=normalized,
        )

    if lower in _VERBS:
        return WordEntry(token=raw, normalized=lower, category="verb")
    if lower == "half":
        return WordEntry(token=raw, normalized=lower, category="fraction", value=0.5)
    if lower in _AGGREGATORS:
        return WordEntry(token=raw, normalized=lower, category="aggregation")

    if normalized and normalized[0].isupper():
        return WordEntry(token=raw, normalized=lower, category="proper_noun", role="entity")

    if lower in _STOPWORDS:
        return WordEntry(token=raw, normalized=lower, category="stopword")

    if lower.isalpha():
        return WordEntry(token=raw, normalized=lower, category="noun")

    return WordEntry(token=raw, normalized=lower, category="symbol")

# Singleton accessor
_word_galaxy: Optional[WordGalaxy] = None


def get_word_galaxy() -> WordGalaxy:
    global _word_galaxy
    if _word_galaxy is None:
        _word_galaxy = WordGalaxy()
    return _word_galaxy
