from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

# Minimal, hand-curated stopwords/clitics per language (kept tiny)
_EN = {
    "the","a","an","of","in","on","and","or","to","for","from","by","with","at",
}
_ES = {
    "el","la","los","las","un","una","de","del","al","en","y","o","para","por","con","a","un","una","unos","unas",
}
_PT = {
    "o","a","os","as","um","uma","de","do","da","dos","das","no","na","nos","nas","em","e","ou","para","por","com","ao","aos","à","às",
}

# Union of tiny lists for cross-lingual normalization without language detection
STOPWORDS = set().union(_EN, _ES, _PT)

_PUNCT_RE = re.compile(r"[\u2010-\u2015\-\u00B7\u2212]+|[\.,;:!\?\(\)\[\]\{\}\'\"/\\]+")
_WS_RE = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    # NFKD + remove combining marks
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def canonicalize(text: str) -> str:
    """Canonical form for labels/queries across en/pt/es without large tokenizers.

    - Unicode NFKD + strip diacritics
    - Lowercase
    - Remove punctuation
    - Remove tiny union stopwords
    - Collapse whitespace
    """
    if not text:
        return ""
    t = _strip_diacritics(str(text)).lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    toks = [tok for tok in _WS_RE.split(t) if tok and tok not in STOPWORDS]
    return " ".join(toks)


def build_gazetteer(labels: Iterable[str]) -> Dict[str, List[str]]:
    """Build a canonical->labels mapping for quick lookup.

    If multiple labels collapse to the same canonical form, keep all.
    """
    gaz: Dict[str, List[str]] = {}
    for lab in labels:
        can = canonicalize(lab)
        if not can:
            continue
        gaz.setdefault(can, []).append(str(lab))
    return gaz


def match_gazetteer(query: str, gaz: Dict[str, List[str]]) -> Tuple[str | None, float | None]:
    """Attempt gazetteer match by canonical form, then by prefix/substring.

    Returns (label, score) where score=1.0 for exact canonical match, else 0.7/0.5 heuristics.
    """
    if not query:
        return None, None
    can = canonicalize(query)
    if not can:
        return None, None
    if can in gaz:
        # prefer the first inserted (stable for current dataset)
        return gaz[can][0], 1.0
    # prefix match over canonical keys
    for k in gaz.keys():
        if k.startswith(can):
            return gaz[k][0], 0.7
    for k in gaz.keys():
        if can in k:
            return gaz[k][0], 0.5
    return None, None

