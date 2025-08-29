from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


BLOCK_KEYWORDS = {
    # harmless starter; expand over time and per-language
    "en": ["kill", "harm", "dox", "doxx", "bomb", "exploit", "hack"],
    "pt": ["matar", "ferir", "dox", "bomba", "invadir", "hackear"],
    "es": ["matar", "herir", "dox", "bomba", "hackear", "invadir"],
}


@dataclass
class EthicsDecision:
    allow: bool
    reason: Optional[str] = None


def detect_lang(text: str) -> str:
    # naive heuristic
    t = text.lower()
    if any(w in t for w in (" você", " você", " para ", " com ")):
        return "pt"
    if any(w in t for w in (" usted", " para ", " con ")):
        return "es"
    return "en"


def check_request(text: str, action: Optional[str]) -> EthicsDecision:
    lang = detect_lang(text)
    for kw in BLOCK_KEYWORDS.get(lang, []) + BLOCK_KEYWORDS.get("en", []):
        if kw in text:
            return EthicsDecision(allow=False, reason=f"blocked keyword '{kw}'")
    return EthicsDecision(allow=True)

