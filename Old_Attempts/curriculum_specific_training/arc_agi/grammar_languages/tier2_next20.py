"""Next 20 languages (Tier 2) grammar rules."""

from __future__ import annotations

from typing import Dict, List

from .grammar_generator import generate_language_rules

TIER_2_LANGUAGES: Dict[str, Dict] = {
    "fr": {"name": "French", "pattern": "SVO"},
    "ur": {"name": "Urdu", "pattern": "SOV"},
    "id": {"name": "Indonesian", "pattern": "SVO"},
    "it": {"name": "Italian", "pattern": "SVO"},
    "tr": {"name": "Turkish", "pattern": "SOV"},
    "vi": {"name": "Vietnamese", "pattern": "SVO"},
    "ko": {"name": "Korean", "pattern": "SOV"},
    "fa": {"name": "Persian", "pattern": "SOV"},
    "pl": {"name": "Polish", "pattern": "SVO"},
    "uk": {"name": "Ukrainian", "pattern": "SVO"},
    "th": {"name": "Thai", "pattern": "SVO"},
    "ro": {"name": "Romanian", "pattern": "SVO"},
    "nl": {"name": "Dutch", "pattern": "SVO"},
    "el": {"name": "Greek", "pattern": "SVO"},
    "hu": {"name": "Hungarian", "pattern": "SVO"},
    "cs": {"name": "Czech", "pattern": "SVO"},
    "sv": {"name": "Swedish", "pattern": "SVO"},
    "bg": {"name": "Bulgarian", "pattern": "SVO"},
    "da": {"name": "Danish", "pattern": "SVO"},
    "fi": {"name": "Finnish", "pattern": "SVO"},
}


def get_tier2_rules() -> List:
    rules = []
    for code, info in TIER_2_LANGUAGES.items():
        rules.extend(generate_language_rules(code, info))
    return rules


__all__ = ["get_tier2_rules", "TIER_2_LANGUAGES"]
