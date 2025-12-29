"""Top 10 languages grammar rules."""

from __future__ import annotations

from typing import Dict, List

from .grammar_generator import generate_language_rules

TOP_10_LANGUAGES: Dict[str, Dict] = {
    "zh": {"name": "Chinese (Mandarin)", "pattern": "SVO"},
    "es": {"name": "Spanish", "pattern": "SVO"},
    "en": {"name": "English", "pattern": "SVO"},
    "hi": {"name": "Hindi", "pattern": "SOV"},
    "ar": {"name": "Arabic", "pattern": "VSO"},
    "pt": {"name": "Portuguese", "pattern": "SVO"},
    "bn": {"name": "Bengali", "pattern": "SOV"},
    "ru": {"name": "Russian", "pattern": "SVO"},
    "ja": {"name": "Japanese", "pattern": "SOV"},
    "de": {"name": "German", "pattern": "SVO"},
}


def get_tier1_rules() -> List:
    rules = []
    for code, info in TOP_10_LANGUAGES.items():
        rules.extend(generate_language_rules(code, info))
    return rules


__all__ = ["get_tier1_rules", "TOP_10_LANGUAGES"]
