"""Next 20 languages (Tier 3) grammar rules."""

from __future__ import annotations

from typing import Dict, List

from .grammar_generator import generate_language_rules

TIER_3_LANGUAGES: Dict[str, Dict] = {
    "he": {"name": "Hebrew", "pattern": "SVO"},
    "no": {"name": "Norwegian", "pattern": "SVO"},
    "sk": {"name": "Slovak", "pattern": "SVO"},
    "hr": {"name": "Croatian", "pattern": "SVO"},
    "lt": {"name": "Lithuanian", "pattern": "SVO"},
    "sl": {"name": "Slovenian", "pattern": "SVO"},
    "et": {"name": "Estonian", "pattern": "SVO"},
    "lv": {"name": "Latvian", "pattern": "SVO"},
    "sw": {"name": "Swahili", "pattern": "SVO"},
    "ta": {"name": "Tamil", "pattern": "SOV"},
    "te": {"name": "Telugu", "pattern": "SOV"},
    "mr": {"name": "Marathi", "pattern": "SOV"},
    "pa": {"name": "Punjabi", "pattern": "SOV"},
    "gu": {"name": "Gujarati", "pattern": "SOV"},
    "kn": {"name": "Kannada", "pattern": "SOV"},
    "ml": {"name": "Malayalam", "pattern": "SOV"},
    "si": {"name": "Sinhala", "pattern": "SOV"},
    "ne": {"name": "Nepali", "pattern": "SOV"},
    "my": {"name": "Burmese", "pattern": "SOV"},
    "km": {"name": "Khmer", "pattern": "SVO"},
}


def get_tier3_rules() -> List:
    rules = []
    for code, info in TIER_3_LANGUAGES.items():
        rules.extend(generate_language_rules(code, info))
    return rules


__all__ = ["get_tier3_rules", "TIER_3_LANGUAGES"]
