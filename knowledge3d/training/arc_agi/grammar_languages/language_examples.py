"""Placeholder example provider for language grammars."""

from __future__ import annotations

from typing import Dict, List


def get_language_examples(lang_code: str, pattern: str) -> List[Dict[str, str]]:
    """Return per-language examples (placeholder)."""
    if pattern == "SVO":
        return [{"subject": f"{lang_code}_subject", "verb": f"{lang_code}_verb", "object": f"{lang_code}_object"}]
    if pattern == "SOV":
        return [{"subject": f"{lang_code}_subject", "object": f"{lang_code}_object", "verb": f"{lang_code}_verb"}]
    return [{"subject": f"{lang_code}_subject", "verb": f"{lang_code}_verb"}]


def get_question_examples(lang_code: str) -> List[Dict[str, str]]:
    return [{"auxiliary": f"{lang_code}_aux", "subject": f"{lang_code}_you", "verb": f"{lang_code}_like", "object": f"{lang_code}_object"}]


def get_imperative_examples(lang_code: str) -> List[Dict[str, str]]:
    return [{"verb": f"{lang_code}_do", "object": f"{lang_code}_task"}]
