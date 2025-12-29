"""Helpers to procedurally generate language grammar rules."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule


def _example_sentence(lang_code: str, pattern: str) -> List[Dict[str, str]]:
    """Generate a minimal example for a language/pattern."""
    if pattern == "SVO":
        return [{"subject": f"{lang_code}_subject", "verb": f"{lang_code}_verb", "object": f"{lang_code}_object"}]
    if pattern == "SOV":
        return [{"subject": f"{lang_code}_subject", "object": f"{lang_code}_object", "verb": f"{lang_code}_verb"}]
    if pattern == "VSO":
        return [{"subject": f"{lang_code}_subject", "verb": f"{lang_code}_verb", "object": f"{lang_code}_object"}]
    return [{"subject": f"{lang_code}_subject", "verb": f"{lang_code}_verb"}]


def _question_examples(lang_code: str) -> List[Dict[str, str]]:
    return [
        {"auxiliary": f"{lang_code}_aux", "subject": f"{lang_code}_you", "verb": f"{lang_code}_like", "object": f"{lang_code}_math"}
    ]


def _imperative_examples(lang_code: str) -> List[Dict[str, str]]:
    return [
        {"verb": f"{lang_code}_do", "object": f"{lang_code}_task"},
        {"verb": f"{lang_code}_open", "object": f"{lang_code}_door"},
    ]


def generate_language_rules(lang_code: str, lang_info: Dict):
    """Generate a small grammar set for a language using its canonical word order."""
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule
    pattern = lang_info.get("pattern", "SVO")
    rules: List["GrammarRule"] = []

    # Simple sentence
    if pattern == "SVO":
        rpn = "SUBJECT RECALL VERB RECALL OBJECT RECALL SVO_ORDER CONCAT_SENTENCE"
    elif pattern == "SOV":
        rpn = "SUBJECT RECALL OBJECT RECALL WO_PARTICLE RECALL VERB RECALL SOV_ORDER CONCAT_SENTENCE"
    elif pattern == "VSO":
        rpn = "VERB RECALL SUBJECT RECALL OBJECT RECALL VSO_ORDER CONCAT_SENTENCE"
    else:
        rpn = "SUBJECT RECALL VERB RECALL OBJECT RECALL CONCAT_SENTENCE"

    rules.append(
        GrammarRule(
            rule_id=f"{lang_code}_simple_sentence",
            language=lang_code,
            domain="text",
            pattern=pattern,
            rpn_program=rpn,
            examples=_example_sentence(lang_code, pattern),
            description=f"{lang_info.get('name', lang_code)} simple sentence",
        )
    )

    # Question
    rules.append(
        GrammarRule(
            rule_id=f"{lang_code}_question",
            language=lang_code,
            domain="text",
            pattern="Q",
            rpn_program="AUXILIARY RECALL SUBJECT RECALL VERB RECALL OBJECT RECALL CONCAT_SENTENCE",
            examples=_question_examples(lang_code),
            description=f"{lang_info.get('name', lang_code)} question",
        )
    )

    # Imperative
    rules.append(
        GrammarRule(
            rule_id=f"{lang_code}_imperative",
            language=lang_code,
            domain="text",
            pattern="V_O",
            rpn_program="VERB RECALL OBJECT RECALL CONCAT_SENTENCE",
            examples=_imperative_examples(lang_code),
            description=f"{lang_info.get('name', lang_code)} imperative",
        )
    )

    return rules
