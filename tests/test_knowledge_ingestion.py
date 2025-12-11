import pytest

from knowledge3d.cranium.word_galaxy import WordGalaxy, WordDefinition
from knowledge3d.cranium.eloquence_galaxy import EloquenceGalaxy, MetaRule
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy


def test_word_galaxy_symlink_validation():
    wg = WordGalaxy(storage_path=WordGalaxy().storage_path)  # reuse default path
    # Valid: ASCII + known math symbol ∂ (8706)
    word = WordDefinition(
        word_id="derivative",
        char_sequence=[ord(c) for c in "derivative"] + [8706],
        definition="Rate of change",
        domain="math_calculus",
        related_symbols=[8706],
    )
    assert word.validate_char_sequence()

    # Invalid: unknown high codepoint should fail
    bad = WordDefinition(
        word_id="bad_word",
        char_sequence=[999999],
        definition="invalid",
        domain="test",
    )
    assert bad.validate_char_sequence() is False


def test_eloquence_meta_rule_refs():
    grammar = GrammarGalaxy()
    assert grammar.has_rule("en_simple_sentence")

    meta = MetaRule(
        meta_id="test_meta",
        category="eloquence",
        condition="1 1 eq",
        action="noop",
        rule_refs=["en_simple_sentence"],
    )
    assert meta.validate_rule_refs()

    meta_bad = MetaRule(
        meta_id="test_meta_bad",
        category="eloquence",
        condition="1 1 eq",
        action="noop",
        rule_refs=["nonexistent_rule_xyz"],
    )
    assert meta_bad.validate_rule_refs() is False
