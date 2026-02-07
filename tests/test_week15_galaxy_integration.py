from __future__ import annotations

from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.drawing_galaxy import DrawingGalaxy
from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy, GrammarRule


def test_knowledgeverse_loads_drawing_galaxy(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    assert isinstance(drawing, DrawingGalaxy)
    assert drawing.name == "Drawing"
    assert len(drawing.entries) > 0
    assert len(drawing.transformations) >= 10
    summary = drawing.summary()
    assert summary["shapes"] > 0
    assert summary["transformations"] > 0


def test_knowledgeverse_loads_grammar_galaxy(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    grammar = kv.galaxy_manager.get_galaxy("Grammar")

    assert isinstance(grammar, GrammarGalaxy)
    assert grammar.name == "Grammar"
    assert len(grammar.entries) > 0
    assert len(grammar.rules) >= 190
    summary = grammar.summary()
    assert summary["canonical_rules"] > 0


def test_drawing_discovery_api(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    initial_count = len(drawing.shapes)
    drawing.add_shape("TEST_SHAPE", "CIRCLE 0.5 0.5 0.2", source={"task": "test"})

    assert len(drawing.shapes) == initial_count + 1
    assert "TEST_SHAPE" in drawing.shapes


def test_grammar_discovery_api(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    grammar = kv.galaxy_manager.get_galaxy("Grammar")

    initial_count = len(grammar.rules)
    test_rule = GrammarRule(
        rule_id="TEST_RULE",
        language="en",
        pattern="test",
        rpn_program="TEST_OP",
    )
    assert grammar.add_rule(test_rule, persist=False) is True
    assert len(grammar.rules) == initial_count + 1
    assert "TEST_RULE" in grammar.rules


def test_galaxy_singleton_pattern(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing_1 = kv.galaxy_manager.get_galaxy("Drawing")
    drawing_2 = kv.galaxy_manager.get_galaxy("Drawing")

    assert drawing_1 is drawing_2


def test_shadow_copy_event_logging_from_discovery(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv")
    drawing = kv.galaxy_manager.get_galaxy("Drawing")
    grammar = kv.galaxy_manager.get_galaxy("Grammar")

    before = len(kv.shadow_copy.event_buffer)
    drawing.add_shape("EVENT_SHAPE", "RECT 0.0 0.0 1.0 1.0")
    grammar.add_rule(
        GrammarRule(
            rule_id="EVENT_RULE",
            language="en",
            pattern="event",
            rpn_program="EVENT_OP",
        ),
        persist=False,
    )
    after = len(kv.shadow_copy.event_buffer)

    assert after >= before + 2
    recent_types = [event["type"] for event in kv.shadow_copy.event_buffer[-4:]]
    assert "drawing_discovery" in recent_types
    assert "grammar_discovery" in recent_types

