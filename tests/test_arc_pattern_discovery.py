from __future__ import annotations

from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy
from knowledge3d.knowledgeverse.trm_navigator import TRMNavigator


def test_arc_bootstrap_rules_present():
    grammar = GrammarGalaxy()
    rule_ids = {rule.rule_id for rule in grammar.list_arc_rules()}
    assert "arc_flip_h" in rule_ids
    assert "arc_rot90_cw" in rule_ids
    assert "arc_color_map" in rule_ids
    assert "arc_rotate_then_color" in rule_ids


def test_propose_arc_transform_flip_horizontal():
    grammar = GrammarGalaxy()
    examples = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[2, 1], [4, 3]],
        },
        {
            "input": [[5, 6], [7, 8]],
            "output": [[6, 5], [8, 7]],
        },
    ]
    proposal = grammar.propose_arc_transform(examples)
    assert proposal["rule_id"] == "arc_flip_h"
    assert proposal["transform"]["op"] == "flip_h"
    assert proposal["confidence"] >= 0.95


def test_propose_arc_transform_composed_rotate_then_color():
    grammar = GrammarGalaxy()
    examples = [
        {
            "input": [[1, 1], [2, 2]],
            "output": [[4, 3], [4, 3]],
        }
    ]
    proposal = grammar.propose_arc_transform(examples)
    assert proposal["rule_id"] == "arc_rotate_then_color"
    assert proposal["transform"]["op"] == "composed"
    steps = proposal["transform"]["steps"]
    assert steps[0]["op"] == "rot90"
    assert steps[1]["op"] == "color_map"
    assert "mapping" in steps[1]
    assert proposal["confidence"] >= 0.95


def test_trm_navigator_prefers_grammar_transform_trace():
    navigator = TRMNavigator()
    examples = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[2, 1], [4, 3]],
        }
    ]
    transform = navigator._infer_arc_transform(examples, prefer_enriched=True)
    trace = "\n".join(navigator.get_reasoning_trace())
    assert transform["op"] == "flip_h"
    assert "source=grammar" in trace
