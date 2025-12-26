from __future__ import annotations


class _DeterministicTRM:
    def rank_rules(self, matches, _problem_text: str):
        return sorted(matches, key=lambda m: (-m.score, getattr(m.rule, "rule_id", "")))

    def validate_result(self, _result, _problem_text: str) -> float:
        return 0.9

    def embed(self, text: str):
        return text

    def compose_from_symbols(self, symbols, _problem_text: str) -> str:
        for sym in symbols:
            tmpl = getattr(sym, "rpn_template", "") or ""
            if tmpl.strip():
                return tmpl
        return ""


class _EchoEngine:
    def evaluate(self, expression: str, *_args, **_kwargs):
        return expression


def test_trm_navigator_records_to_shadow_copy():
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi import math_grammar_rules
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(drawing, grammar, staged=True)

    nav = TRMMathNavigator(
        rule_bank=math_grammar_rules.SOVEREIGN_MATH_RULES,
        math_galaxy=MATH_GALAXY,
        rpn_engine=_EchoEngine(),
        trm_engine=_DeterministicTRM(),
        shadow_copy=shadow,
    )

    result, meta = nav.solve("\\frac{24}{4}")
    assert result == "24 4 /"
    assert meta["rule_used"] == "latex_frac"

    assert len(shadow.library) >= 1
    assert shadow.library[0]["program_type"] == "math"


def test_pattern_confidence_updated():
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi import math_grammar_rules
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator

    shadow = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)
    nav = TRMMathNavigator(
        rule_bank=math_grammar_rules.SOVEREIGN_MATH_RULES,
        math_galaxy=MATH_GALAXY,
        rpn_engine=_EchoEngine(),
        trm_engine=_DeterministicTRM(),
        shadow_copy=shadow,
    )

    nav.solve("\\frac{10}{2}")
    nav.solve("\\frac{20}{4}")
    nav.solve("\\frac{30}{6}")

    conf = shadow.get_pattern_success_rate("latex_frac")
    assert conf is not None
    assert conf > 0.5


def test_shadow_copy_persists(tmp_path):
    from pathlib import Path

    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

    shadow = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)
    shadow.record(
        task_signature={"problem_text": "x", "rule_id": "r", "result": "1", "problem_type": "math_arithmetic"},
        program="24 4 /",
        program_type="math",
        score=0.9,
        task_id="math_1",
    )
    shadow.commit_pending()

    ckpt = Path(tmp_path) / "shadow_copy.json"
    shadow.save(ckpt)

    shadow2 = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)
    shadow2.load(ckpt)
    assert len(shadow2.library) == len(shadow.library)


def test_shadow_copy_query_by_patterns_prefers_overlap(tmp_path):
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

    shadow = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)

    shadow.record(
        task_signature={"problem_text": "a", "problem_type": "gsm8k"},
        program="15 3 -",
        program_type="composition",
        score=0.95,
        semantic_context={
            "template_used": "extract_operate_aggregate",
            "patterns_matched": ["galaxy_has_quantity", "galaxy_gave_to"],
            "composition_steps": [{"step": 1}],
        },
    )
    shadow.record(
        task_signature={"problem_text": "b", "problem_type": "gsm8k"},
        program="15 3 -",
        program_type="composition",
        score=0.95,
        semantic_context={
            "template_used": "simple_apply",
            "patterns_matched": ["galaxy_has_quantity"],
            "composition_steps": [{"step": 1}],
        },
    )

    hit = shadow.query_by_patterns(frozenset({"galaxy_has_quantity", "galaxy_gave_to"}))
    assert hit is not None
    ctx = hit.get("semantic_context", {})
    assert ctx.get("template_used") == "extract_operate_aggregate"


def test_shadow_copy_allows_multiple_composition_entries_same_program(tmp_path):
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

    shadow = DualShadowCopy(DrawingGalaxy(), GrammarGalaxy(), staged=True)
    shadow.record(
        task_signature={"problem_text": "a", "problem_type": "gsm8k"},
        program="15 3 -",
        program_type="composition",
        score=0.95,
        semantic_context={"template_used": "extract_operate_aggregate", "patterns_matched": ["p1"], "composition_steps": []},
    )
    shadow.record(
        task_signature={"problem_text": "b", "problem_type": "gsm8k"},
        program="15 3 -",
        program_type="composition",
        score=0.95,
        semantic_context={"template_used": "extract_operate_aggregate", "patterns_matched": ["p2"], "composition_steps": []},
    )
    # Same RPN program, but different pattern signatures → should both be recorded.
    assert len([e for e in shadow.library if e.get("program_type") == "composition"]) == 2


def test_trm_galaxy_reader_emits_retrieval_selection_meta(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    wg = WordGalaxy(storage_path=tmp_path)
    gg = GrammarGalaxy()
    shadow = DualShadowCopy(DrawingGalaxy(), gg, staged=True)

    # Seed a composition entry so retrieval can hit.
    shadow.record(
        task_signature={"problem_text": "seed", "problem_type": "gsm8k"},
        program="48 48 2 / +",
        program_type="composition",
        score=0.95,
        semantic_context={
            "template_used": "extract_operate_aggregate",
            "patterns_matched": ["galaxy_sold_to_quantity", "galaxy_half_as_many", "galaxy_altogether"],
            "composition_steps": [],
        },
    )

    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=shadow,
    )

    text = "Natalia sold clips to 48 friends. She sold half as many in May. How many altogether?"
    u, t = reader.read_problem(text)
    assert u.is_complete()
    _ = reader.compose_rpn(u, trace=t)
    meta = reader.get_last_composition_meta()
    assert meta.get("template_used")
    assert meta.get("heuristic_template")
    assert meta.get("template_selected_by") in {"retrieval", "heuristic"}


def test_math_discoveries_commit_into_grammar_galaxy():
    from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
    from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(DrawingGalaxy(), grammar, staged=True)
    shadow.record(
        task_signature={"problem_text": "x", "rule_id": "r", "result": "1", "problem_type": "math_arithmetic"},
        program="24 4 /",
        program_type="math",
        score=0.9,
        task_id="math_2",
    )
    shadow.commit_pending()

    assert any(rule_id.startswith("DISCOVERED_MATH_RULE_") for rule_id in grammar.rules)
