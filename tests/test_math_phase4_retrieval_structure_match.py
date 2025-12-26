from __future__ import annotations

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.math_benchmarks.trm_galaxy_reader import (
    ProblemUnderstanding,
    TRMGalaxyReader,
)


def test_shadow_copy_structure_match_required() -> None:
    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(drawing, grammar, staged=True)

    shadow.record(
        task_signature={"task": "t"},
        program="15 3 -",
        program_type="composition",
        score=1.0,
        semantic_context={
            "template_used": "extract_operate_aggregate",
            "patterns_matched": ["galaxy_has_quantity", "galaxy_gave_to"],
            "structure": {
                "n_quantities": 1,
                "n_operations": 1,
                "aggregation": "",
                "multi_step_indicator": False,
                "has_rate": False,
                "has_duration": False,
            },
        },
    )

    entry, _score = shadow.query_by_patterns_scored(
        frozenset({"galaxy_has_quantity", "galaxy_gave_to"}),
        structure={
            "n_quantities": 1,
            "n_operations": 3,
            "aggregation": "",
            "multi_step_indicator": True,
            "has_rate": False,
            "has_duration": False,
        },
    )
    assert entry is None


def test_reader_can_disable_retrieval() -> None:
    class _Shadow:
        def query_by_patterns_scored(self, *_args, **_kwargs):
            raise AssertionError("retrieval should be disabled")

    reader = TRMGalaxyReader(
        word_galaxy=object(),
        grammar_galaxy=object(),
        math_galaxy=object(),
        rule_bank=[],
        shadow_copy=_Shadow(),
        use_retrieval=False,
    )
    understanding = ProblemUnderstanding(quantities=[{"value": 15.0, "source": "galaxy_has_quantity"}])
    rpn = reader.compose_rpn(
        understanding,
        trace={"patterns": [{"rule_id": "galaxy_has_quantity"}]},
        problem_text="John has 15 apples.",
    )
    assert rpn

