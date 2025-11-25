from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.sovereign_trm_router import SovereignTRMRouter
from knowledge3d.training.arc_agi.program_composer import ProgramComposer
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline


def test_grid_to_drawing_rpn_and_routing():
    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    router = SovereignTRMRouter(drawing, grammar, matryoshka_dim=128)

    grid = [[1, 0], [0, 2]]
    rpn = router.grid_to_drawing_rpn(grid)
    assert "GRID 2 2" in rpn
    assert "CELL 0 0 1 FILL" in rpn
    assert "CELL 1 1 2 FILL" in rpn

    candidates = router.route(grid, top_k=2)
    assert candidates
    assert candidates[0].drawing_program.startswith("GRID")


def test_dual_shadow_copy_records_growth():
    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(drawing, grammar)
    start_shapes = len(drawing.shapes)
    start_rules = len(grammar.rules)

    shadow.record({"grid_rows": 1}, "GRID 1 1 CELL 0 0 1 FILL", "visual", 0.95)
    summary = shadow.summary()

    assert summary["entries"] == 1
    assert len(drawing.shapes) == start_shapes + 1
    assert len(grammar.rules) == start_rules  # visual should not add grammar rule here


def test_program_composer_classifies():
    composer = ProgramComposer()
    programs = composer.compose("GRID 1 1", [])
    assert programs == []
    assert composer.classify("GRID 1 1 CELL 0 0 1 FILL") == "visual"
    assert composer.classify("rotate 90") in {"transformation", "hybrid"}


def test_pipeline_merges_procedural_and_trm_candidates():
    pipeline = SovereignAIPipeline(matryoshka_dim=128)
    train_examples = [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}]
    result = pipeline.process_task(
        "task_merge",
        test_input=[[1, 0], [0, 0]],
        train_examples=train_examples,
        expected_output=[[0, 1], [0, 0]],
        top_k=1,
    )
    assert result.best_program
    assert result.output_grid is not None
    assert result.score >= 0.0
