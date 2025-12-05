import pytest
from typing import List

from knowledge3d.training.arc_agi.hybrid_generator import HybridCandidateGenerator, adaptive_routing_ternary
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.candidate_generator import Candidate
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy


class FakeParallel:
    def __init__(self, candidates):
        self._candidates = candidates

    def generate_parallel(self, *, input_grid, train_examples, semantic_hints, expected_output):
        return list(self._candidates)


class FakeCorePool:
    def __init__(self):
        self.spawned = 0
        self.released = 0

    def spawn_core(self, tier=1, reuse=True):
        self.spawned += 1
        return self.spawned

    def release_core(self, core_id, pool=True):
        self.released += 1


class DummyExecutor:
    def __init__(self, delta: int = 1):
        self.delta = delta

    def execute(self, grid, program):
        # Return a grid with the first cell modified to ensure improvement detection
        if not grid or not grid[0]:
            return [[self.delta]]
        new_grid = [list(row) for row in grid]
        new_grid[0][0] = new_grid[0][0] + self.delta
        return new_grid


def make_shadow(library=None):
    drawing = DrawingGalaxy()
    grammar = GrammarGalaxy()
    shadow = DualShadowCopy(drawing, grammar, staged=False)
    shadow.library = library or []
    return shadow


def test_hybrid_quick_path_skips_deep():
    quick: List[Candidate] = [([[1]], "instr", "prog")]
    shadow = make_shadow([])
    core_pool = FakeCorePool()
    drawing = DrawingGalaxy()

    def refiner_fn(**kwargs):
        raise AssertionError("refiner should not be called for quick solve")

    hybrid = HybridCandidateGenerator(
        parallel_gen=FakeParallel(quick),
        shadow_copy=shadow,
        drawing_galaxy=drawing,
        core_pool=core_pool,
        quick_solve_threshold=0.95,
        refiner_fn=refiner_fn,
    )

    result = hybrid.generate_hybrid(
        input_grid=[[1]],
        train_examples=[],
        semantic_hints=None,
        expected_output=[[1]],
    )

    assert result == quick


def test_hybrid_triggers_deep_and_appends_candidates():
    quick: List[Candidate] = [([[0]], "instr", "prog0")]
    shadow_library = [
        {"program": "p1", "quality_score": 0.9},
    ]
    shadow = make_shadow(shadow_library)
    core_pool = FakeCorePool()
    drawing = DrawingGalaxy()
    called = {}

    def refiner_fn(**kwargs):
        called["seed"] = kwargs.get("initial_candidate")
        return [[9]], ["p1"]

    hybrid = HybridCandidateGenerator(
        parallel_gen=FakeParallel(quick),
        shadow_copy=shadow,
        drawing_galaxy=drawing,
        core_pool=core_pool,
        quick_solve_threshold=0.95,
        refiner_fn=refiner_fn,
    )

    result = hybrid.generate_hybrid(
        input_grid=[[0]],
        train_examples=[],
        semantic_hints=None,
        expected_output=[[1]],
    )

    assert len(result) == 2  # quick + deep
    assert called["seed"] == quick[0]
    assert result[1][0] == [[9]]
    assert "DEEP REFINEMENT" in result[1][1]


def test_routing_skip_when_quick_high():
    decision = adaptive_routing_ternary(
        quick_score=0.99,
        task_history=[],
        shadow_confidence=0.4,
        task_complexity=0.2,
        quick_threshold=0.95,
    )
    assert decision == "skip_deep"


def test_routing_partial_for_mixed_signals():
    decision = adaptive_routing_ternary(
        quick_score=0.90,
        task_history=[0.8, 0.82, 0.81],
        shadow_confidence=0.8,
        task_complexity=0.6,
        quick_threshold=0.95,
    )
    assert decision == "activate_partial"


def test_routing_full_when_low_score_and_hard():
    decision = adaptive_routing_ternary(
        quick_score=0.6,
        task_history=[0.5, 0.52, 0.5],
        shadow_confidence=0.6,
        task_complexity=0.9,
        quick_threshold=0.9,
    )
    assert decision == "activate_full"
