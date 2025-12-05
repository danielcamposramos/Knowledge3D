"""
Hybrid parallel+sequential candidate generator.

Combines K3D's parallel breadth (9 workers × 6 candidates) with TRM-style
sequential depth (3 workers × 21 refinements). Adaptive gating uses ternary
logic for the quick-solve threshold.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from knowledge3d.training.arc_agi.candidate_generator import Candidate
from knowledge3d.training.arc_agi.parallel_generator import ParallelCandidateGenerator
from knowledge3d.training.arc_agi.sequential_refiner import k3d_sequential_refine, k3d_sequential_refine_adaptive, _ternary_sign
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool, get_global_math_core_pool
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.training.arc_agi.sovereign_utils import grids_equal


class HybridCandidateGenerator:
    """Adaptive parallel+sequential generator with ternary gating."""

    def __init__(
        self,
        *,
        parallel_gen: ParallelCandidateGenerator,
        shadow_copy: DualShadowCopy,
        drawing_galaxy: DrawingGalaxy,
        grammar_galaxy: Optional[GrammarGalaxy] = None,
        core_pool: Optional[MathCorePool] = None,
        quick_solve_threshold: float = 0.95,
        embedding_galaxy: Optional[Dict[int, List[float]]] = None,
        cosine_bridge: Optional[CosineSimilarityBridge] = None,
        refiner_fn=k3d_sequential_refine_adaptive,
    ) -> None:
        self.parallel_gen = parallel_gen
        self.shadow_copy = shadow_copy
        self.drawing_galaxy = drawing_galaxy
        self.grammar_galaxy = grammar_galaxy
        self.core_pool = core_pool or get_global_math_core_pool()
        self.quick_solve_threshold = quick_solve_threshold
        self.embedding_galaxy = embedding_galaxy
        self.cosine_bridge = cosine_bridge or CosineSimilarityBridge()
        self.refiner_fn = refiner_fn

    def generate_hybrid(
        self,
        input_grid: Sequence[Sequence[int]],
        train_examples: List[Dict[str, Any]],
        semantic_hints: Optional[List[str]],
        expected_output: Optional[Sequence[Sequence[int]]],
        task_history: Optional[List[float]] = None,
        task_complexity: float = 0.0,
    ) -> List[Candidate]:
        # Phase 1: quick parallel breadth
        quick_candidates = self.parallel_gen.generate_parallel(
            input_grid=input_grid,
            train_examples=train_examples,
            semantic_hints=semantic_hints,
            expected_output=expected_output,
        )
        print(f"  [HYBRID] Quick parallel generated {len(quick_candidates)} candidates")

        # Ternary routing decision: skip / partial / full
        routing = "activate_full"
        if expected_output is not None and quick_candidates:
            best_quick = quick_candidates[0]
            quick_score = _evaluate_candidate_accuracy(best_quick[0], expected_output)
            shadow_conf = getattr(self.shadow_copy, "get_average_confidence", lambda: 0.5)()
            routing = adaptive_routing_ternary(
                quick_score=quick_score,
                task_history=task_history or [],
                shadow_confidence=shadow_conf,
                task_complexity=task_complexity,
                quick_threshold=self.quick_solve_threshold,
            )
        else:
            routing = "activate_partial"  # conservative when no ground truth

        # Seeds for deep workers
        if routing == "skip_deep":
            print("  [ROUTING] Ternary decision: SKIP (quick solved)")
            return quick_candidates

        seeds = quick_candidates[:3] if len(quick_candidates) >= 3 else quick_candidates
        if not seeds:
            return quick_candidates

        n, T = (3, 2) if routing == "activate_partial" else (6, 3)
        print(f"  [ROUTING] Ternary decision: {routing.upper()} (n={n}, T={T})")

        deep_candidates: List[Candidate] = []
        for idx, seed in enumerate(seeds):
            refined_grid, applied_patterns = self.refiner_fn(
                input_grid=input_grid,
                initial_candidate=seed,
                shadow_copy=self.shadow_copy,
                drawing_galaxy=self.drawing_galaxy,
                grammar_galaxy=self.grammar_galaxy,
                core_pool=self.core_pool,
                n=n,
                T=T,
            )
            instruction = f"[DEEP REFINEMENT {idx}] {len(applied_patterns)} patterns applied"
            program = " | ".join(applied_patterns[:3]) if applied_patterns else ""
            deep_candidates.append((refined_grid, instruction, program))
        print(f"  [HYBRID] Deep refinement generated {len(deep_candidates)} candidates")

        combined = quick_candidates + deep_candidates
        deduped = _deduplicate_candidates(combined)

        if expected_output is not None and self.embedding_galaxy:
            deduped = _rank_by_similarity_hybrid(
                candidates=deduped,
                expected_output=expected_output,
                embedding_galaxy=self.embedding_galaxy,
                cosine_bridge=self.cosine_bridge,
            )
        print(f"  [HYBRID] Returning {len(deduped)} unique candidates after dedup+ranking")
        return deduped


def _deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    seen: Set[Tuple[Tuple[int, ...], ...]] = set()
    deduped: List[Candidate] = []
    for grid, instr, prog in candidates:
        key = tuple(tuple(row) for row in grid)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((grid, instr, prog))
    return deduped


def _evaluate_candidate_accuracy(
    candidate_grid: Sequence[Sequence[int]],
    expected_grid: Sequence[Sequence[int]],
) -> float:
    if not grids_equal(candidate_grid, expected_grid):
        # fall back to overlap measure
        if len(candidate_grid) != len(expected_grid):
            return 0.0
        total = 0
        matches = 0
        for row_c, row_e in zip(candidate_grid, expected_grid):
            if len(row_c) != len(row_e):
                return 0.0
            for a, b in zip(row_c, row_e):
                total += 1
                if a == b:
                    matches += 1
        return matches / total if total else 0.0
    return 1.0


def _rank_by_similarity_hybrid(
    candidates: List[Candidate],
    expected_output: Sequence[Sequence[int]],
    embedding_galaxy: Dict[int, List[float]],
    cosine_bridge: CosineSimilarityBridge,
) -> List[Candidate]:
    if not candidates:
        return candidates
    expected_hash = hash(tuple(tuple(row) for row in expected_output))
    expected_emb = embedding_galaxy.get(expected_hash)
    if expected_emb is None:
        print(f"  [RANKING] Expected output embedding not in Galaxy (hash={expected_hash})")
        return candidates

    emb_list: List[List[float]] = []
    valid_candidates: List[Candidate] = []
    for cand in candidates:
        grid_hash = hash(tuple(tuple(row) for row in cand[0]))
        emb = embedding_galaxy.get(grid_hash)
        if emb is None:
            continue
        emb_list.append(emb)
        valid_candidates.append(cand)

    if not emb_list:
        print("  [RANKING] No candidate embeddings in Galaxy, returning unranked")
        return candidates

    scores = cosine_bridge.compute_similarities(emb_list, expected_emb)
    scored = list(zip(scores, valid_candidates))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]


def adaptive_routing_ternary(
    quick_score: float,
    task_history: List[float],
    shadow_confidence: float,
    task_complexity: float,
    quick_threshold: float = 0.95,
) -> str:
    """
    Ternary routing heuristic: returns skip_deep | activate_partial | activate_full.
    """
    # Fast-path skip when quick solution meets or exceeds threshold
    if quick_score >= quick_threshold:
        return "skip_deep"

    score_signal = _ternary_sign(quick_score - quick_threshold)
    plateau_signal = 0
    history_tail = task_history[-3:] if task_history else []
    if len(history_tail) >= 2:
        deltas = [b - a for a, b in zip(history_tail[:-1], history_tail[1:])]
        plateau_signal = _ternary_sign(sum(deltas))
    confidence_signal = _ternary_sign(shadow_confidence - 0.75)
    complexity_signal = _ternary_sign(task_complexity - 0.70)

    # Strong negative signals → force full depth
    if score_signal < 0 and confidence_signal < 0:
        return "activate_full"

    aggregate = score_signal + plateau_signal + confidence_signal + complexity_signal
    if aggregate >= 2:
        return "skip_deep"
    if aggregate <= -2:
        return "activate_full"
    return "activate_partial"


__all__ = ["HybridCandidateGenerator", "adaptive_routing_ternary"]
