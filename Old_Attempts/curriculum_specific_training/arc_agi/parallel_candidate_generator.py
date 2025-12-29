"""
Parallel candidate generation with TRM swarm coordination.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator, Candidate
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, get_grammar_galaxy
from knowledge3d.training.arc_agi.trm_swarm_coordinator import TRMSwarmCoordinator

# Module-level grammar for worker reuse (initialized from snapshot)
_worker_grammar: Optional[GrammarGalaxy] = None


def _init_worker(grammar_snapshot: bytes) -> None:
    """Worker initializer — runs once per worker process."""
    global _worker_grammar
    if _worker_grammar is None:
        _worker_grammar = GrammarGalaxy(snapshot=grammar_snapshot)


def _worker_generate(
    task: Dict,
    specialist_id: Optional[str],
    confidence: float,
    grammar_snapshot: bytes,
    worker_id: int,
    n_workers: int,
) -> Tuple[Optional[str], List[Candidate], float, Dict]:
    """
    Worker function with specialist context.
    """
    global _worker_grammar
    if _worker_grammar is None:
        _init_worker(grammar_snapshot)

    semantic_hints = task.get("semantic_hints") or []
    worker_hints = semantic_hints[worker_id::n_workers] if semantic_hints else None

    gen = CandidateGenerator(
        matryoshka_dim=task.get("matryoshka_dim", 512),
        max_candidates=task.get("candidates_per_worker", 6),
    )
    candidates = gen.generate_candidates(
        input_grid=task["input_grid"],
        train_examples=task.get("train_examples", []),
        semantic_hints=worker_hints,
        expected_output=task.get("expected_output"),
    )

    discoveries = dict(getattr(_worker_grammar, "_local_discoveries", {}))
    return specialist_id, candidates, confidence, discoveries


class ParallelCandidateGenerator:
    """
    Parallel candidate generation with TRM swarm coordination.
    """

    def __init__(
        self,
        *,
        matryoshka_dim: int = 512,
        num_workers: int = 9,
        candidates_per_worker: int = 6,
    ) -> None:
        self.matryoshka_dim = matryoshka_dim
        self.num_workers = num_workers
        self.candidates_per_worker = candidates_per_worker
        self.swarm = TRMSwarmCoordinator(max_workers=num_workers)
        self.grammar = get_grammar_galaxy()

    def generate_parallel(
        self,
        input_grid: Sequence[Sequence[int]],
        train_examples: List[Dict[str, Any]],
        semantic_hints: Optional[List[str]],
        expected_output: Optional[Sequence[Sequence[int]]],
    ) -> List[Candidate]:
        """
        Generate candidates using dynamic worker spawning.
        """
        task: Dict[str, Any] = {
            "input_grid": input_grid,
            "train_examples": train_examples or [],
            "semantic_hints": semantic_hints or [],
            "expected_output": expected_output,
            "matryoshka_dim": self.matryoshka_dim,
            "candidates_per_worker": self.candidates_per_worker,
            "train": train_examples or [],
        }

        task_embedding = self.swarm.embed_task(task)
        task["task_embedding"] = task_embedding

        grammar_snapshot = self.grammar.to_snapshot()
        futures = self.swarm.spawn_workers(task, _worker_generate, grammar_snapshot)

        all_candidates: List[Candidate] = []
        for spec_id, future in futures:
            try:
                sid, candidates, confidence, discoveries = future.result()
                all_candidates.extend(candidates)
                self.grammar.merge_discoveries(discoveries)
            except Exception as exc:
                print(f"[WORKER ERROR] {spec_id}: {exc}")

        return all_candidates


__all__ = ["ParallelCandidateGenerator"]
