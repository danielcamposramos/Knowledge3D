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
        embedding_galaxy=task.get("embedding_galaxy") or {},
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
        top_k: int = 3,
        shadow_copy: Any | None = None,
        drawing_galaxy: Any | None = None,
        codec_embedder: Any | None = None,
        embedding_galaxy: dict[int, list[float]] | None = None,
        cosine_bridge: Any | None = None,
        knowledgeverse: Any | None = None,
        use_process_workers: bool | None = None,
    ) -> None:
        self.matryoshka_dim = matryoshka_dim
        self.num_workers = num_workers
        self.candidates_per_worker = candidates_per_worker
        self.top_k = top_k
        self.shadow_copy = shadow_copy
        self.codec_embedder = codec_embedder
        self.embedding_galaxy = embedding_galaxy if embedding_galaxy is not None else {}
        self.cosine_bridge = cosine_bridge
        self.knowledgeverse = knowledgeverse
        self.use_process_workers = (
            bool(use_process_workers)
            if use_process_workers is not None
            else (self.knowledgeverse is None)
        )
        self.core_pool = None
        self.swarm = TRMSwarmCoordinator(max_workers=num_workers)
        if self.knowledgeverse is not None:
            self.grammar = self.knowledgeverse.galaxy_manager.get_galaxy("Grammar")
            self.drawing_galaxy = drawing_galaxy or self.knowledgeverse.galaxy_manager.get_galaxy("Drawing")
        else:
            self.grammar = get_grammar_galaxy()
            self.drawing_galaxy = drawing_galaxy

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
            "embedding_galaxy": self.embedding_galaxy,
            "train": train_examples or [],
        }

        task_embedding = self.swarm.embed_task(task)
        task["task_embedding"] = task_embedding

        all_candidates: List[Candidate] = []
        if self.use_process_workers:
            grammar_snapshot = self.grammar.to_snapshot()
            futures = self.swarm.spawn_workers(task, _worker_generate, grammar_snapshot)
            for spec_id, future in futures:
                try:
                    _, candidates, _, discoveries = future.result()
                    all_candidates.extend(candidates)
                    self.grammar.merge_discoveries(discoveries)
                except Exception as exc:
                    print(f"[WORKER ERROR] {spec_id}: {exc}")
        else:
            routing = self.swarm.route_task(task_embedding)
            if not routing:
                routing = [(None, 0.0)]
            worker_count = len(routing)
            for worker_id, (spec_id, confidence) in enumerate(routing):
                try:
                    _, candidates, _, discoveries = self._worker_generate_local(
                        task=task,
                        specialist_id=spec_id,
                        confidence=confidence,
                        worker_id=worker_id,
                        n_workers=worker_count,
                    )
                    all_candidates.extend(candidates)
                    if discoveries and hasattr(self.grammar, "merge_discoveries"):
                        self.grammar.merge_discoveries(discoveries)
                except Exception as exc:
                    print(f"[WORKER ERROR] {spec_id}: {exc}")

        return all_candidates

    def _worker_generate_local(
        self,
        *,
        task: Dict[str, Any],
        specialist_id: str | None,
        confidence: float,
        worker_id: int,
        n_workers: int,
    ) -> tuple[str | None, list[Candidate], float, dict[str, Any]]:
        semantic_hints = task.get("semantic_hints") or []
        worker_hints = semantic_hints[worker_id::n_workers] if semantic_hints else None
        gen = CandidateGenerator(
            matryoshka_dim=task.get("matryoshka_dim", 512),
            max_candidates=task.get("candidates_per_worker", 6),
            shadow_copy=self.shadow_copy,
            drawing_galaxy=self.drawing_galaxy,
            codec_embedder=self.codec_embedder,
            embedding_galaxy=self.embedding_galaxy,
            cosine_bridge=self.cosine_bridge,
        )
        candidates = gen.generate_candidates(
            input_grid=task["input_grid"],
            train_examples=task.get("train_examples", []),
            semantic_hints=worker_hints,
            expected_output=task.get("expected_output"),
        )
        discoveries = dict(getattr(self.grammar, "_local_discoveries", {}))
        return specialist_id, candidates, confidence, discoveries


__all__ = ["ParallelCandidateGenerator"]
