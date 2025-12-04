"""
Parallel candidate generation (Tesla 3-6-9 pattern).

Uses MathCorePool to allocate GPU math cores, then fans out candidate
generation across workers. Keeps orchestration minimal and avoids numpy in the
hot path.
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional

from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator, Candidate
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool, get_global_math_core_pool
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


class ParallelCandidateGenerator:
    def __init__(
        self,
        *,
        num_workers: int = 9,
        candidates_per_worker: int = 6,
        top_k: int = 3,
        matryoshka_dim: int = 512,
        shadow_copy: Optional[DualShadowCopy] = None,
        drawing_galaxy: Optional[DrawingGalaxy] = None,
        codec_embedder: Any | None = None,
        embedding_galaxy: Optional[Dict[int, List[float]]] = None,
        cosine_bridge: Any | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.candidates_per_worker = candidates_per_worker
        self.top_k = top_k
        self.matryoshka_dim = matryoshka_dim
        self.shadow_copy = shadow_copy
        self.drawing_galaxy = drawing_galaxy
        self.core_pool = get_global_math_core_pool()
        self.codec_embedder = codec_embedder
        self.embedding_galaxy = embedding_galaxy
        self.cosine_bridge = cosine_bridge

    def generate_parallel(
        self,
        input_grid: Sequence[Sequence[int]],
        train_examples: List[Dict[str, Any]],
        semantic_hints: Optional[List[str]],
        expected_output: Optional[Sequence[Sequence[int]]],
    ) -> List[Candidate]:
        # Allocate math cores (best-effort; will raise if capacity exceeded)
        core_ids: List[int] = []
        executors: List[ARCRPNExecutor] = []
        try:
            for _ in range(self.num_workers):
                core_id = self.core_pool.spawn_core(tier=1, reuse=True)
                core_ids.append(core_id)
                executors.append(ARCRPNExecutor(pool=self.core_pool, instance_id=core_id))
        except Exception as e:
            # If pool is constrained, fall back to whatever we acquired
            print(f"  [PARALLEL GEN] Limited by MathCorePool ({len(core_ids)} cores acquired): {e}")
            # If we failed before creating executors for all cores acquired, create for remaining.
            missing = len(core_ids) - len(executors)
            for i in range(missing):
                executors.append(ARCRPNExecutor(pool=self.core_pool, instance_id=core_ids[len(executors)]))

        # Partition semantic hints across available workers to avoid redundant work.
        num_worker_slots = len(executors) if executors else 1
        semantic_partitions: List[Optional[List[str]]] = []
        if semantic_hints:
            total_hints = len(semantic_hints)
            hints_per_worker = max(1, total_hints // num_worker_slots)
            for worker_idx in range(num_worker_slots):
                start_idx = worker_idx * hints_per_worker
                end_idx = start_idx + hints_per_worker if worker_idx < num_worker_slots - 1 else total_hints
                worker_hints = semantic_hints[start_idx:end_idx] if start_idx < total_hints else []
                semantic_partitions.append(worker_hints)
                print(f"  [WORKER {worker_idx}] Assigned hints {start_idx}:{end_idx} ({len(worker_hints)} hints)")
        else:
            semantic_partitions = [None for _ in range(num_worker_slots)]

        all_candidates: List[Candidate] = []
        pairs = (
            list(zip(core_ids, executors, semantic_partitions))
            if core_ids
            else [(None, ARCRPNExecutor(pool=self.core_pool, instance_id=None), semantic_partitions[0])]
        )
        for worker_idx, (core_id, executor, worker_hints) in enumerate(pairs):
            try:
                gen = CandidateGenerator(
                    matryoshka_dim=self.matryoshka_dim,
                    max_candidates=self.candidates_per_worker,
                    shadow_copy=self.shadow_copy,
                    drawing_galaxy=self.drawing_galaxy,
                    executor=executor,
                    codec_embedder=self.codec_embedder,
                    embedding_galaxy=self.embedding_galaxy,
                    cosine_bridge=self.cosine_bridge,
                )
                cand_list = gen.generate_candidates(
                    input_grid=input_grid,
                    train_examples=train_examples,
                    semantic_hints=worker_hints,
                    expected_output=expected_output,
                )
                hint_count = len(worker_hints) if worker_hints else 0
                print(f"  [WORKER {worker_idx}] Generated {len(cand_list)} candidates from {hint_count} hints")
                all_candidates.extend(cand_list)
            finally:
                if core_id is not None:
                    self.core_pool.release_core(core_id, pool=True)

        # Instrumentation: report PTX vs fallback usage across executors.
        if executors:
            succ = sum(getattr(ex, "ptx_success_count", 0) for ex in executors)
            fallback = sum(getattr(ex, "ptx_fallback_count", 0) for ex in executors)
            total = succ + fallback
            rate = (100.0 * succ / total) if total else 0.0
            print(f"  [PARALLEL GEN] PTX success={succ}, fallback={fallback}, rate={rate:.1f}%")

        print(f"  [PARALLEL GEN] Total candidates before dedup: {len(all_candidates)}")

        # Sovereign: return all candidates for downstream semantic ranking (PTX cosine).
        print(f"  [PARALLEL GEN] Returning all {len(all_candidates)} candidates for semantic ranking")
        return all_candidates


__all__ = ["ParallelCandidateGenerator"]
