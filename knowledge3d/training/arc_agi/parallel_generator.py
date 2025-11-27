"""
Parallel candidate generation (Tesla 3-6-9 pattern).

Uses MathCorePool to allocate GPU math cores, then fans out candidate
generation across workers. Keeps orchestration minimal and avoids numpy in the
hot path.
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional

from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator, Candidate
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
    ) -> None:
        self.num_workers = num_workers
        self.candidates_per_worker = candidates_per_worker
        self.top_k = top_k
        self.matryoshka_dim = matryoshka_dim
        self.shadow_copy = shadow_copy
        self.core_pool = get_global_math_core_pool()

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

        all_candidates: List[Candidate] = []
        pairs = list(zip(core_ids, executors)) if core_ids else [(None, ARCRPNExecutor(pool=self.core_pool, instance_id=None))]
        for core_id, executor in pairs:
            try:
                gen = CandidateGenerator(
                    matryoshka_dim=self.matryoshka_dim,
                    max_candidates=self.candidates_per_worker,
                    shadow_copy=self.shadow_copy,
                    executor=executor,
                )
                cand_list = gen.generate_candidates(
                    input_grid=input_grid,
                    train_examples=train_examples,
                    semantic_hints=semantic_hints,
                    expected_output=expected_output,
                )
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

        # Score and select top-K by overlap with expected output if available; otherwise keep first K.
        if expected_output:
            scored = []
            for grid, instr, prog in all_candidates:
                scored.append((self._score(grid, expected_output), grid, instr, prog))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: self.top_k]
            return [(g, i, p) for _, g, i, p in top]

        # Fallback: return first top_k
        return all_candidates[: self.top_k]

    @staticmethod
    def _score(output_grid: Sequence[Sequence[int]], expected_grid: Sequence[Sequence[int]]) -> float:
        if not output_grid or not expected_grid:
            return 0.0
        if len(output_grid) != len(expected_grid) or len(output_grid[0]) != len(expected_grid[0]):
            return 0.0
        matches = 0
        total = 0
        for r1, r2 in zip(output_grid, expected_grid):
            for a, b in zip(r1, r2):
                total += 1
                if a == b:
                    matches += 1
        return matches / total if total else 0.0


__all__ = ["ParallelCandidateGenerator"]
