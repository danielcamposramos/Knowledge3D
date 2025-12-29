"""
Compositional candidate generation for ARC-AGI.

Lightweight beam search over discovered programs (DualShadowCopy) to build
N-step chains that may solve harder tasks. Keeps computation bounded via
beam_width and max_depth.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


class CompositionalCandidateGenerator:
    """Generate N-step compositions from discovered programs."""

    def __init__(
        self,
        shadow_copy: DualShadowCopy,
        executor: ARCRPNExecutor,
        *,
        max_depth: int = 4,
        beam_width: int = 10,
        min_score: float = 0.45,
    ) -> None:
        self.shadow_copy = shadow_copy
        self.executor = executor
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_score = min_score

    def generate_compositions(
        self,
        input_grid: Sequence[Sequence[int]],
        expected_output: Sequence[Sequence[int]],
    ) -> List[Dict]:
        """Return compositional candidates as dictionaries."""
        programs = self._top_programs()
        if not programs:
            return []

        # Seed beam with single-program executions
        beam = []
        for prog in programs:
            try:
                output = self.executor.execute(input_grid, prog["program"])
            except Exception:
                continue
            score = self._score_output(output, expected_output)
            beam.append(
                {
                    "chain": [prog],
                    "output": output,
                    "score": score,
                    "program": prog["program"],
                    "description": prog.get("description", prog.get("name", "prog")),
                    "depth": 1,
                }
            )

        beam = sorted(beam, key=lambda c: c["score"], reverse=True)[: self.beam_width]
        all_candidates: List[Dict] = []

        # Iteratively deepen
        for depth in range(2, self.max_depth + 1):
            new_beam: List[Dict] = []
            for entry in beam:
                for prog in programs:
                    if entry["chain"] and entry["chain"][-1] is prog:
                        continue  # skip immediate repeats
                    try:
                        output = self.executor.execute(entry["output"], prog["program"])
                    except Exception:
                        continue
                    score = self._score_output(output, expected_output)
                    chained_program = f"{entry['program']} {prog['program']}".strip()
                    description = f"{entry['description']} → {prog.get('description', prog.get('name', 'prog'))}"
                    new_beam.append(
                        {
                            "chain": entry["chain"] + [prog],
                            "output": output,
                            "score": score,
                            "program": chained_program,
                            "description": description,
                            "depth": depth,
                        }
                    )

            if not new_beam:
                break

            # Keep best by score
            new_beam = sorted(new_beam, key=lambda c: c["score"], reverse=True)[: self.beam_width]
            high_quality = [c for c in new_beam if c["score"] >= self.min_score]
            all_candidates.extend(high_quality)
            beam = new_beam

            # Stop early if no high-quality chains emerge
            if not high_quality:
                break

        return all_candidates

    def _top_programs(self) -> List[Dict]:
        """Fetch top programs from shadow copy by quality."""
        lib = getattr(self.shadow_copy, "library", []) or []
        # Sort by quality_score descending
        sorted_lib = sorted(lib, key=lambda e: e.get("quality_score", 0.0), reverse=True)
        return sorted_lib[: self.beam_width]

    @staticmethod
    def _score_output(output_grid: Sequence[Sequence[int]], expected_grid: Sequence[Sequence[int]]) -> float:
        """Simple overlap score (exact match -> 1.0)."""
        if not output_grid or not expected_grid:
            return 0.0
        if len(output_grid) != len(expected_grid) or len(output_grid[0]) != len(expected_grid[0]):
            return 0.0
        matches = 0
        total = 0
        for row_out, row_exp in zip(output_grid, expected_grid):
            for a, b in zip(row_out, row_exp):
                total += 1
                if a == b:
                    matches += 1
        return matches / total if total else 0.0
