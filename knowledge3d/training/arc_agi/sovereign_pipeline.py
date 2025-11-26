"""
Sovereign ARC-AGI pipeline (Drawing + Grammar galaxies, TRM router).

This is a thin orchestrator that wires:
- DrawingGalaxy (visual atoms)
- GrammarGalaxy (196+ rules)
- SovereignTRMRouter (matryoshka + adapter; no external ML)
- ProgramComposer (cross-galaxy compositions)
- DualShadowCopy (evolution tracking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

# SOVEREIGN: No numpy in hot path! Use plain lists for grids.

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.sovereign_trm_router import SovereignTRMRouter, rule_score_hint
from knowledge3d.training.arc_agi.program_composer import ProgramComposer
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy


@dataclass
class TaskResult:
    task_id: str
    best_program: str
    program_type: str
    score: float
    signature: Dict
    output_grid: Optional[List[List[int]]] = None


def _grids_equal(grid1: Sequence[Sequence[int]], grid2: Sequence[Sequence[int]]) -> bool:
    """SOVEREIGN: Compare grids without numpy."""
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2) or list(row1) != list(row2):
            return False
    return True


class SovereignAIPipeline:
    """End-to-end sovereign pipeline (no external ML, GPU-only matryoshka)."""

    def __init__(self, matryoshka_dim: int = 512, *, staged_shadow: bool = False) -> None:
        self.drawing = DrawingGalaxy()
        self.grammar = GrammarGalaxy()
        self.shadow = DualShadowCopy(self.drawing, self.grammar, staged=staged_shadow)
        self.router = SovereignTRMRouter(self.drawing, self.grammar, shadow_copy=self.shadow, matryoshka_dim=matryoshka_dim)
        self.composer = ProgramComposer()
        self.results: List[TaskResult] = []

    def process_task(
        self,
        task_id: str,
        test_input: Sequence[Sequence[int]],
        *,
        train_examples: Optional[List[Dict]] = None,
        expected_output: Optional[Sequence[Sequence[int]]] = None,
        top_k: int = 12,
    ) -> TaskResult:
        """Process task merging procedural baseline + sovereign routing."""

        # 1) Procedural baseline candidates (train example analysis).
        from knowledge3d.training.arc_agi import CandidateGenerator
        from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor

        executor = ARCRPNExecutor()
        procedural_candidates: List[Dict] = []
        if train_examples:
            gen = CandidateGenerator(matryoshka_dim=self.router.matryoshka_dim)
            procedural_candidates = gen.generate_candidates(test_input, train_examples)

        # 2) TRM router candidates.
        trm_candidates = self.router.route(test_input, top_k=top_k)

        # 3) Merge candidate programs.
        merged: List[Dict] = []
        for output, instruction, rpn in procedural_candidates:
            merged.append(
                {
                    "program": rpn,
                    "program_type": "procedural",
                    "source": "baseline",
                    "output": output,  # SOVEREIGN: Keep as list, no numpy conversion
                }
            )

        for cand in trm_candidates:
            if "program" in cand:
                merged.append(
                    {
                        "program": cand["program"],
                        "program_type": cand.get("program_type", "semantic"),
                        "source": cand.get("source", "semantic_match"),
                        "output": None,
                        "signature": cand.get("semantic_context") or cand.get("signature", {}),
                    }
                )
            else:
                compositions = self.composer.compose(cand["drawing_program"], [cand["grammar_rule"]])
                for prog, ptype in compositions:
                    merged.append(
                        {
                            "program": prog,
                            "program_type": ptype,
                            "source": cand.get("source", "trm"),
                            "output": None,  # to be executed
                            "signature": cand.get("signature", {}),
                        }
                    )

        if not merged:
            raise RuntimeError("No candidates generated")

        # SOVEREIGN: Keep grids as lists, no numpy conversion
        test_input_list = [list(row) for row in test_input]
        expected_list = [list(row) for row in expected_output] if expected_output is not None else None

        # 4) Execute programs needing execution and score.
        best = None
        exact_proc_best = None
        for cand in merged:
            if cand["output"] is None:
                try:
                    cand["output"] = executor.execute(test_input, cand["program"])  # Returns list already
                except Exception:
                    cand["output"] = test_input_list
            score = self._score_candidate(
                cand["output"],
                test_input_list,
                expected_list,
                source=cand["source"],
            )
            cand["score"] = score
            # Track exact procedural matches first
            if expected_list is not None and cand["source"] == "baseline" and _grids_equal(cand["output"], expected_list):
                if exact_proc_best is None or score > exact_proc_best["score"]:
                    exact_proc_best = cand
            if best is None or score > best["score"]:
                best = cand

        # Prefer exact procedural match if available
        chosen = exact_proc_best or best

        assert chosen is not None
        signature = chosen.get("signature") or {"source": chosen["source"]}

        # Record discoveries according to shadow thresholds (with semantic context)
        self.shadow.record(
            signature,
            chosen["program"],
            chosen["program_type"],
            chosen["score"],
            input_grid=test_input_list,  # SOVEREIGN: Pass list, not numpy array
            output_grid=chosen["output"],
            task_id=task_id,
        )

        result = TaskResult(
            task_id=task_id,
            best_program=chosen["program"],
            program_type=chosen["program_type"],
            score=float(chosen["score"]),
            signature=signature,
            output_grid=chosen["output"],  # SOVEREIGN: Already a list, no .tolist() needed
        )
        self.results.append(result)
        return result

    def summary(self) -> Dict[str, int]:
        return {
            "tasks": len(self.results),
            "shadow_entries": len(self.shadow.library),
            "drawing_shapes": len(self.drawing.shapes),
            "grammar_rules": len(self.grammar.rules),
        }

    @staticmethod
    def _score_candidate(
        output: Sequence[Sequence[int]],
        input_grid: Sequence[Sequence[int]],
        expected: Optional[Sequence[Sequence[int]]] = None,
        *,
        source: str = "unknown",
    ) -> float:
        """Score candidates (SOVEREIGN: no numpy!); align thresholds with DualShadowCopy recording."""
        if expected is not None and _grids_equal(output, expected):
            return 1.0

        score = 0.4  # below recording threshold

        # Reward non-identity transforms
        if not _grids_equal(output, input_grid):
            score += 0.2  # 0.6 baseline for any change

        # Count unique colors (no numpy unique!)
        all_values = [val for row in output for val in row]
        num_colors = len(set(all_values)) if all_values else 0

        # Compute filled ratio (no numpy count_nonzero!)
        total_cells = sum(len(row) for row in output)
        nonzero_count = sum(1 for val in all_values if val != 0)
        filled_ratio = float(nonzero_count) / float(total_cells or 1)

        if num_colors > 1:
            score += 0.1  # 0.7
        if 0.1 < filled_ratio < 0.9:
            score += 0.1  # up to 0.8

        if source == "baseline" and score < 1.0:
            score = min(score + 0.1, 0.7)  # slight procedural bias, capped

        return min(score, 1.0)


__all__ = ["SovereignAIPipeline", "TaskResult"]
