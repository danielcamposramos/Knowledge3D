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

import numpy as np

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
                    "output": np.asarray(output, dtype=np.int32),
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

        test_input_arr = np.asarray(test_input, dtype=np.int32)
        expected_arr = np.asarray(expected_output, dtype=np.int32) if expected_output is not None else None

        # 4) Execute programs needing execution and score.
        best = None
        exact_proc_best = None
        for cand in merged:
            if cand["output"] is None:
                try:
                    cand["output"] = np.asarray(executor.execute(test_input, cand["program"]), dtype=np.int32)
                except Exception:
                    cand["output"] = test_input_arr
            score = self._score_candidate(
                cand["output"],
                test_input_arr,
                expected_arr,
                source=cand["source"],
            )
            cand["score"] = score
            # Track exact procedural matches first
            if expected_arr is not None and cand["source"] == "baseline" and np.array_equal(cand["output"], expected_arr):
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
            input_grid=test_input_arr,
            output_grid=chosen["output"],
            task_id=task_id,
        )

        result = TaskResult(
            task_id=task_id,
            best_program=chosen["program"],
            program_type=chosen["program_type"],
            score=float(chosen["score"]),
            signature=signature,
            output_grid=chosen["output"].tolist(),
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
        output: np.ndarray,
        input_grid: np.ndarray,
        expected: Optional[np.ndarray] = None,
        *,
        source: str = "unknown",
    ) -> float:
        """Score candidates; align thresholds with DualShadowCopy recording."""
        if expected is not None and np.array_equal(output, expected):
            return 1.0

        score = 0.4  # below recording threshold

        # Reward non-identity transforms
        if not np.array_equal(output, input_grid):
            score += 0.2  # 0.6 baseline for any change

        num_colors = len(np.unique(output))
        filled_ratio = float(np.count_nonzero(output)) / float(output.size or 1)

        if num_colors > 1:
            score += 0.1  # 0.7
        if 0.1 < filled_ratio < 0.9:
            score += 0.1  # up to 0.8

        if source == "baseline" and score < 1.0:
            score = min(score + 0.1, 0.7)  # slight procedural bias, capped

        return min(score, 1.0)


__all__ = ["SovereignAIPipeline", "TaskResult"]
