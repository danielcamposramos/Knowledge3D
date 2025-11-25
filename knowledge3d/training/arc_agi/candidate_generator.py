"""Generate multiple candidate solutions for ARC-AGI tasks.

This module expands the Phase 2 pure-procedural baseline by exploring a small
space of deterministic transformations (rotate/flip/translate/recolor and
lightweight compositions). It keeps the procedural path untouched and surfaces
up to ~20 unique candidates that downstream TRM ranking can score.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.arc_agi.multimodal_parser import MultimodalSemanticParser
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler


Candidate = Tuple[List[List[int]], str, str]  # (output_grid, instruction, rpn_program)


class CandidateGenerator:
    """Generate multiple candidate solutions for ARC tasks."""

    def __init__(self, matryoshka_dim: int = 512, max_candidates: int = 69):
        self.parser = MultimodalSemanticParser()
        self.compiler = SemanticToRPNCompiler()
        self.executor = ARCRPNExecutor()
        self.processor = ARCGridProcessor(matryoshka_dim=matryoshka_dim, embedder_type="procedural")
        self.max_candidates = max_candidates

    def generate_candidates(
        self, input_grid: Sequence[Sequence[int]], train_examples: List[Dict]
    ) -> List[Candidate]:
        """
        Generate multiple deterministic candidates for a single ARC grid.

        Args:
            input_grid: The grid to transform.
            train_examples: Upstream examples (input/output) for the same task
                used to infer likely primitives.

        Returns:
            List of (output_grid, instruction, rpn_program) tuples.
        """
        candidates: List[Candidate] = []

        # 1) Example-driven inference: reuse primitives seen in prior examples.
        for example in train_examples[:3]:
            inferred = self._infer_from_example(example, input_grid)
            if inferred:
                candidates.append(inferred)

        # 2) Primitive search: rotations, flips, recolors, simple translations.
        candidates.extend(self._generate_primitive_candidates(input_grid))

        # 3) Compositions: simple rotate/flip + recolor combos.
        candidates.extend(self._generate_composition_candidates(input_grid))

        # 4) Math-style patterns: checkerboard even/odd fills.
        candidates.extend(self._generate_math_candidates(input_grid))

        # Deduplicate by output grid content and cap the list.
        return self._deduplicate_candidates(candidates)[: self.max_candidates]

    # ------------------------------------------------------------------ #
    # Example-driven inference
    # ------------------------------------------------------------------ #
    def _infer_from_example(
        self, example: Dict, target_grid: Sequence[Sequence[int]]
    ) -> Candidate | None:
        """Infer a transformation from a train example and apply to target."""
        if "input" not in example or "output" not in example:
            return None

        try:
            primitive = self.processor.detect_spatial_primitive(example["input"], example["output"])
        except Exception:
            return None

        if primitive.get("primitive") == "UNKNOWN" or not primitive.get("rpn_program"):
            return None

        rpn_program = primitive["rpn_program"]
        instruction = primitive.get("primitive", "inferred")

        try:
            output_grid = self.executor.execute(target_grid, rpn_program)
        except Exception:
            return None

        return (output_grid, instruction, rpn_program)

    # ------------------------------------------------------------------ #
    # Primitive candidates
    # ------------------------------------------------------------------ #
    def _generate_primitive_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        candidates: List[Candidate] = []
        arr = np.asarray(grid, dtype=int)

        # Rotations
        for k, angle in ((1, 90), (2, 180), (3, 270)):
            rotated = self.processor._apply_rotation(grid, angle)
            candidates.append((rotated, f"Rotate {angle} degrees", f"{k} rotate"))

        # Flips
        candidates.append(
            (self.processor._apply_flip_horizontal(grid), "Flip horizontally", "FLIP_H")
        )
        candidates.append((self.processor._apply_flip_vertical(grid), "Flip vertically", "FLIP_V"))

        # Translations to canonical anchors using bounding box.
        bbox = self._bounding_box(arr)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            h, w = arr.shape
            anchors = {
                "top-left": (0 - y0, 0 - x0),
                "top-right": (0 - y0, (w - 1) - x1),
                "bottom-left": ((h - 1) - y1, 0 - x0),
                "bottom-right": ((h - 1) - y1, (w - 1) - x1),
                "center": (h // 2 - (y0 + y1) // 2, w // 2 - (x0 + x1) // 2),
            }
            for name, (dy, dx) in anchors.items():
                translated = self.processor._apply_translation(grid, dx=int(dx), dy=int(dy))
                rpn = f"{dx} {dy} TRANSLATE"
                candidates.append((translated, f"Move object to {name}", rpn))

        # Single-color recolors (try observed colors).
        unique_colors = [int(c) for c in np.unique(arr) if c != 0]
        for src in unique_colors:
            for dst in range(1, 10):
                if dst == src:
                    continue
                recolored = arr.copy()
                recolored[recolored == src] = dst
                candidates.append((recolored.tolist(), f"Recolor {src}->{dst}", f"{src} {dst} RECOLOR"))

        return candidates

    # ------------------------------------------------------------------ #
    # Composition candidates
    # ------------------------------------------------------------------ #
    def _generate_composition_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        candidates: List[Candidate] = []
        arr = np.asarray(grid, dtype=int)
        colors = [int(c) for c in np.unique(arr) if c != 0]
        colors = colors or [1]

        # Rotate then recolor dominant color.
        if colors:
            dominant = colors[0]
            for k, angle in ((1, 90), (2, 180), (3, 270)):
                rotated = np.array(self.processor._apply_rotation(grid, angle))
                for dst in range(1, 4):
                    if dst == dominant:
                        continue
                    recolored = rotated.copy()
                    recolored[recolored == dominant] = dst
                    candidates.append(
                        (
                            recolored.tolist(),
                            f"Rotate {angle} then recolor {dominant}->{dst}",
                            f"{k} rotate {dominant} {dst} RECOLOR",
                        )
                    )

        # Flip then recolor.
        for flip_name, flipped in [
            ("Flip horizontally", np.array(self.processor._apply_flip_horizontal(grid))),
            ("Flip vertically", np.array(self.processor._apply_flip_vertical(grid))),
        ]:
            for src in colors:
                for dst in range(1, 4):
                    if dst == src:
                        continue
                    recolored = flipped.copy()
                    recolored[recolored == src] = dst
                    candidates.append(
                        (
                            recolored.tolist(),
                            f"{flip_name} then recolor {src}->{dst}",
                            f"{'FLIP_H' if 'horizontally' in flip_name else 'FLIP_V'} {src} {dst} RECOLOR",
                        )
                    )

        return candidates

    # ------------------------------------------------------------------ #
    # Math-style candidates
    # ------------------------------------------------------------------ #
    def _generate_math_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        candidates: List[Candidate] = []
        arr = np.asarray(grid, dtype=int)
        h, w = arr.shape

        for condition, parity in (("even", 0), ("odd", 1)):
            mask = ((np.arange(h)[:, None] + np.arange(w)) % 2) == parity
            for color in range(1, 4):
                filled = arr.copy()
                filled[mask] = color
                instruction = f"Fill cells where row+col is {condition} with {color}"
                rpn_program = (
                    "FOR_EACH_CELL GET_ROW GET_COL ADD 2 MOD "
                    f"{parity} EQ IF_TRUE {color} FILL"
                )
                candidates.append((filled.tolist(), instruction, rpn_program))

        return candidates

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _deduplicate_candidates(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        """Remove duplicate output grids while preserving order."""
        seen = set()
        unique: List[Candidate] = []
        for output, instruction, rpn in candidates:
            key = tuple(tuple(row) for row in output)
            if key in seen:
                continue
            seen.add(key)
            unique.append((output, instruction, rpn))
        return unique

    @staticmethod
    def _bounding_box(arr: np.ndarray) -> Tuple[int, int, int, int] | None:
        """Return bounding box (y0, y1, x0, x1) of non-zero pixels."""
        mask = arr != 0
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        return ys.min(), ys.max(), xs.min(), xs.max()

    # ------------------------------------------------------------------ #
    # Optional: semantic execution path (kept for completeness)
    # ------------------------------------------------------------------ #
    def _execute_instruction(self, grid: Sequence[Sequence[int]], instruction: str) -> List[List[int]]:
        """
        Execute a textual instruction by routing through the parser/compiler.

        This is a best-effort helper; if parsing or compilation fails, the
        caller should catch the exception and ignore the candidate.
        """
        semantic = self.parser.parse(instruction)
        rpn_program = self.compiler.compile(semantic)
        return self.executor.execute(grid, rpn_program)


__all__ = ["CandidateGenerator", "Candidate"]
