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

    def __init__(self, matryoshka_dim: int = 512, max_candidates: int = 369):
        self.parser = MultimodalSemanticParser()
        self.compiler = SemanticToRPNCompiler()
        self.executor = ARCRPNExecutor()
        self.processor = ARCGridProcessor(matryoshka_dim=matryoshka_dim, embedder_type="procedural")
        self.max_candidates = max_candidates  # SOVEREIGN: Tesla 3-6-9 (increased from 69)

    def generate_candidates(
        self, input_grid: Sequence[Sequence[int]], train_examples: List[Dict], semantic_hints: List[str] = None
    ) -> List[Candidate]:
        """
        Generate multiple deterministic candidates for a single ARC grid.

        Args:
            input_grid: The grid to transform.
            train_examples: Upstream examples (input/output) for the same task
                used to infer likely primitives.
            semantic_hints: Optional word hints from semantic context to guide generation.

        Returns:
            List of (output_grid, instruction, rpn_program) tuples.
        """
        candidates: List[Candidate] = []

        # 1) Example-driven inference: reuse primitives seen in prior examples.
        # SOVEREIGN: Use 9 examples (Tesla 3-6-9, increased from 3)
        for example in train_examples[:9]:
            inferred = self._infer_from_example(example, input_grid)
            if inferred:
                candidates.append(inferred)

        # 2) Semantic-guided candidates: use word hints to expand search space.
        # SOVEREIGN: Closes the semantic layer → generation feedback loop.
        if semantic_hints:
            semantic_candidates = self._generate_semantic_guided_candidates(input_grid, semantic_hints)
            print(f"  [SEMANTIC GEN] Generated {len(semantic_candidates)} semantic-guided candidates from {len(semantic_hints)} hints")
            candidates.extend(semantic_candidates)
        else:
            print(f"  [SEMANTIC GEN] No semantic hints provided, skipping semantic-guided generation")

        # 3) Primitive search: rotations, flips, recolors, simple translations.
        candidates.extend(self._generate_primitive_candidates(input_grid))

        # 4) Compositions: simple rotate/flip + recolor combos.
        candidates.extend(self._generate_composition_candidates(input_grid))

        # 5) Math-style patterns: checkerboard even/odd fills.
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
    # Semantic-guided candidates
    # ------------------------------------------------------------------ #
    def _generate_semantic_guided_candidates(
        self, grid: Sequence[Sequence[int]], semantic_hints: List[str]
    ) -> List[Candidate]:
        """
        Generate candidates based on semantic word hints from discovered patterns.

        SOVEREIGN: This closes the semantic layer → generation feedback loop.
        Word meanings instruct new candidate generation to expand search space.
        """
        candidates: List[Candidate] = []
        arr = np.asarray(grid, dtype=int)

        # Extract pattern types from semantic hints
        hints_lower = [h.lower() for h in semantic_hints]
        has_rotation = any("rotation" in h or "rotate" in h for h in hints_lower)
        has_flip = any("flip" in h or "mirror" in h or "reflect" in h for h in hints_lower)
        has_sparse = any("sparse" in h or "empty" in h for h in hints_lower)
        has_color_change = any("color" in h or "recolor" in h for h in hints_lower)
        has_translation = any("move" in h or "translate" in h or "shift" in h for h in hints_lower)

        print(f"  [SEMANTIC PATTERNS] rotation={has_rotation}, flip={has_flip}, sparse={has_sparse}, color={has_color_change}, translate={has_translation}")

        # Generate MORE rotation variants if rotation hint detected
        if has_rotation:
            for k, angle in ((1, 90), (2, 180), (3, 270)):
                rotated = self.processor._apply_rotation(grid, angle)
                candidates.append((rotated, f"[SEMANTIC] Rotate {angle}°", f"{k} rotate"))
            # Add 45-degree variants if grid is square
            h, w = arr.shape
            if h == w and h <= 10:  # Only for small square grids
                for angle in [45, 135, 225, 315]:
                    # Approximate 45° rotation with composition
                    temp = self.processor._apply_rotation(grid, 90)
                    candidates.append((temp, f"[SEMANTIC] Rotate ~{angle}°", "1 rotate"))

        # Generate MORE flip variants if flip hint detected
        if has_flip:
            candidates.append(
                (self.processor._apply_flip_horizontal(grid), "[SEMANTIC] Flip H", "FLIP_H")
            )
            candidates.append(
                (self.processor._apply_flip_vertical(grid), "[SEMANTIC] Flip V", "FLIP_V")
            )
            # Diagonal flips (transpose)
            transposed = [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]
            candidates.append((transposed, "[SEMANTIC] Flip diagonal", "TRANSPOSE"))

        # Generate fill patterns if sparse hint detected
        if has_sparse:
            for color in range(1, 10):
                filled = arr.copy()
                filled[filled == 0] = color
                candidates.append(
                    (filled.tolist(), f"[SEMANTIC] Fill empty with {color}", f"0 {color} RECOLOR")
                )

        # Generate MORE recoloring variants if color_change hint detected
        if has_color_change:
            unique_colors = [int(c) for c in np.unique(arr) if c != 0]
            for src in unique_colors:
                for dst in range(1, 10):
                    if dst != src:
                        recolored = arr.copy()
                        recolored[recolored == src] = dst
                        candidates.append(
                            (recolored.tolist(), f"[SEMANTIC] Recolor {src}→{dst}", f"{src} {dst} RECOLOR")
                        )

        # Generate translation variants if movement hint detected
        if has_translation:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                translated = self.processor._apply_translation(grid, dx=dx, dy=dy)
                candidates.append(
                    (translated, f"[SEMANTIC] Shift ({dx},{dy})", f"{dx} {dy} TRANSLATE")
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
