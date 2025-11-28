"""Generate multiple candidate solutions for ARC-AGI tasks.

This module expands the Phase 2 pure-procedural baseline by exploring a small
space of deterministic transformations (rotate/flip/translate/recolor and
lightweight compositions). It keeps the procedural path untouched and surfaces
up to ~20 unique candidates that downstream TRM ranking can score.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.training.arc_agi.multimodal_parser import MultimodalSemanticParser
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.sovereign_utils import (
    bounding_box_nonzero,
    dot,
    grids_equal,
    grid_shape,
    is_grid,
    l2_norm,
    to_int_grid,
    translate_grid,
    unique_nonzero,
)


Candidate = Tuple[List[List[int]], str, str]  # (output_grid, instruction, rpn_program)


class CandidateGenerator:
    """Generate multiple candidate solutions for ARC tasks."""

    def __init__(
        self,
        matryoshka_dim: int = 512,
        max_candidates: int = 369,
        shadow_copy: Optional[DualShadowCopy] = None,
        executor: Optional[ARCRPNExecutor] = None,
        codec_embedder: Any | None = None,
        embedder_type: str = "multimodal",
        embedding_galaxy: Optional[Dict[int, List[float]]] = None,
        cosine_bridge: Optional[CosineSimilarityBridge] = None,
    ):
        self.parser = MultimodalSemanticParser()
        self.compiler = SemanticToRPNCompiler()
        self.executor = executor or ARCRPNExecutor()
        self.processor = ARCGridProcessor(
            matryoshka_dim=matryoshka_dim,
            codec_embedder=codec_embedder,
            embedder_type=embedder_type,
            executor=self.executor,
        )
        self.max_candidates = max_candidates  # SOVEREIGN: Tesla 3-6-9 (increased from 69)
        self.shadow_copy = shadow_copy  # Optional access to discovered programs for compositions
        self.embedding_galaxy = embedding_galaxy
        self.cosine_bridge = cosine_bridge or CosineSimilarityBridge()

    def _exec_rpn(self, grid: Sequence[Sequence[int]], program: str) -> Optional[List[List[int]]]:
        """Execute RPN program via executor; return None on failure."""
        try:
            return self.executor.execute(grid, program)
        except Exception:
            return None

    def generate_candidates(
        self,
        input_grid: Sequence[Sequence[int]],
        train_examples: List[Dict],
        semantic_hints: List[str] = None,
        expected_output: Optional[Sequence[Sequence[int]]] = None,
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

        # 6) Compositional discovery: chain discovered programs (if available).
        if self.shadow_copy is not None and expected_output is not None:
            compositional = self._generate_compositional_candidates(input_grid, expected_output)
            if compositional:
                print(f"  [COMPOSITIONAL GEN] Generated {len(compositional)} compositional candidates")
            candidates.extend(compositional)

        # Sovereign: embedding galaxy must exist; batch-compute any missing embeddings on GPU.
        if self.embedding_galaxy is None:
            raise RuntimeError(
                "SOVEREIGNTY VIOLATION: embedding_galaxy is None. "
                "Run preprocessing: python scripts/preprocess_arc_embeddings.py"
            )

        missing_grids: List[Sequence[Sequence[int]]] = []
        missing_hashes: List[int] = []

        if expected_output is not None:
            exp_hash = self._hash_grid(expected_output)
            if exp_hash not in self.embedding_galaxy:
                missing_grids.append(expected_output)
                missing_hashes.append(exp_hash)

        for grid, _, _ in candidates:
            h = self._hash_grid(grid)
            if h not in self.embedding_galaxy:
                missing_grids.append(grid)
                missing_hashes.append(h)

        if missing_grids:
            print(f"  [GALAXY LAZY] Computing {len(missing_grids)} missing embeddings (batch GPU)")
            batch_embeddings = self.processor._grid_to_spatial_embedding_batch(missing_grids)
            for h, emb in zip(missing_hashes, batch_embeddings):
                self.embedding_galaxy[h] = emb

        # Deduplicate by output grid content.
        deduped = self._deduplicate_candidates(candidates)

        # Semantic ranking using sovereign embeddings when expected output is available.
        if expected_output is not None and deduped:
            deduped = self._rank_by_similarity(deduped, expected_output)

        # Cap the list.
        return deduped[: self.max_candidates]

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
        base_grid = to_int_grid(grid)

        # Rotations
        for k, angle in ((1, 90), (2, 180), (3, 270)):
            rpn = f"{k} rotate"
            rotated = self._exec_rpn(grid, rpn)
            if rotated is None:
                rotated = self.processor._apply_rotation(grid, angle)
            candidates.append((rotated, f"Rotate {angle} degrees", rpn))

        # Flips
        fh = self._exec_rpn(grid, "FLIP_H") or self.processor._apply_flip_horizontal(grid)
        fv = self._exec_rpn(grid, "FLIP_V") or self.processor._apply_flip_vertical(grid)
        candidates.append((fh, "Flip horizontally", "FLIP_H"))
        candidates.append((fv, "Flip vertically", "FLIP_V"))

        # Translations to canonical anchors using bounding box.
        bbox = self._bounding_box(base_grid)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            h, w = grid_shape(base_grid)
            anchors = {
                "top-left": (0 - y0, 0 - x0),
                "top-right": (0 - y0, (w - 1) - x1),
                "bottom-left": ((h - 1) - y1, 0 - x0),
                "bottom-right": ((h - 1) - y1, (w - 1) - x1),
                "center": (h // 2 - (y0 + y1) // 2, w // 2 - (x0 + x1) // 2),
            }
            for name, (dy, dx) in anchors.items():
                rpn = f"{dx} {dy} TRANSLATE"
                translated = self._exec_rpn(grid, rpn) or self.processor._apply_translation(grid, dx=int(dx), dy=int(dy))
                candidates.append((translated, f"Move object to {name}", rpn))

        # Single-color recolors (try observed colors).
        unique_colors = self._get_unique_colors(base_grid)
        for src in unique_colors:
            for dst in range(1, 10):
                if dst == src:
                    continue
                recolored = self._recolor_grid(base_grid, src, dst)
                candidates.append((recolored, f"Recolor {src}->{dst}", f"{src} {dst} RECOLOR"))

        return candidates

    # ------------------------------------------------------------------ #
    # Compositional discovery (beam over discovered programs)
    # ------------------------------------------------------------------ #
    def _generate_compositional_candidates(
        self,
        input_grid: Sequence[Sequence[int]],
        expected_output: Sequence[Sequence[int]],
        *,
        max_depth: int = 4,
        beam_width: int = 10,
    ) -> List[Candidate]:
        """
        Generate N-step compositions from discovered programs using beam search.

        This is intentionally lightweight: we only explore top-k library entries
        by quality and keep the beam pruned by score at each depth.
        """
        try:
            from knowledge3d.training.arc_agi.compositional_generator import CompositionalCandidateGenerator
        except Exception as e:
            print(f"  [COMPOSITIONAL GEN] Skipping (import failed): {e}")
            return []

        if self.shadow_copy is None or not getattr(self.shadow_copy, "library", None):
            return []

        comp_gen = CompositionalCandidateGenerator(
            shadow_copy=self.shadow_copy,
            executor=self.executor,
            max_depth=max_depth,
            beam_width=beam_width,
        )
        comps = comp_gen.generate_compositions(
            input_grid=input_grid,
            expected_output=expected_output,
        )

        # Map to Candidate tuples
        return [(c["output"], c["description"], c["program"]) for c in comps]

    # ------------------------------------------------------------------ #
    # Composition candidates
    # ------------------------------------------------------------------ #
    def _generate_composition_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        candidates: List[Candidate] = []
        base_grid = to_int_grid(grid)
        colors = self._get_unique_colors(base_grid)
        colors = colors or [1]

        # Rotate then recolor dominant color.
        if colors:
            dominant = colors[0]
            for k, angle in ((1, 90), (2, 180), (3, 270)):
                rpn_rotate = f"{k} rotate"
                rotated = self._exec_rpn(grid, rpn_rotate) or self.processor._apply_rotation(grid, angle)
                for dst in range(1, 4):
                    if dst == dominant:
                        continue
                    rpn = f"{rpn_rotate} {dominant} {dst} RECOLOR"
                    recolored = self._exec_rpn(rotated, f"{dominant} {dst} RECOLOR") or self._recolor_grid(rotated, dominant, dst)
                    candidates.append(
                        (
                            recolored,
                            f"Rotate {angle} then recolor {dominant}->{dst}",
                            rpn,
                        )
                    )

        # Flip then recolor.
        for flip_name, flipped in [
            ("Flip horizontally", self._exec_rpn(grid, "FLIP_H") or self.processor._apply_flip_horizontal(grid)),
            ("Flip vertically", self._exec_rpn(grid, "FLIP_V") or self.processor._apply_flip_vertical(grid)),
        ]:
            for src in colors:
                for dst in range(1, 4):
                    if dst == src:
                        continue
                    recolored = self._exec_rpn(flipped, f"{src} {dst} RECOLOR") or self._recolor_grid(flipped, src, dst)
                    candidates.append(
                        (
                            recolored,
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
        base_grid = to_int_grid(grid)

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
                rpn = f"{k} rotate"
                rotated = self._exec_rpn(grid, rpn) or self.processor._apply_rotation(grid, angle)
                candidates.append((rotated, f"[SEMANTIC] Rotate {angle}°", rpn))
            # Add 45-degree variants if grid is square
            h, w = grid_shape(base_grid)
            if h == w and h <= 10:  # Only for small square grids
                for angle in [45, 135, 225, 315]:
                    # Approximate 45° rotation with composition
                    temp = self._exec_rpn(grid, "1 rotate") or self.processor._apply_rotation(grid, 90)
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
                filled = self._fill_empty(base_grid, color)
                candidates.append((filled, f"[SEMANTIC] Fill empty with {color}", f"0 {color} RECOLOR"))

        # Generate MORE recoloring variants if color_change hint detected
        if has_color_change:
            unique_colors = self._get_unique_colors(base_grid)
            for src in unique_colors:
                for dst in range(1, 10):
                    if dst != src:
                        recolored = self._exec_rpn(base_grid, f"{src} {dst} RECOLOR") or self._recolor_grid(base_grid, src, dst)
                        candidates.append(
                            (recolored, f"[SEMANTIC] Recolor {src}→{dst}", f"{src} {dst} RECOLOR")
                        )

        # Generate translation variants if movement hint detected
        if has_translation:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                translated = self.processor._apply_translation(grid, dx=dx, dy=dy)
                candidates.append(
                    (translated, f"[SEMANTIC] Shift ({dx},{dy})", f"{dx} {dy} TRANSLATE")
                )

        # Cross-pattern compositions when multiple patterns detected
        cross_candidates = self._generate_cross_pattern_candidates(
            grid,
            has_rotation=has_rotation,
            has_flip=has_flip,
            has_color_change=has_color_change,
            has_translation=has_translation,
        )
        if cross_candidates:
            print(f"  [SEMANTIC CROSS] Generated {len(cross_candidates)} cross-pattern candidates")
            candidates.extend(cross_candidates)

        return candidates

    def _generate_cross_pattern_candidates(
        self,
        grid: Sequence[Sequence[int]],
        *,
        has_rotation: bool,
        has_flip: bool,
        has_color_change: bool,
        has_translation: bool,
    ) -> List[Candidate]:
        """Generate simple two-step cross-pattern compositions driven by hints."""
        base_grid = to_int_grid(grid)
        candidates: List[Candidate] = []

        # rotation + color
        if has_rotation and has_color_change:
            colors = self._get_unique_colors(base_grid) or [1]
            dominant = colors[0]
            for k, angle in ((1, 90), (2, 180), (3, 270)):
                rpn_rotate = f"{k} rotate"
                rotated = self._exec_rpn(grid, rpn_rotate) or self.processor._apply_rotation(grid, angle)
                for dst in range(1, 4):
                    if dst == dominant:
                        continue
                    recolored = self._exec_rpn(rotated, f"{dominant} {dst} RECOLOR") or self._recolor_grid(rotated, dominant, dst)
                    candidates.append(
                        (
                            recolored,
                            f"[SEMANTIC CROSS] Rotate {angle} then recolor {dominant}->{dst}",
                            f"{rpn_rotate} {dominant} {dst} RECOLOR",
                        )
                    )

        # flip + translation
        if has_flip and has_translation:
            bbox = self._bounding_box(base_grid)
            if bbox is not None:
                y0, y1, x0, x1 = bbox
                h, w = grid_shape(base_grid)
                anchors = [
                    (0 - y0, 0 - x0, "top-left"),
                    (0 - y0, (w - 1) - x1, "top-right"),
                    ((h - 1) - y1, 0 - x0, "bottom-left"),
                    ((h - 1) - y1, (w - 1) - x1, "bottom-right"),
                ]
                flips = [
                    ("FLIP_H", self._exec_rpn(grid, "FLIP_H") or self.processor._apply_flip_horizontal(grid)),
                    ("FLIP_V", self._exec_rpn(grid, "FLIP_V") or self.processor._apply_flip_vertical(grid)),
                ]
                for flip_name, flipped in flips:
                    for dy, dx, label in anchors:
                        translated = self._exec_rpn(flipped, f"{int(dx)} {int(dy)} TRANSLATE") or self.processor._apply_translation(flipped, dx=int(dx), dy=int(dy))
                        rpn = f"{flip_name} {int(dx)} {int(dy)} TRANSLATE"
                        candidates.append(
                            (
                                translated,
                                f"[SEMANTIC CROSS] {flip_name} then move to {label}",
                                rpn,
                            )
                        )

        # rotation + translation (rotate then center)
        if has_rotation and has_translation:
            bbox = self._bounding_box(base_grid)
            if bbox is not None:
                y0, y1, x0, x1 = bbox
                h, w = grid_shape(base_grid)
                dy = h // 2 - (y0 + y1) // 2
                dx = w // 2 - (x0 + x1) // 2
                for k, angle in ((1, 90), (2, 180), (3, 270)):
                    rpn_rotate = f"{k} rotate"
                    rotated = self._exec_rpn(grid, rpn_rotate) or self.processor._apply_rotation(grid, angle)
                    translated = self._exec_rpn(rotated, f"{int(dx)} {int(dy)} TRANSLATE") or self.processor._apply_translation(rotated, dx=int(dx), dy=int(dy))
                    rpn = f"{k} rotate {int(dx)} {int(dy)} TRANSLATE"
                    candidates.append(
                        (
                            translated,
                            f"[SEMANTIC CROSS] Rotate {angle} then center",
                            rpn,
                        )
                    )

        return candidates

    # ------------------------------------------------------------------ #
    # Math-style candidates
    # ------------------------------------------------------------------ #
    def _generate_math_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        candidates: List[Candidate] = []
        base_grid = to_int_grid(grid)
        h, w = grid_shape(base_grid)

        for condition, parity in (("even", 0), ("odd", 1)):
            for color in range(1, 4):
                rpn_fill = f"FOR_EACH_CELL GET_ROW GET_COL ADD 2 MOD {parity} EQ IF_TRUE {color} FILL"
                filled = self._exec_rpn(base_grid, rpn_fill) or self._checkerboard_fill(base_grid, parity, color)
                instruction = f"Fill cells where row+col is {condition} with {color}"
                rpn_program = (
                    "FOR_EACH_CELL GET_ROW GET_COL ADD 2 MOD "
                    f"{parity} EQ IF_TRUE {color} FILL"
                )
                candidates.append((filled, instruction, rpn_program))

        return candidates

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _deduplicate_candidates(self, candidates: Iterable[Candidate]) -> List[Candidate]:
        """Remove duplicate output grids while preserving order."""
        seen = set()
        unique: List[Candidate] = []
        for output, instruction, rpn in candidates:
            if not is_grid(output):
                continue
            key = tuple(tuple(row) for row in output)
            if key in seen:
                continue
            seen.add(key)
            unique.append((output, instruction, rpn))
        return unique

    @staticmethod
    def _bounding_box(grid: Sequence[Sequence[int]]) -> Tuple[int, int, int, int] | None:
        """Return bounding box (y0, y1, x0, x1) of non-zero pixels."""
        return bounding_box_nonzero(grid)

    @staticmethod
    def _get_unique_colors(grid: Sequence[Sequence[int]]) -> List[int]:
        """Unique non-zero colors from grid."""
        return unique_nonzero(grid)

    def _rank_by_similarity(
        self,
        candidates: List[Candidate],
        expected_output: Sequence[Sequence[int]],
    ) -> List[Candidate]:
        """Rank candidates by cosine similarity using SOVEREIGN Galaxy + PTX (no fallbacks)."""
        if not candidates:
            return candidates

        if self.embedding_galaxy is None:
            raise RuntimeError(
                "SOVEREIGNTY VIOLATION: embedding_galaxy is None. "
                "Run preprocessing: python scripts/preprocess_arc_embeddings.py"
            )

        expected_hash = self._hash_grid(expected_output)
        expected_emb = self.embedding_galaxy.get(expected_hash)
        if expected_emb is None:
            raise RuntimeError(
                f"SOVEREIGNTY VIOLATION: Expected output embedding not found in Galaxy (hash={expected_hash}). "
                "Preprocessing incomplete or grid not in preprocessing set."
            )

        embeddings: List[List[float]] = []
        for grid, _, _ in candidates:
            h = self._hash_grid(grid)
            emb = self.embedding_galaxy.get(h)
            if emb is None:
                raise RuntimeError(
                    f"SOVEREIGNTY VIOLATION: Candidate embedding not found in Galaxy (hash={h}). "
                    "Grid not present in preprocessing set."
                )
            embeddings.append(emb)

        scores = self.cosine_bridge.compute_similarities(embeddings, expected_emb)
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored]

    @staticmethod
    def _hash_grid(grid: Sequence[Sequence[int]]) -> int:
        return hash(tuple(tuple(int(c) for c in row) for row in grid))

    @staticmethod
    def _recolor_grid(grid: Sequence[Sequence[int]], src: int, dst: int) -> List[List[int]]:
        """Recolor src→dst using pure Python."""
        recolored = []
        for row in grid:
            recolored.append([dst if int(cell) == src else int(cell) for cell in row])
        return recolored

    @staticmethod
    def _fill_empty(grid: Sequence[Sequence[int]], color: int) -> List[List[int]]:
        """Fill zero cells with the provided color."""
        return [[color if int(cell) == 0 else int(cell) for cell in row] for row in grid]

    @staticmethod
    def _checkerboard_fill(grid: Sequence[Sequence[int]], parity: int, color: int) -> List[List[int]]:
        """Fill cells where (row+col) % 2 == parity with color."""
        filled = []
        for y, row in enumerate(grid):
            new_row = []
            for x, cell in enumerate(row):
                if (y + x) % 2 == parity:
                    new_row.append(color)
                else:
                    new_row.append(int(cell))
            filled.append(new_row)
        return filled

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
