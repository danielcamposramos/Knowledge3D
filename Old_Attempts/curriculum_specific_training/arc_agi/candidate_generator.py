"""Generate multiple candidate solutions for ARC-AGI tasks.

This module expands the Phase 2 pure-procedural baseline by exploring a small
space of deterministic transformations (rotate/flip/translate/recolor and
lightweight compositions). It keeps the procedural path untouched and surfaces
up to ~20 unique candidates that downstream TRM ranking can score.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from knowledge3d.training.arc_agi.grid_processor import ARCGridProcessor
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy, DrawingItem
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.training.arc_agi.multimodal_parser import MultimodalSemanticParser
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.arc_agi.semantic_compiler import SemanticToRPNCompiler
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.semantic_context import SemanticVocabulary
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
        drawing_galaxy: Optional[DrawingGalaxy] = None,
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
        # Sovereign embedding policy:
        # - compute: legacy behavior (compute missing embeddings on demand)
        # - skip: do not compute during hot path; skip embedding-based rerank if missing
        # - fail: fail-fast when missing embeddings are detected
        self.embedding_lazy_mode = os.getenv("K3D_ARC_EMBEDDING_LAZY_MODE", "compute").strip().lower()
        self.cosine_bridge = cosine_bridge or CosineSimilarityBridge()
        self.drawing_galaxy = drawing_galaxy

    def _exec_rpn(self, grid: Sequence[Sequence[int]], program: str) -> Optional[List[List[int]]]:
        """Execute RPN program via executor; return None on failure."""
        try:
            return self.executor.execute(grid, program)
        except Exception:
            return None

    def _detect_size_pattern(self, train_examples: List[Dict]) -> str:
        """Detect whether outputs shrink/expand relative to inputs."""
        ratios: List[Tuple[float, float]] = []
        for ex in train_examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])
            if not inp or not out or not inp[0] or not out[0]:
                continue
            h_ratio = len(out) / max(1, len(inp))
            w_ratio = len(out[0]) / max(1, len(inp[0]))
            ratios.append((h_ratio, w_ratio))
        if not ratios:
            return "same"
        avg_h = sum(r[0] for r in ratios) / len(ratios)
        avg_w = sum(r[1] for r in ratios) / len(ratios)
        if avg_h < 0.6 and avg_w < 0.6:
            return "extract"
        if avg_h > 1.5 and avg_w > 1.5:
            return "expand"
        return "same"

    def _bbox_nonzero(self, grid: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
        """Compute bbox of non-zero cells (inclusive)."""
        min_y = len(grid)
        min_x = len(grid[0]) if grid and grid[0] else 0
        max_y = -1
        max_x = -1
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                if val != 0:
                    if y < min_y:
                        min_y = y
                    if x < min_x:
                        min_x = x
                    if y > max_y:
                        max_y = y
                    if x > max_x:
                        max_x = x
        if max_y < 0:
            return 0, 0, 0, 0
        return min_y, min_x, max_y, max_x

    def _generate_extraction_candidates(self, input_grid: Sequence[Sequence[int]]) -> List[Candidate]:
        """Generate candidates focused on cropping/extraction."""
        candidates: List[Candidate] = []
        min_y, min_x, max_y, max_x = self._bbox_nonzero(input_grid)
        if max_y >= min_y and max_x >= min_x:
            h = max_y - min_y + 1
            w = max_x - min_x + 1
            rpn = f"{min_y} {min_x} {h} {w} CROP"
            out = self._exec_rpn(input_grid, rpn)
            if out:
                candidates.append((out, "crop_nonzero", rpn))
        # Dominant non-zero color bbox
        color_counts = {}
        for row in input_grid:
            for val in row:
                if val != 0:
                    color_counts[val] = color_counts.get(val, 0) + 1
        if color_counts:
            color = max(color_counts, key=color_counts.get)
            rpn = f"{color} EXTRACT_BBOX"
            out = self._exec_rpn(input_grid, rpn)
            if out:
                candidates.append((out, "extract_bbox", rpn))
        return candidates

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

        # Size-aware extraction/expansion heuristics.
        size_pattern = self._detect_size_pattern(train_examples)
        if size_pattern == "extract":
            extraction = self._generate_extraction_candidates(input_grid)
            if extraction:
                print(f"  [EXTRACT GEN] Generated {len(extraction)} extraction candidates")
            candidates.extend(extraction)

        # 2) Semantic-guided candidates: use word hints to expand search space.
        # SOVEREIGN: Closes the semantic layer → generation feedback loop.
        if semantic_hints:
            semantic_hints = self._expand_semantic_hints_with_neighbors(semantic_hints)
            semantic_candidates = self._generate_semantic_guided_candidates(input_grid, semantic_hints)
            print(f"  [SEMANTIC GEN] Generated {len(semantic_candidates)} semantic-guided candidates from {len(semantic_hints)} hints")
            # Dedup semantic early to reduce redundancy
            seen_sem: set[Tuple[str, int]] = set()
            for grid, instr, prog in semantic_candidates:
                h = self._hash_grid(grid)
                key = (prog, h)
                if key in seen_sem:
                    continue
                seen_sem.add(key)
                candidates.append((grid, instr, prog))
        else:
            print(f"  [SEMANTIC GEN] No semantic hints provided, skipping semantic-guided generation")

        # 3) Primitive search: rotations, flips, recolors, simple translations.
        candidates.extend(self._generate_primitive_candidates(input_grid))

        # 4) Compositions: simple rotate/flip + recolor combos.
        candidates.extend(self._generate_composition_candidates(input_grid))

        # 5) Math-style patterns: checkerboard even/odd fills.
        candidates.extend(self._generate_math_candidates(input_grid))

        # 6) Scale-invariant primitives registered in Drawing Galaxy.
        scale_inv_candidates = self._generate_scale_invariant_candidates(input_grid)
        candidates.extend(scale_inv_candidates)
        if scale_inv_candidates:
            print(
                f"  [SCALE-INV GEN] Generated {len(scale_inv_candidates)} candidate(s) from scale-invariant primitives"
            )

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

        missing_embedding_count = len(missing_grids)
        if missing_embedding_count:
            if self.embedding_lazy_mode == "compute":
                print(f"  [GALAXY LAZY] Computing {missing_embedding_count} missing embeddings (batch GPU)")
                batch_embeddings = self.processor._grid_to_spatial_embedding_batch(missing_grids)
                for h, emb in zip(missing_hashes, batch_embeddings):
                    self.embedding_galaxy[h] = emb
            elif self.embedding_lazy_mode == "fail":
                raise RuntimeError(
                    "SOVEREIGNTY VIOLATION: missing embeddings during ARC hot path "
                    f"(count={missing_embedding_count}). Precompute embeddings before benchmark execution."
                )
            elif self.embedding_lazy_mode == "skip":
                # Explicitly avoid lazy embedding computation in benchmark runtime.
                expected_output = None
            else:
                raise RuntimeError(
                    f"Invalid K3D_ARC_EMBEDDING_LAZY_MODE={self.embedding_lazy_mode!r}; "
                    "expected one of: compute|skip|fail"
                )

        # Deduplicate by output grid content.
        deduped = self._deduplicate_candidates(candidates)

        # Semantic ranking using sovereign embeddings when expected output is available.
        if expected_output is not None and deduped:
            deduped = self._rank_by_similarity(deduped, expected_output)

        # Cap the list.
        return deduped[: self.max_candidates]

    def _expand_semantic_hints_with_neighbors(self, semantic_hints: List[str], neighbor_depth: int = 1) -> List[str]:
        vocab = SemanticVocabulary()
        vocab_words = list(vocab.words.keys())
        word_index = {w: i for i, w in enumerate(vocab_words)}

        def neighbor(word: str, direction: int) -> str:
            if word not in word_index:
                return word
            idx = word_index[word] + direction
            idx = max(0, min(len(vocab_words) - 1, idx))
            return vocab_words[idx]

        expanded: List[str] = []
        for hint in semantic_hints:
            words = hint.split()
            if not words:
                continue
            patterns = [
                [+1] * len(words),
                [-1] * len(words),
                [(-1) ** i for i in range(len(words))],
                [(-1) ** (i + 1) for i in range(len(words))],
            ]
            for pattern in patterns:
                varied = [neighbor(w, d) for w, d in zip(words, pattern)]
                expanded.append(" ".join(varied))
        expanded.extend(semantic_hints)
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for h in expanded:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        print(f"  [SEMANTIC EXPAND] Expanded hints to {len(uniq)} variations")
        return uniq

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
    # Scale-invariant primitives
    # ------------------------------------------------------------------ #
    def _generate_scale_invariant_candidates(self, grid: Sequence[Sequence[int]]) -> List[Candidate]:
        if not self.drawing_galaxy:
            return []

        param_library: Dict[str, List[Dict[str, float]]] = {
            "REL_LINE": [
                {"x0_frac": 0.0, "y0_frac": 0.0, "x1_frac": 1.0, "y1_frac": 1.0},
                {"x0_frac": 0.0, "y0_frac": 1.0, "x1_frac": 1.0, "y1_frac": 0.0},
                {"x0_frac": 0.0, "y0_frac": 0.5, "x1_frac": 1.0, "y1_frac": 0.5},
                {"x0_frac": 0.5, "y0_frac": 0.0, "x1_frac": 0.5, "y1_frac": 1.0},
            ],
            "REL_RECT": [
                {"x_frac": 0.0, "y_frac": 0.0, "w_frac": 1.0, "h_frac": 1.0},
                {"x_frac": 0.25, "y_frac": 0.25, "w_frac": 0.5, "h_frac": 0.5},
                {"x_frac": 0.1, "y_frac": 0.1, "w_frac": 0.8, "h_frac": 0.8},
            ],
            "PROPORTIONAL_GRID": [
                {"rows": 2, "cols": 2},
                {"rows": 3, "cols": 3},
                {"rows": 4, "cols": 4},
            ],
            "FLOOD_FILL_REL": [
                {"x_frac": 0.5, "y_frac": 0.5},
                {"x_frac": 0.25, "y_frac": 0.25},
                {"x_frac": 0.75, "y_frac": 0.75},
            ],
        }

        candidates: List[Candidate] = []
        for shape_id, item in self.drawing_galaxy.shapes.items():
            payload = item.payload if isinstance(item, DrawingItem) else item
            if payload.get("type") != "scale_invariant":
                continue
            template = payload.get("visual_rpn")
            if not template:
                composition = payload.get("procedural_programs", {}).get("composition")
                template = composition
            if not template:
                continue
            param_sets = param_library.get(shape_id, [])
            if not param_sets:
                continue
            for params in param_sets:
                try:
                    rpn_program = template.format(**params)
                except Exception:
                    continue
                output = self._exec_rpn(grid, rpn_program)
                if output is None:
                    output = self._render_scale_primitive(shape_id, params, grid)
                if output is None:
                    continue
                candidates.append((output, f"[SCALE] {shape_id}", rpn_program))
        return candidates

    def _render_scale_primitive(
        self,
        shape_id: str,
        params: Dict[str, float],
        grid: Sequence[Sequence[int]],
    ) -> Optional[List[List[int]]]:
        if not grid or not grid[0]:
            return None
        height = len(grid)
        width = len(grid[0])
        canvas = [list(row) for row in grid]
        color = self._dominant_color(canvas)
        if shape_id == "REL_LINE":
            x0 = int(params.get("x0_frac", 0.0) * max(0, width - 1))
            y0 = int(params.get("y0_frac", 0.0) * max(0, height - 1))
            x1 = int(params.get("x1_frac", 1.0) * max(0, width - 1))
            y1 = int(params.get("y1_frac", 1.0) * max(0, height - 1))
            self._draw_line(canvas, x0, y0, x1, y1, color)
        elif shape_id == "REL_RECT":
            x = int(params.get("x_frac", 0.0) * width)
            y = int(params.get("y_frac", 0.0) * height)
            w = max(1, int(params.get("w_frac", 1.0) * width))
            h = max(1, int(params.get("h_frac", 1.0) * height))
            self._fill_rect(canvas, x, y, w, h, color)
        elif shape_id == "PROPORTIONAL_GRID":
            rows = max(1, int(params.get("rows", 2)))
            cols = max(1, int(params.get("cols", 2)))
            self._draw_grid(canvas, rows, cols, color)
        elif shape_id == "FLOOD_FILL_REL":
            x = int(params.get("x_frac", 0.5) * max(0, width - 1))
            y = int(params.get("y_frac", 0.5) * max(0, height - 1))
            self._flood_fill(canvas, x, y, color)
        else:
            return None
        return canvas

    def _dominant_color(self, grid: Sequence[Sequence[int]]) -> int:
        counts: Dict[int, int] = {}
        for row in grid:
            for val in row:
                counts[val] = counts.get(val, 0) + 1
        counts.pop(0, None)
        if not counts:
            return 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _draw_line(self, canvas: List[List[int]], x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if 0 <= y0 < len(canvas) and 0 <= x0 < len(canvas[0]):
                canvas[y0][x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _fill_rect(self, canvas: List[List[int]], x: int, y: int, w: int, h: int, color: int) -> None:
        for yy in range(y, min(y + h, len(canvas))):
            for xx in range(x, min(x + w, len(canvas[0]))):
                canvas[yy][xx] = color

    def _draw_grid(self, canvas: List[List[int]], rows: int, cols: int, color: int) -> None:
        h = len(canvas)
        w = len(canvas[0])
        if rows <= 0 or cols <= 0:
            return
        row_step = max(1, h // rows)
        col_step = max(1, w // cols)
        for r in range(0, h, row_step):
            for c in range(w):
                canvas[r][c] = color
        for c in range(0, w, col_step):
            for r in range(h):
                canvas[r][c] = color

    def _flood_fill(self, canvas: List[List[int]], x: int, y: int, color: int) -> None:
        h = len(canvas)
        w = len(canvas[0])
        if not (0 <= x < w and 0 <= y < h):
            return
        target = canvas[y][x]
        if target == color:
            canvas[y][x] = color
            return
        stack = [(x, y)]
        visited = set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if canvas[cy][cx] != target:
                continue
            canvas[cy][cx] = color
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

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
