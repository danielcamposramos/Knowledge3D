"""Semantic parser: natural language instruction → semantic representation."""

from __future__ import annotations

import re
import difflib
from typing import Dict, Optional, List, Tuple

import numpy as np

from .semantic_primitives import (
    SPATIAL_SEMANTICS,
    COLOR_SEMANTICS,
    SHAPE_SEMANTICS,
    SIZE_SEMANTICS,
    ACTION_SEMANTICS,
)
from .grammar_galaxy import GrammarGalaxy
from .grammar_executor import GrammarRPNExecutor


class SemanticParser:
    """Parse natural language instructions to semantic representations."""

    def __init__(self, language: str = "en", normalizer: Optional[object] = None, grammar: Optional[GrammarGalaxy] = None):
        self.spatial = SPATIAL_SEMANTICS
        self.colors = COLOR_SEMANTICS
        self.shapes = SHAPE_SEMANTICS
        self.sizes = SIZE_SEMANTICS
        self.actions = ACTION_SEMANTICS
        self.language = language
        self.grammar = grammar or GrammarGalaxy()
        self.grammar_exec = GrammarRPNExecutor()

        # Pre-compile simple regex patterns for fast matching.
        self._patterns = [
            ("move_to_position", re.compile(r"move the (\w+)\s+object to the ([\w-]+)(?:\s+corner)?")),
            ("move_direction", re.compile(r"move the (\w+)\s+object\s+(left|right|up|down)")),
            ("fill_shape_color", re.compile(r"fill the (largest|smallest)?\s*(\w+)?\s*with\s+(\w+)")),
            ("rotate_pattern", re.compile(r"rotate .*? (90|180|270) degrees(?:\s+(clockwise|counterclockwise))?")),
            ("continue_direction", re.compile(r"continue .*? to the (right|left|up|down)")),
            ("copy_to_position", re.compile(r"copy (?:the\s+)?(\w+)\s+object to (?:the\s+)?([\w-]+)(?:\s+corner)?")),
            ("flip_axis", re.compile(r"(flip|mirror) .*?(horizontally|vertically)")),
            ("recolor", re.compile(r"(change|replace|recolor|paint)\s+(\w+)\s+(?:to|with)\s+(\w+)")),
        ]

        # Optional phrase→grammar rule map for quick hits (ARC-like patterns or scene descriptions).
        self._phrase_rule_map: Dict[str, str] = {
            "describe the scene": "en_visual_description",
            "describe the grid": "en_visual_description",
            "describe the video": "en_video_scene_graph",
            "describe the audio": "en_audio_description",
        }

        # Optional normalizer to handle slang/typos (via grammar galaxy variants).
        if normalizer is None:
            try:
                from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
                self.normalizer = GrammarNormalizer()
            except Exception:
                self.normalizer = None
        else:
            self.normalizer = normalizer

        # Cache of grammar example surfaces for fuzzy/embedding-like matching.
        self._grammar_surfaces: List[Tuple[str, str]] = []
        self._grammar_vectors: List[Tuple[str, Dict[str, int]]] = []
        self._grammar_embs: List[Tuple[str, np.ndarray]] = []
        for rule in self.grammar.list_rules(language=self.language):
            for ex in rule.examples:
                surface = " ".join(ex.values()).lower()
                self._grammar_surfaces.append((surface, rule.rule_id))
                self._grammar_vectors.append((rule.rule_id, self._embed_bow(surface)))
                self._grammar_embs.append((rule.rule_id, self._embed_dense(surface)))

    def parse(self, instruction: str) -> Dict:
        """
        Parse instruction to semantic structure.

        Args:
            instruction: Natural language instruction

        Returns:
            Semantic representation dictionary
        """
        text = instruction.strip().lower()

        # Normalize slang/typos if normalizer is available.
        if self.normalizer is not None:
            try:
                text = self.normalizer.normalize_text(text, self.language)
            except Exception:
                pass

        # Try grammar-galaxy phrase matches (exact then fuzzy).
        exact = next(((rid, surface) for surface, rid in self._grammar_surfaces if surface == text), None)
        if exact:
            rid, surface = exact
            ctx = self._find_example_context(rid, surface)
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Fuzzy match (closest example).
        surfaces = [s for s, _ in self._grammar_surfaces]
        if surfaces:
            best = difflib.get_close_matches(text, surfaces, n=1, cutoff=0.8)
            if best:
                surface = best[0]
                rid = next(r for s, r in self._grammar_surfaces if s == surface)
                ctx = self._find_example_context(rid, surface)
                return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Token-overlap heuristic: choose example with highest token overlap.
        rid_ctx = self._best_overlap_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Bag-of-words cosine similarity to grammar examples.
        rid_ctx = self._best_cosine_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Dense embedding similarity (placeholder: simple hashing to vector).
        rid_ctx = self._best_dense_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Phrase map fallback.
        for phrase, rid in self._phrase_rule_map.items():
            if phrase in text:
                ctx = self._find_example_context(rid, None)
                return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        # Pattern 1: Move [color] object to [position]
        m = self._match("move_to_position", text)
        if m:
            color_token, dest_token = m.groups()
            color = self._lookup_color(color_token)
            dest = dest_token
            return {
                "action": "move",
                "object": {"color": color, "type": "object"},
                "destination": {"position": dest, "type": "position"},
            }

        # Pattern 2: Fill [size] [shape] with [color]
        m = self._match("fill_shape_color", text)
        if m:
            size_token, shape_token, color_token = m.groups()
            size = self._lookup_size(size_token)
            shape = self._lookup_shape(shape_token)
            color = self._lookup_color(color_token)
            return {
                "action": "fill",
                "object": {"shape": shape, "size": size},
                "color": color,
            }

        # Pattern 3: Rotate ... [angle] degrees [direction]
        m = self._match("rotate_pattern", text)
        if m:
            angle_token, direction_token = m.groups()
            angle = int(angle_token)
            direction = direction_token or "counterclockwise"
            return {
                "action": "rotate",
                "object": "pattern",
                "angle": angle,
                "direction": direction,
            }

        # Pattern 4: Continue the sequence to the [direction]
        m = self._match("continue_direction", text)
        if m:
            direction_token = m.group(1)
            return {
                "action": "continue",
                "object": "sequence",
                "direction": direction_token,
            }

        # Pattern 5: Copy object to position
        m = self._match("copy_to_position", text)
        if m:
            color_token, dest_token = m.groups()
            color = self._lookup_color(color_token)
            return {
                "action": "copy",
                "object": {"color": color, "type": "object"},
                "destination": {"position": dest_token, "type": "position"},
            }

        # Pattern 6: Move by direction (step = 1)
        m = self._match("move_direction", text)
        if m:
            color_token, direction_token = m.groups()
            color = self._lookup_color(color_token)
            return {
                "action": "move",
                "object": {"color": color, "type": "object"},
                "direction": direction_token,
                "steps": 1,
            }

        # Pattern 7: Flip / mirror horizontally or vertically
        m = self._match("flip_axis", text)
        if m:
            _, axis_token = m.groups()
            axis = "horizontal" if "horiz" in axis_token else "vertical"
            return {
                "action": "flip",
                "axis": axis,
            }

        # Pattern 8: Recolor
        m = self._match("recolor", text)
        if m:
            _, src_token, dst_token = m.groups()
            src = self._lookup_color(src_token)
            dst = self._lookup_color(dst_token)
            return {
                "action": "recolor",
                "source_color": src,
                "target_color": dst,
            }

        raise ValueError(f"Could not parse instruction: {instruction}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _lookup_color(self, token: Optional[str]) -> str:
        if token and token in self.colors:
            return token
        raise ValueError(f"Unknown color token: {token}")

    def _lookup_shape(self, token: Optional[str]) -> str:
        if token and token in self.shapes:
            return token
        # Default to rectangle if unspecified shape
        return "rectangle"

    def _lookup_size(self, token: Optional[str]) -> str:
        if token and token in self.sizes:
            return token
        # Default size when not specified
        return "largest"

    def _match(self, key: str, text: str):
        for name, pattern in self._patterns:
            if name == key:
                return pattern.search(text)
        return None

    def _find_example_context(self, rule_id: str, surface: str | None) -> Dict[str, str]:
        """Find example dict for a given rule and surface string (or first example)."""
        for rule in self.grammar.list_rules(language=self.language):
            if rule.rule_id != rule_id:
                continue
            for ex in rule.examples:
                if surface is None or " ".join(ex.values()).lower() == surface:
                    return ex
        return {}

    def _best_overlap_rule(self, text: str) -> Tuple[str, Dict[str, str]] | None:
        """Pick the rule/example with highest token overlap against input text."""
        tokens = set(text.split())
        best_score = 0.0
        best = None
        for rule in self.grammar.list_rules(language=self.language):
            for ex in rule.examples:
                surface = " ".join(ex.values()).lower()
                ex_tokens = set(surface.split())
                if not ex_tokens:
                    continue
                overlap = len(tokens & ex_tokens) / len(ex_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best = (rule.rule_id, ex)
        if best_score >= 0.5:
            rid, ctx = best
            return rid, ctx
        return None

    def _embed_bow(self, text: str) -> Dict[str, int]:
        vec: Dict[str, int] = {}
        for tok in text.split():
            vec[tok] = vec.get(tok, 0) + 1
        return vec

    def _cosine(self, v1: Dict[str, int], v2: Dict[str, int]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in v1.keys())
        n1 = sum(v * v for v in v1.values()) ** 0.5
        n2 = sum(v * v for v in v2.values()) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def _best_cosine_rule(self, text: str) -> Tuple[str, Dict[str, str]] | None:
        vec = self._embed_bow(text)
        best_score = 0.0
        best_rid = None
        for rid, rvec in self._grammar_vectors:
            score = self._cosine(vec, rvec)
            if score > best_score:
                best_score = score
                best_rid = rid
        if best_score >= 0.6 and best_rid:
            ctx = self._find_example_context(best_rid, None)
            return best_rid, ctx
        return None

    def _embed_dense(self, text: str, dim: int = 32) -> np.ndarray:
        """Very lightweight hashing embed (placeholder for real embeddings)."""
        vec = np.zeros(dim, dtype=np.float32)
        for tok in text.split():
            h = hash(tok) % dim
            vec[h] += 1.0
        # L2 norm
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _best_dense_rule(self, text: str) -> Tuple[str, Dict[str, str]] | None:
        vec = self._embed_dense(text)
        best_score = 0.0
        best_rid = None
        for rid, rvec in self._grammar_embs:
            score = float(np.dot(vec, rvec))
            if score > best_score:
                best_score = score
                best_rid = rid
        if best_score >= 0.7 and best_rid:
            ctx = self._find_example_context(best_rid, None)
            return best_rid, ctx
        return None


__all__ = ["SemanticParser"]
