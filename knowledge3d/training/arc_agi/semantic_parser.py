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

        # Optional normalizer to handle slang/typos (via grammar galaxy variants).
        if normalizer is None:
            try:
                from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
                self.normalizer = GrammarNormalizer()
            except Exception:
                self.normalizer = None
        else:
            self.normalizer = normalizer

        # Optional phrase→grammar rule map for quick hits (ARC-like patterns or scene descriptions).
        self._phrase_rule_map: Dict[str, str] = {
            "describe the scene": "en_visual_description",
            "describe the grid": "en_visual_description",
            "describe the video": "en_video_scene_graph",
            "describe the audio": "en_audio_description",
        }

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

        spatial_parsers = [
            self._parse_move_instruction,
            self._parse_fill_instruction,
            self._parse_rotate_instruction,
            self._parse_flip_instruction,
            self._parse_continue_instruction,
            self._parse_copy_instruction,
            self._parse_recolor_instruction,
        ]

        for parser in spatial_parsers:
            result = parser(text)
            if result:
                return result

        grammar_result = self._parse_via_grammar(text)
        if grammar_result:
            return grammar_result

        raise ValueError(f"Could not parse instruction: {instruction}")

    # ------------------------------------------------------------------ #
    # Spatial parsers
    # ------------------------------------------------------------------ #
    def _parse_move_instruction(self, text: str) -> Dict | None:
        """
        Parse move/translate instructions.

        Examples:
            "Move the red object to the bottom-right corner"
            "Move red square to center"
            "Translate blue left"
        """
        # Move to explicit position
        m = re.search(r"(move|translate)\s+(?:the\s+)?(\w+)(?:\s+(\w+))?\s+(?:to\s+)?(?:the\s+)?([\w-]+)", text)
        if m:
            _, token_a, token_b, dest = m.groups()
            obj = self._build_object(token_a, token_b)
            if obj:
                return {
                    "action": "move",
                    "object": obj,
                    "destination": {"position": dest, "type": "position"},
                }

        # Move in direction by one step (optionally with count)
        m = re.search(r"(move|translate)\s+(?:the\s+)?(\w+)(?:\s+(\w+))?\s+(left|right|up|down)(?:\s+by\s+(\d+))?", text)
        if m:
            _, token_a, token_b, direction, steps = m.groups()
            obj = self._build_object(token_a, token_b)
            steps_int = int(steps) if steps else 1
            if obj:
                return {
                    "action": "move",
                    "object": obj,
                    "direction": direction,
                    "steps": steps_int,
                }

        return None

    def _parse_fill_instruction(self, text: str) -> Dict | None:
        """
        Parse fill/paint instructions.

        Examples:
            "Fill the largest rectangle with blue"
            "Fill center with red"
            "Paint the square blue"
        """
        m = re.search(r"(fill|paint)\s+(?:the\s+)?(?:(\w+)\s+)?(?:(\w+)\s+)?(?:with\s+)?(\w+)", text)
        if not m:
            return None

        _, mod1, mod2, color_token = m.groups()
        obj: Dict[str, str] = {}
        for mod in (mod1, mod2):
            if not mod:
                continue
            if mod in self.sizes:
                obj["size"] = mod
            elif mod in self.shapes:
                obj["shape"] = mod
            elif mod in self.spatial:
                obj["position"] = mod

        if not obj:
            obj["type"] = "region"

        if color_token not in self.colors:
            return None

        return {
            "action": "fill",
            "object": obj,
            "color": color_token,
        }

    def _parse_rotate_instruction(self, text: str) -> Dict | None:
        """
        Parse rotation/turn instructions.

        Examples:
            "Rotate the pattern 90 degrees clockwise"
            "Rotate 180 degrees"
            "Turn the grid clockwise"
        """
        m = re.search(r"(rotate|turn)\s+(?:the\s+)?(\w+)?\s*(\d+)?\s*(?:degrees?)?\s*(\w+)?", text)
        if not m:
            return None

        _, obj_token, angle_token, direction_token = m.groups()
        angle = int(angle_token) if angle_token else 90
        direction = direction_token if direction_token in self.spatial else "counterclockwise"
        result: Dict[str, object] = {"action": "rotate", "angle": angle, "direction": direction}

        if obj_token and obj_token not in {"the", "it", "pattern", "grid"}:
            result["object"] = obj_token
        else:
            result["object"] = "pattern"

        return result

    def _parse_flip_instruction(self, text: str) -> Dict | None:
        """
        Parse flip/mirror instructions.

        Examples:
            "Flip the pattern vertically"
            "Flip horizontally"
            "Mirror the grid"
        """
        m = re.search(r"(flip|mirror)\s+(?:the\s+)?(\w+)?\s*(vertical|horizontal|vert|horiz|vertically|horizontally)?", text)
        if not m:
            return None

        _, obj_token, axis_token = m.groups()
        axis = "horizontal"
        if axis_token:
            if "vert" in axis_token:
                axis = "vertical"
            elif "horiz" in axis_token:
                axis = "horizontal"

        result: Dict[str, object] = {"action": "flip", "axis": axis}
        if obj_token and obj_token not in {"pattern", "grid", "it", "the"}:
            result["object"] = obj_token
        return result

    def _parse_continue_instruction(self, text: str) -> Dict | None:
        """
        Parse continuation/extension instructions.

        Examples:
            "Continue the sequence to the right"
            "Extend the pattern downward"
            "Repeat to the left"
        """
        m = re.search(r"(continue|extend|repeat)\s+(?:the\s+)?(\w+)?\s+(?:to\s+)?(?:the\s+)?(\w+)", text)
        if not m:
            return None

        _, obj_token, direction = m.groups()
        if direction not in self.spatial:
            return None

        result: Dict[str, object] = {"action": "continue", "direction": direction}
        if obj_token and obj_token not in {"sequence", "pattern", "it", "the"}:
            result["object"] = obj_token
        else:
            result["object"] = "sequence"
        return result

    def _parse_copy_instruction(self, text: str) -> Dict | None:
        """
        Parse copy/duplicate instructions.

        Examples:
            "Copy the red object"
            "Duplicate the pattern to bottom-right"
        """
        m = re.search(r"(copy|duplicate)\s+(?:the\s+)?(\w+)?\s*(\w+)?(?:\s+to\s+(?:the\s+)?([\w-]+))?", text)
        if not m:
            return None

        _, token_a, token_b, dest = m.groups()
        obj = self._build_object(token_a, token_b)
        if not obj:
            return None

        result: Dict[str, object] = {"action": "copy", "object": obj}
        if dest:
            result["destination"] = {"position": dest, "type": "position"}
        return result

    def _parse_recolor_instruction(self, text: str) -> Dict | None:
        """
        Parse recolor/paint instructions.

        Examples:
            "Change red to blue"
            "Replace green with yellow"
        """
        m = re.search(r"(change|replace|recolor|paint|color|colour)\s+(\w+)\s+(?:to|with)\s+(\w+)", text)
        if not m:
            return None

        _, src_token, dst_token = m.groups()
        if src_token not in self.colors or dst_token not in self.colors:
            return None

        return {
            "action": "recolor",
            "source_color": src_token,
            "target_color": dst_token,
        }

    # ------------------------------------------------------------------ #
    # Grammar fallback
    # ------------------------------------------------------------------ #
    def _parse_via_grammar(self, text: str) -> Dict | None:
        """Fallback: map to Grammar Galaxy rules using multiple matching strategies."""
        exact = next(((rid, surface) for surface, rid in self._grammar_surfaces if surface == text), None)
        if exact:
            rid, surface = exact
            ctx = self._find_example_context(rid, surface)
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        surfaces = [s for s, _ in self._grammar_surfaces]
        if surfaces:
            best = difflib.get_close_matches(text, surfaces, n=1, cutoff=0.8)
            if best:
                surface = best[0]
                rid = next(r for s, r in self._grammar_surfaces if s == surface)
                ctx = self._find_example_context(rid, surface)
                return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        rid_ctx = self._best_overlap_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        rid_ctx = self._best_cosine_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        rid_ctx = self._best_dense_rule(text)
        if rid_ctx:
            rid, ctx = rid_ctx
            return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        for phrase, rid in self._phrase_rule_map.items():
            if phrase in text:
                ctx = self._find_example_context(rid, None)
                return {"action": "grammar_rule", "rule_id": rid, "context": ctx}

        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_object(self, token_a: Optional[str], token_b: Optional[str]) -> Dict[str, str] | None:
        """Build an object dict from up to two tokens (color/shape/size)."""
        obj: Dict[str, str] = {}
        for tok in (token_a, token_b):
            if not tok:
                continue
            if tok in self.colors and "color" not in obj:
                obj["color"] = tok
                obj.setdefault("type", "object")
            elif tok in self.shapes and "shape" not in obj:
                obj["shape"] = tok
                obj.setdefault("type", "shape")
            elif tok in self.sizes and "size" not in obj:
                obj["size"] = tok

        if not obj and token_a:
            obj["type"] = "object"
        return obj if obj else None

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
        if best_score >= 0.5 and best:
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
