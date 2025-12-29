"""Multimodal semantic parser: spatial + math + drawing + text."""

from __future__ import annotations

import re
from typing import Dict, Optional

from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, get_grammar_galaxy
from knowledge3d.training.arc_agi.grammar_normalizer import GrammarNormalizer
from knowledge3d.training.arc_agi.semantic_parser import SemanticParser


class MultimodalSemanticParser:
    """
    Parse instructions across spatial, math, drawing, and text domains.

    Priority:
        1) Spatial (ARC semantics)
        2) Math (conditions, symmetry, periodicity)
        3) Drawing (shapes/positions)
        4) Text (fallback grammar galaxy)
    """

    def __init__(self):
        self.galaxy = get_grammar_galaxy()
        self.normalizer = GrammarNormalizer(self.galaxy)
        self.spatial_parser = SemanticParser()

    def parse(self, instruction: str, debug: bool = False) -> Dict:
        """Parse instruction into a semantic dict with domain + action + params."""
        normalized = instruction
        try:
            normalized = self.normalizer.normalize_text(instruction, "en")
        except Exception:
            pass

        # 1) Spatial (highest priority)
        try:
            spatial = self._parse_spatial(normalized)
            if spatial and spatial.get("action") != "unknown":
                spatial["domain"] = "spatial"
                if debug:
                    print(f"[multimodal] domain=spatial instruction={instruction}")
                return spatial
        except Exception as e:
            if debug:
                print(f"[multimodal] spatial parse failed: {e}")

        # 2) Math
        math = self._parse_math(normalized)
        if math:
            math["domain"] = "math"
            if debug:
                print(f"[multimodal] domain=math instruction={instruction}")
            return math

        # 3) Drawing
        drawing = self._parse_drawing(normalized)
        if drawing:
            drawing["domain"] = "drawing"
            if debug:
                print(f"[multimodal] domain=drawing instruction={instruction}")
            return drawing

        # 4) Text fallback
        text = self._parse_text(normalized)
        if text:
            text["domain"] = "text"
            if debug:
                print(f"[multimodal] domain=text instruction={instruction}")
            return text

        return {"domain": "unknown", "action": "unknown", "instruction": instruction}

    # ------------------------------------------------------------------ #
    # Spatial
    # ------------------------------------------------------------------ #
    def _parse_spatial(self, instruction: str) -> Optional[Dict]:
        """Delegate to existing spatial semantic parser."""
        return self.spatial_parser.parse(instruction)

    # ------------------------------------------------------------------ #
    # Math
    # ------------------------------------------------------------------ #
    def _parse_math(self, instruction: str) -> Optional[Dict]:
        """Detect simple math-driven patterns."""
        # Conditional fill: row/col arithmetic
        if "where" in instruction and any(op in instruction for op in ["+", "-", "*", "×", "/"]):
            match = re.search(r"where\s+(.+?)\s+(is|are)\s+(even|odd|positive|negative)", instruction)
            if match:
                expression, _, condition = match.groups()
                return {
                    "action": "fill_conditional",
                    "expression": expression.strip(),
                    "condition": condition,
                }

        # Rotational symmetry by order
        if "symmetry" in instruction and "order" in instruction:
            match = re.search(r"order\s+(\d+)", instruction)
            if match:
                order = int(match.group(1))
                angle = 360 // max(order, 1)
                return {
                    "action": "check_symmetry",
                    "pattern": "rotational_symmetry",
                    "order": order,
                    "angle": angle,
                }

        # Periodic repeat
        if "repeat" in instruction and "every" in instruction:
            match = re.search(r"every\s+(\d+)", instruction)
            if match:
                period = int(match.group(1))
                return {
                    "action": "repeat_pattern",
                    "period": period,
                }

        # Dimensions (rows x cols)
        dim_match = re.search(r"(\d+)\s*[×x]\s*(\d+)", instruction)
        if dim_match:
            rows, cols = int(dim_match.group(1)), int(dim_match.group(2))
            return {
                "action": "grid_dimensions",
                "rows": rows,
                "cols": cols,
            }

        return None

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _parse_drawing(self, instruction: str) -> Optional[Dict]:
        """Detect drawing primitives and simple compositions."""
        shapes = ["square", "rectangle", "circle", "line", "diagonal", "cross"]
        for shape in shapes:
            if shape in instruction:
                position = None
                for pos in ["center", "top-left", "top-right", "bottom-left", "bottom-right"]:
                    if pos in instruction:
                        position = pos
                        break

                # Line with start/end
                if shape in ["line", "diagonal"]:
                    start_match = re.search(r"from\s+([a-z-]+)", instruction)
                    end_match = re.search(r"to\s+([a-z-]+)", instruction)
                    return {
                        "action": "draw_shape",
                        "shape": shape,
                        "start": start_match.group(1) if start_match else None,
                        "end": end_match.group(1) if end_match else None,
                    }

                return {
                    "action": "draw_shape",
                    "shape": shape,
                    "position": position,
                }

        if "pattern" in instruction and any(word in instruction for word in ["fill", "draw"]):
            return {"action": "fill_pattern", "pattern": "detect"}

        return None

    # ------------------------------------------------------------------ #
    # Text fallback
    # ------------------------------------------------------------------ #
    def _parse_text(self, instruction: str) -> Optional[Dict]:
        """Basic text understanding (placeholder)."""
        return {"action": "text_understanding", "instruction": instruction}


__all__ = ["MultimodalSemanticParser"]
