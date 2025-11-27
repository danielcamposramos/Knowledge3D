"""
Sovereign TRM router for ARC-AGI: bridges Drawing + Grammar galaxies.

Key responsibilities:
- Convert ARC grids into Drawing Galaxy RPN programs (atomic visual view).
- Produce task signatures for heuristic routing.
- Suggest Drawing+Grammar combinations using sovereign components
  (MatryoshkaTRM + SelfUpdatingAdapter + RPNMathCore) without external ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

# SOVEREIGN: No numpy in hot path! Removed numpy import.
# TODO: MatryoshkaTRM should use RPNMathCore instead of numpy for matrix ops

from knowledge3d.cranium.matryoshka_trm import MatryoshkaTRM
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter
from knowledge3d.cranium.ptx_runtime.rpn_math_core import RPNMathCore
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, GrammarRule
from knowledge3d.training.arc_agi.semantic_context import SemanticContext
from knowledge3d.training.arc_agi.sovereign_utils import l2_norm, pad_or_truncate


class SovereignTRMRouter:
    """Heuristic router that keeps all computation sovereign (no torch)."""

    def __init__(
        self,
        drawing_galaxy: DrawingGalaxy,
        grammar_galaxy: GrammarGalaxy,
        shadow_copy=None,
        matryoshka_dim: int = 512,
    ):
        self.drawing = drawing_galaxy
        self.grammar = grammar_galaxy
        self.matryoshka_dim = int(matryoshka_dim)
        self.semantic_context: SemanticContext | None = None
        if shadow_copy is not None and hasattr(shadow_copy, "semantic_context"):
            self.semantic_context = shadow_copy.semantic_context

        # Sovereign components (instantiated, not trained here)
        self.base_trm = MatryoshkaTRM(max_dims=self.matryoshka_dim, min_dims=64)
        # Keep adapter square to satisfy AdapterWeights shape requirements; rule count
        # is handled heuristically for now (adapter can be wired later when GPU toolchain is present).
        self.router_adapter = SelfUpdatingAdapter(
            shape=(self.matryoshka_dim, self.matryoshka_dim),
            rank=64,
            specialist_name="arc_router",
        )
        self.math_core = RPNMathCore()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def grid_to_drawing_rpn(self, grid: Sequence[Sequence[int]]) -> str:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        parts: List[str] = [f"GRID {rows} {cols}"]
        for r in range(rows):
            for c in range(cols):
                color = int(grid[r][c])
                if color != 0:
                    parts.append(f"CELL {r} {c} {color} FILL")
        return " ".join(parts)

    def task_signature(self, grid: Sequence[Sequence[int]]) -> Dict[str, int]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        colors = {}
        non_zero = 0
        for r in range(rows):
            for c in range(cols):
                val = int(grid[r][c])
                if val != 0:
                    non_zero += 1
                    colors[val] = colors.get(val, 0) + 1
        return {
            "grid_rows": rows,
            "grid_cols": cols,
            "num_colors": len(colors),
            "filled_cells": non_zero,
        }

    def embed_task(self, grid: Sequence[Sequence[int]]) -> List[float]:
        """Deterministic embedding: flatten grid stats → Matryoshka projection."""
        signature = self.task_signature(grid)
        flat_features = [
            float(signature["grid_rows"]),
            float(signature["grid_cols"]),
            float(signature["num_colors"]),
            float(signature["filled_cells"]),
        ]
        norm = l2_norm(flat_features) or 1.0
        flat_features = [v / norm for v in flat_features]
        padded = pad_or_truncate(flat_features, self.matryoshka_dim, 0.0)
        projected = self.base_trm.project_vector(padded, target_dim=self.matryoshka_dim)
        return [float(v) for v in projected]

    class _AttrDict(dict):
        """Dict with attribute access for compatibility with legacy tests."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    def route(self, grid: Sequence[Sequence[int]], top_k: int = 27, use_semantics: bool = True) -> List[Dict[str, Any]]:  # SOVEREIGN: Tesla 3-6-9 (increased from 3)
        drawing_program = self.grid_to_drawing_rpn(grid)
        signature = self.task_signature(grid)
        # embedding = self.embed_task(grid)  # TODO: Not currently used, remove to avoid numpy

        candidates: List[Dict[str, Any]] = []

        # Semantic matches (prioritized)
        if use_semantics and self.semantic_context is not None:
            try:
                # SOVEREIGN FIX: Pass grid as-is (no numpy conversion!)
                matches = self.semantic_context.find_matching_contexts(grid, top_k=top_k * 3)  # SOVEREIGN: Tesla 3-6-9 (increased from top_k*2)
                for ctx in matches:
                    candidates.append(
                        {
                            "program": ctx["program"],
                            "program_type": "semantic",
                            "source": "semantic_match",
                            "score": ctx.get("score", 0.6),
                            "signature": signature,
                            "semantic_context": ctx,
                        }
                    )
                print(f"  [SEMANTIC] Found {len(matches)} context matches")
            except Exception as e:
                print(f"  [SEMANTIC] Warning: semantic matching failed: {e}")

        # Grammar-driven candidates (existing path)
        for rule in self._rank_rules(top_k=top_k):
            candidates.append(
                {
                    "drawing_program": drawing_program,
                    "grammar_rule": rule,
                    "signature": signature,
                    "source": "grammar",
                    "score": rule_score_hint(rule, signature),
                }
            )

        # Deduplicate by program string when available
        unique: List[Dict[str, Any]] = []
        seen = set()
        for cand in candidates:
            prog = cand.get("program") or cand.get("grammar_rule").rpn_program if cand.get("grammar_rule") else None
            if prog and prog in seen:
                continue
            if prog:
                seen.add(prog)
            unique.append(cand)

        # Trim to top_k but keep semantic matches first
        semantic_first = [c for c in unique if c.get("source") == "semantic_match"]
        rest = [c for c in unique if c.get("source") != "semantic_match"]
        ordered = semantic_first + rest
        # Wrap to provide attribute access while retaining dict interface.
        return [self._AttrDict(c) for c in ordered[:top_k]]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def update_from_discoveries(self, shadow_copy, top_n: int = 100) -> Dict[str, int]:
        """
        Update router weights from discovered patterns in shadow copy.

        SOVEREIGN: This closes the discovery → routing feedback loop.
        High-quality discoveries should influence future routing decisions.

        Args:
            shadow_copy: DualShadowCopy with library of discoveries
            top_n: Number of recent high-quality discoveries to consider

        Returns:
            Dict with update statistics
        """
        if not hasattr(shadow_copy, 'library') or not shadow_copy.library:
            return {"processed": 0, "high_quality": 0}

        # Get recent high-quality discoveries (sorted by quality score)
        recent = sorted(shadow_copy.library, key=lambda e: e.get("quality_score", 0.0), reverse=True)[:top_n]
        high_quality = [e for e in recent if e.get("quality_score", 0.0) >= 0.75]

        # Update router_adapter with successful patterns
        # For now, track pattern frequencies to bias future routing
        pattern_counts: Dict[str, int] = {}
        for entry in high_quality:
            program = entry.get("program", "")
            # Extract pattern type from program (simplified heuristic)
            if "rotate" in program.lower():
                pattern_counts["rotation"] = pattern_counts.get("rotation", 0) + 1
            if "flip" in program.lower():
                pattern_counts["flip"] = pattern_counts.get("flip", 0) + 1
            if "recolor" in program.lower():
                pattern_counts["recolor"] = pattern_counts.get("recolor", 0) + 1
            if "translate" in program.lower():
                pattern_counts["translate"] = pattern_counts.get("translate", 0) + 1

        # TODO: When GPU toolchain is wired, update self.router_adapter weights
        # For now, store pattern preferences for heuristic ranking
        if not hasattr(self, '_pattern_prefs'):
            self._pattern_prefs: Dict[str, int] = {}
        for pattern, count in pattern_counts.items():
            self._pattern_prefs[pattern] = self._pattern_prefs.get(pattern, 0) + count

        return {
            "processed": len(recent),
            "high_quality": len(high_quality),
            "pattern_types": len(pattern_counts),
        }

    def _rank_rules(self, top_k: int) -> List[GrammarRule]:
        """
        Rank rules heuristically (no embedding needed for now).

        SOVEREIGN: Simple domain-based filtering + learned pattern preferences.
        TODO: Use RPN-based semantic similarity when embeddings are needed.
        """
        # Simple heuristic: prefer drawing/spatial domain rules, then others.
        drawing_rules = [r for r in self.grammar.list_rules() if getattr(r, "domain", "") in {"drawing", "spatial"}]
        other_rules = [r for r in self.grammar.list_rules() if r not in drawing_rules]

        # SOVEREIGN: Apply learned pattern preferences if available
        if hasattr(self, '_pattern_prefs') and self._pattern_prefs:
            def rule_priority(rule: GrammarRule) -> int:
                priority = 0
                rule_id_lower = rule.rule_id.lower()
                for pattern, count in self._pattern_prefs.items():
                    if pattern in rule_id_lower:
                        priority += count
                return priority

            drawing_rules = sorted(drawing_rules, key=rule_priority, reverse=True)
            other_rules = sorted(other_rules, key=rule_priority, reverse=True)

        ordered = drawing_rules + other_rules
        return ordered[: max(1, int(top_k * 2))]


def rule_score_hint(rule: GrammarRule, signature: Dict[str, int]) -> float:
    """Lightweight heuristic for ranking; keeps everything deterministic."""
    score = 0.1
    if getattr(rule, "domain", "") in {"drawing", "spatial"}:
        score += 0.4
    if signature.get("num_colors", 0) > 1 and "recolor" in rule.rule_id:
        score += 0.2
    if signature.get("filled_cells", 0) <= 4 and "primitive" in rule.pattern:
        score += 0.1
    return score


__all__ = ["SovereignTRMRouter", "RoutingCandidate"]
