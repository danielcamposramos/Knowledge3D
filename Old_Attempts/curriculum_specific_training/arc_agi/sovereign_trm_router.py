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
from knowledge3d.training.arc_agi.router_scratchpad import RouterScratchpad
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter


class DomainSpecialist:
    """Lightweight specialist profile for domain-aware routing."""

    def __init__(self, name: str, domains: Sequence[str], weight: float = 1.0, adapter=None):
        self.name = name
        self.domains = set(domains)
        self.weight = float(weight)
        self.adapter = adapter  # Optional low-rank adapter for scoring
        # Deterministic ternary mask derived from name for cheap scoring
        self._ternary_mask = self._build_ternary_mask(name)
        # Domain-derived ternary mask (per-domain seed)
        self._domain_mask = self._build_domain_mask(domains)

    def _build_ternary_mask(self, seed: str) -> List[int]:
        vals = []
        for ch in seed[:4].ljust(4, "x"):
            h = (ord(ch) % 3) - 1  # maps to {-1,0,1}
            vals.append(h)
        return vals or [0, 0, 0, 0]

    def _build_domain_mask(self, domains: Sequence[str]) -> List[int]:
        vals = [0, 0, 0, 0]
        for idx, dom in enumerate(domains):
            if idx >= 4:
                break
            vals[idx] = (sum(ord(c) for c in dom) % 3) - 1
        return vals

    def _ternary_score(self, signature: Dict[str, int]) -> float:
        feats = [
            signature.get("grid_rows", 0),
            signature.get("grid_cols", 0),
            signature.get("num_colors", 0),
            signature.get("filled_cells", 0),
        ]
        # Quantize features to ternary {-1,0,1}
        quantized = []
        for v in feats:
            if v > 8:
                quantized.append(1)
            elif v == 0:
                quantized.append(0)
            else:
                quantized.append(-1 if v <= 2 else 0)
        base = sum(m * q for m, q in zip(self._ternary_mask, quantized))
        dom = sum(m * q for m, q in zip(self._domain_mask, quantized))
        return (base + dom) * 0.05

    def score(self, rule: GrammarRule, signature: Dict[str, int] | None = None) -> float:
        base = self.weight if getattr(rule, "domain", "") in self.domains else 0.0
        if signature is None:
            return base
        return base + self._ternary_score(signature)


class SovereignTRMRouter:
    """Heuristic router that keeps all computation sovereign (no torch)."""

    def __init__(
        self,
        drawing_galaxy: DrawingGalaxy,
        grammar_galaxy: GrammarGalaxy,
        shadow_copy=None,
        matryoshka_dim: int = 512,
        adapter_rank: int = 32,
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
        # Domain specialists (swappable adapters later)
        self.specialists: List[DomainSpecialist] = [
            DomainSpecialist(
                "spatial",
                ["drawing", "spatial"],
                weight=1.2,
                adapter=SelfUpdatingAdapter(
                    shape=(self.matryoshka_dim, self.matryoshka_dim),
                    rank=adapter_rank,
                    specialist_name="spatial_router",
                ),
            ),
            DomainSpecialist(
                "color",
                ["color", "recolor"],
                weight=0.8,
                adapter=SelfUpdatingAdapter(
                    shape=(self.matryoshka_dim, self.matryoshka_dim),
                    rank=adapter_rank,
                    specialist_name="color_router",
                ),
            ),
            DomainSpecialist(
                "logic",
                ["math_logic", "logic"],
                weight=0.5,
                adapter=SelfUpdatingAdapter(
                    shape=(self.matryoshka_dim, self.matryoshka_dim),
                    rank=adapter_rank,
                    specialist_name="logic_router",
                ),
            ),
            DomainSpecialist(
                "number_theory",
                ["math_number_theory", "math_arithmetic"],
                weight=0.6,
                adapter=SelfUpdatingAdapter(
                    shape=(self.matryoshka_dim, self.matryoshka_dim),
                    rank=adapter_rank,
                    specialist_name="number_router",
                ),
            ),
        ]
        # Allow CPU fallback for adapters (sovereign but no GPU gradients here)
        for spec in self.specialists:
            if spec.adapter is not None and hasattr(spec.adapter, "config"):
                spec.adapter.config.require_gpu = False
        # Scratchpad for temporary bindings (cleared per route call)
        self.scratchpad = RouterScratchpad()

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

    def _make_signature_vector(self, signature: Dict[str, int]) -> List[float]:
        """Build normalized signature vector (Matryoshka dim)."""
        flat = [
            float(signature.get("grid_rows", 0)),
            float(signature.get("grid_cols", 0)),
            float(signature.get("num_colors", 0)),
            float(signature.get("filled_cells", 0)),
        ]
        norm = l2_norm(flat) or 1.0
        flat = [v / norm for v in flat]
        return pad_or_truncate(flat, self.matryoshka_dim, 0.0)

    def _adapter_score(self, specialist: DomainSpecialist) -> float:
        """Cheap adapter-based score using diagonal projection and ternary quantization."""
        adapter = getattr(specialist, "adapter", None)
        if adapter is None or not hasattr(adapter, "weights"):
            return 0.0
        try:
            delta = adapter.weights.get_delta()
            vec = getattr(self, "_signature_vector", None)
            if vec is None or delta is None:
                return 0.0
            lim = min(len(vec), delta.shape[0])
            score = 0.0
            for i in range(lim):
                v = vec[i]
                q = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
                score += float(delta[i][i]) * q
            return score * 0.001
        except Exception:
            return 0.0

    def stash_gradient(self) -> None:
        """
        Store a lightweight, sovereign gradient map in scratchpad for adapters.

        Uses ternary-quantized signature to build diagonal gradients; stored as
        vector-dot style payload (list-of-lists) per specialist.
        """
        if not hasattr(self, "_signature_hint"):
            return
        sig_vec = self._make_signature_vector(self._signature_hint)
        grad_scale = 0.001
        for specialist in self.specialists:
            adapter = getattr(specialist, "adapter", None)
            if adapter is None or not hasattr(adapter, "weights"):
                continue
            lim = min(len(sig_vec), adapter.weights.shape[0])
            diag = [
                [0.0 for _ in range(adapter.weights.shape[1])]
                for _ in range(adapter.weights.shape[0])
            ]
            for i in range(lim):
                # ternary quantize signature entry
                v = sig_vec[i]
                q = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
                diag[i][i] = q * grad_scale
            # Optional VectorDotMap-style compression placeholder
            compressed = self._vectordotmap_compress(diag)
            self.scratchpad.set(f"grad_{specialist.name}", compressed)

    def apply_stashed_gradients(
        self,
        lr: float = 0.001,
        success_count: int = 0,
        shadow_copy=None,
        epochs: int = 1,
    ) -> None:
        """
        Decode and apply stashed gradients to specialist adapters using pure
        Python sparse loops over A/B (LoRA-style). Optionally logs into a
        shadow_copy replay buffer for consolidation.
        """
        if success_count <= 0:
            # No successes to justify adaptation; clear any stale gradients
            for specialist in self.specialists:
                self.scratchpad.set(f"grad_{specialist.name}", None)
            return
        applied_total = 0
        applied_by_specialist: Dict[str, int] = {}
        success_by_specialist: Dict[str, int] = {}
        replay_buffer = None
        if shadow_copy is not None:
            replay_buffer = getattr(shadow_copy, "replay_buffer", None)
            if replay_buffer is None and hasattr(shadow_copy, "__dict__"):
                shadow_copy.replay_buffer = []
                replay_buffer = shadow_copy.replay_buffer
        epochs_use = max(1, int(epochs))
        for epoch in range(epochs_use):
            for specialist in self.specialists:
                adapter = getattr(specialist, "adapter", None)
                if adapter is None or not hasattr(adapter, "weights"):
                    continue
                payload = self.scratchpad.get(f"grad_{specialist.name}")
                if not payload or payload.get("type") != "vdm_diag":
                    continue
                coeffs = payload.get("coeffs", [])
                A = adapter.weights.A
                B = adapter.weights.B
                rank = adapter.weights.rank
                lr_use = lr if lr is not None else adapter.config.learning_rate
                for idx, val in coeffs:
                    i = int(idx)
                    g = float(val)
                    # grad_A[i, k] = g * B[k, i]
                    for k in range(rank):
                        A[i, k] -= lr_use * (g * B[k, i])
                    # grad_B[k, i] = A[k, i] * g
                    for k in range(rank):
                        B[k, i] -= lr_use * (A[k, i] * g)
                    applied_total += 1
                    applied_by_specialist[specialist.name] = applied_by_specialist.get(specialist.name, 0) + 1
                if applied_by_specialist.get(specialist.name):
                    success_by_specialist[specialist.name] = success_by_specialist.get(specialist.name, 0) + 1
                # Log to shadow copy replay buffer if available
                if replay_buffer is not None:
                    replay_buffer.append(
                        {
                            "specialist": specialist.name,
                            "coeffs": coeffs,
                            "lr": lr_use,
                            "signature": getattr(self, "_signature_hint", {}),
                            "epoch": epoch,
                        }
                    )
        # Clear gradient after all epochs
        for specialist in self.specialists:
            self.scratchpad.set(f"grad_{specialist.name}", None)
        if applied_total:
            summary = ", ".join(f"{k}:{v}" for k, v in applied_by_specialist.items())
            succ_summary = ", ".join(f"{k}:{v}" for k, v in success_by_specialist.items())
            print(
                f"[Router] Applied {applied_total} adapter gradient entries "
                f"(successes={success_count}, epochs={epochs_use}) [{summary}] "
                f"specialist_successes=[{succ_summary}]"
            )
    def _vectordotmap_compress(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Compress a sparse diagonal matrix into a tiny dict suitable for
        VectorDotMap-style storage. This is a placeholder and keeps data
        sovereign and minimal.
        """
        coeffs: List[Tuple[int, float]] = []
        for i, row in enumerate(matrix):
            val = row[i]
            if val != 0.0:
                coeffs.append((i, val))
        return {"type": "vdm_diag", "size": len(matrix), "coeffs": coeffs}

    class _AttrDict(dict):
        """Dict with attribute access for compatibility with legacy tests."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    def route(self, grid: Sequence[Sequence[int]], top_k: int = 27, use_semantics: bool = True) -> List[Dict[str, Any]]:  # SOVEREIGN: Tesla 3-6-9 (increased from 3)
        # Reset scratchpad for this route call
        self.scratchpad.clear()
        drawing_program = self.grid_to_drawing_rpn(grid)
        signature = self.task_signature(grid)
        # Save signature for downstream scoring
        self._signature_hint = signature
        self._signature_vector = self._make_signature_vector(signature)
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
        # Stash gradients for specialists (ternary diag) and apply when successes exist
        self.stash_gradient()
        self.apply_stashed_gradients(success_count=len(high_quality), shadow_copy=shadow_copy, epochs=5)
        replay_stats = {"replayed": 0, "epochs": 0}
        # Apply replay buffer gradients (shadow copy consolidation)
        try:
            replay_stats = self.train_from_replay(shadow_copy, epochs=5)
        except Exception:
            replay_stats = {"replayed": 0, "epochs": 0}

        # Simple adapter feedback: reward recent successes on diagonal
        return {
            "processed": len(recent),
            "high_quality": len(high_quality),
            "pattern_types": len(pattern_counts),
            "replayed": replay_stats.get("replayed", 0),
            "replay_epochs": replay_stats.get("epochs", 0),
        }

    def train_from_replay(self, shadow_copy, lr: float = 0.001, epochs: int = 1) -> Dict[str, int]:
        """
        Apply gradients stored in shadow_copy.replay_buffer to adapters.

        Replay entries are tiny VDM-style payloads; clears buffer after use.
        """
        buffer = getattr(shadow_copy, "replay_buffer", None)
        if not buffer:
            return {"replayed": 0}

        applied = 0
        epochs_use = max(1, int(epochs))
        for entry in list(buffer):
            spec_name = entry.get("specialist")
            coeffs = entry.get("coeffs", [])
            lr_use = entry.get("lr", lr)
            specialist = next((s for s in self.specialists if s.name == spec_name), None)
            if specialist is None:
                continue
            adapter = getattr(specialist, "adapter", None)
            if adapter is None or not hasattr(adapter, "weights"):
                continue
            A = adapter.weights.A
            B = adapter.weights.B
            rank = adapter.weights.rank
            for epoch in range(epochs_use):
                for idx, val in coeffs:
                    i = int(idx)
                    g = float(val)
                    for k in range(rank):
                        A[i, k] -= lr_use * (g * B[k, i])
                    for k in range(rank):
                        B[k, i] -= lr_use * (A[k, i] * g)
                    applied += 1
        # Clear buffer
        shadow_copy.replay_buffer = []
        if applied:
            print(f"[Router] Replayed {applied} adapter gradient entries from shadow_copy (epochs={epochs_use})")
        return {"replayed": applied, "epochs": epochs_use}

    def _rank_rules(self, top_k: int) -> List[GrammarRule]:
        """
        Rank rules heuristically (no embedding needed for now).

        SOVEREIGN: Simple domain-based filtering + learned pattern preferences.
        TODO: Use RPN-based semantic similarity when embeddings are needed.
        """
        all_rules = list(self.grammar.list_rules())

        def rule_priority(rule: GrammarRule) -> float:
            priority = 0.0
            # Learned pattern preferences
            if hasattr(self, "_pattern_prefs") and self._pattern_prefs:
                rid = rule.rule_id.lower()
                for pattern, count in self._pattern_prefs.items():
                    if pattern in rid:
                        priority += count
            # Domain specialists
            for specialist in self.specialists:
                priority += specialist.score(rule, getattr(self, "_signature_hint", {}))
                # Adapter-based scoring (cheap diag projection)
                if getattr(specialist, "adapter", None) is not None:
                    priority += self._adapter_score(specialist)
            # Domain hint: drawing/spatial first
            if getattr(rule, "domain", "") in {"drawing", "spatial"}:
                priority += 0.5
            return priority

        ordered = sorted(all_rules, key=rule_priority, reverse=True)
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
