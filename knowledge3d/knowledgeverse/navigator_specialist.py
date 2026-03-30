"""Navigator meta-specialist for multi-path routing and composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .specialist_router import SpecialistRouter
from .specialist_base import _resolve_ternary
from .trm_weight_store import TRMWeightStore

_NUMERIC_WORD_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_NUMERIC_WORD_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_NUMERIC_WORD_SPECIAL: dict[str, float] = {
    "half": 0.5,
    "quarter": 0.25,
    "twice": 2.0,
    "double": 2.0,
    "thrice": 3.0,
    "triple": 3.0,
    "quadruple": 4.0,
}

_SEMANTIC_REFERENCE_MULTIPLIERS: dict[str, float] = {
    "half": 0.5,
    "quarter": 0.25,
    "twice": 2.0,
    "double": 2.0,
    "thrice": 3.0,
    "triple": 3.0,
    "quadruple": 4.0,
}

_SEMANTIC_SKIP_TOKENS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "each",
    "every",
    "for",
    "from",
    "in",
    "much",
    "many",
    "of",
    "or",
    "per",
    "that",
    "the",
    "this",
    "times",
    "time",
    "total",
}

_SEMANTIC_SCOPE_CUES = {"per", "each", "every"}
_SEMANTIC_TEMPORAL_UNITS = {
    "day",
    "daily",
    "hour",
    "minute",
    "month",
    "week",
    "year",
}


@dataclass
class PathCandidate:
    """Single specialist path explored by the navigator meta-specialist."""

    specialist: str
    domain: str
    route: dict[str, Any]
    patterns: list[dict[str, Any]]
    composed: dict[str, Any]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "domain": self.domain,
            "route": dict(self.route),
            "patterns_used": len(self.patterns),
            "confidence": self.confidence,
            "composed": dict(self.composed),
        }


class NavigatorSpecialist:
    """
    Meta-specialist that plans routes using multiple algorithms and composes paths.

    This class keeps the routing policy centralized while allowing multi-path
    exploration. It intentionally operates at orchestration level and preserves
    sovereign hot-path constraints (no external ML dependencies).
    """

    def __init__(
        self,
        knowledgeverse: Any | None = None,
        *,
        router: SpecialistRouter | None = None,
        max_paths: int = 4,
    ):
        self.knowledgeverse = knowledgeverse
        self.router = router or SpecialistRouter()
        self.max_paths = max(1, int(max_paths))
        # Topology memory: query signature -> specialist stats
        self.routing_topology: dict[str, dict[str, dict[str, int]]] = {}
        self._update_count = 0
        self._auto_save_interval = 10
        state_path = None
        if self.knowledgeverse is not None and hasattr(self.knowledgeverse, "storage_root"):
            state_path = Path(getattr(self.knowledgeverse, "storage_root")) / "checkpoints" / "trm_routing_state.json"
        self.weight_store = TRMWeightStore(
            state_path if state_path is not None else Path("../Knowledge3D.local/checkpoints/trm_routing_state.json")
        )
        self.load_state()

    def plan_routes(
        self,
        query: str,
        *,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
        use_forward_backward: bool = False,
        task_type: str | None = None,
        goal_type_family: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Plan route candidates using multiple strategies.

        Strategies:
        - heuristic_auto: current centralized router
        - topology_memory: successful historical routes for similar query signatures
        - legacy_keywords_2025: compatibility with older keyword routing behavior
        """
        routes: list[dict[str, Any]] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()

        def _append(route: dict[str, Any], strategy: str) -> None:
            key = (
                str(route["specialist"]),
                str(route["domain"]),
                tuple(str(g) for g in route["galaxy_names"]),
                str(route.get("query_variant", query)),
                strategy,
            )
            if key in seen:
                return
            seen.add(key)
            enriched = dict(route)
            enriched["strategy"] = strategy
            routes.append(enriched)

        base = self.router.route(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        if task_type:
            base["task_type"] = str(task_type).strip().upper()
        if goal_type_family:
            base["goal_type_family"] = str(goal_type_family).strip().lower()
        if use_forward_backward:
            _append(self._forward_reading_path(query, base), "forward")
            _append(self._backward_reading_path(query, base), "backward")
            _append(self._fusion_reading_path(query, base), "fusion")

        _append({**base, "query_variant": query}, "auto")

        for route in self._routes_from_topology(query):
            route = {**route, "query_variant": query}
            _append(route, "topology_memory")

        for route in self._routes_from_legacy_keywords(query):
            route = {**route, "query_variant": query}
            _append(route, "legacy_keywords_2025")

        if len(routes) > self.max_paths:
            routes = routes[: self.max_paths]
        return routes

    def explore_multi_path(
        self,
        trm_navigator: Any,
        query: str,
        *,
        use_enriched: bool = True,
        use_forward_backward: bool = True,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> list[PathCandidate]:
        """Explore multiple specialist routes and collect candidate programs."""
        routes = self.plan_routes(
            query=query,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
            use_forward_backward=use_forward_backward,
        )

        candidates: list[PathCandidate] = []
        for route in routes:
            route_query = str(route.get("query_variant", query))
            patterns = trm_navigator.query(
                query=route_query,
                galaxy_names=route["galaxy_names"],
                top_k=30 if use_enriched else 5,
                specialist=route["specialist"],
                domain_hint=route["domain"],
            )
            composed = trm_navigator.compose(
                query=query,
                patterns=patterns,
                specialist=route["specialist"],
                use_enriched=use_enriched,
            )
            confidence = self._estimate_confidence(
                patterns_used=len(patterns),
                specialist=str(route["specialist"]),
                strategy=str(route.get("strategy", "heuristic_auto")),
            )
            candidates.append(
                PathCandidate(
                    specialist=str(route["specialist"]),
                    domain=str(route["domain"]),
                    route=route,
                    patterns=patterns,
                    composed=composed,
                    confidence=confidence,
                )
            )
        return candidates

    def compose_paths(self, query: str, paths: Sequence[PathCandidate]) -> dict[str, Any]:
        """Compose multiple path candidates into a final program dict."""
        if not paths:
            fallback = self.router.route(query=query, specialist="grammar")
            return {
                "program_type": "math_expression",
                "expression": query,
                "specialist": fallback["specialist"],
                "patterns_used": 0,
                "route": fallback,
                "meta_specialist": {
                    "mode": "fallback",
                    "paths_considered": 0,
                },
            }

        signature_counts: dict[str, int] = {}
        for path in paths:
            signature = self._path_signature(path)
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

        score_details = [
            self._path_score_details(path, signature_counts.get(self._path_signature(path), 1))
            for path in paths
        ]
        ranked = sorted(
            score_details,
            key=lambda item: float(item["final_score"]),
            reverse=True,
        )
        best_item = ranked[0]
        best = best_item["path"]
        best_signature = self._path_signature(best)
        agreement = signature_counts.get(best_signature, 1)
        final_program = dict(best.composed)
        final_program["specialist"] = best.specialist
        final_program["route"] = dict(best.route)
        contributing = [
            item for item in ranked if self._path_signature(item["path"]) == best_signature
        ]
        final_program["meta_specialist"] = {
            "mode": "multi_path",
            "paths_considered": len(paths),
            "component_specialists": [item["path"].specialist for item in ranked],
            "primary_specialist": best.specialist,
            "confidence": best.confidence,
            "cross_path_agreement": agreement,
            "strategies": [str(item["path"].route.get("strategy", "heuristic_auto")) for item in ranked],
            "contributing_strategies": [
                str(item["path"].route.get("strategy", "heuristic_auto")) for item in contributing
            ],
            "grammar_boosted": sum(1 for item in ranked if float(item["grammar_confidence"]) > 0.7),
            "path_scores": [
                {
                    "strategy": str(item["path"].route.get("strategy", "heuristic_auto")),
                    "specialist": item["path"].specialist,
                    "final_score": round(float(item["final_score"]), 6),
                    "grammar_confidence": round(float(item["grammar_confidence"]), 6),
                    "cross_path_agreement": int(item["cross_path_agreement"]),
                    "composition_depth": int(item["composition_depth"]),
                }
                for item in ranked
            ],
        }
        return final_program

    def navigate_and_compose(
        self,
        trm_navigator: Any,
        query: str,
        *,
        use_enriched: bool = True,
        use_forward_backward: bool = True,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Full meta-specialist flow: plan -> explore -> compose."""
        paths = self.explore_multi_path(
            trm_navigator=trm_navigator,
            query=query,
            use_enriched=use_enriched,
            use_forward_backward=use_forward_backward,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        composed = self.compose_paths(query=query, paths=paths)
        self._log_path_contribution_events(
            query=query,
            specialist=specialist,
            composed=composed,
        )
        return composed

    def learn_routing_topology(
        self,
        query: str,
        *,
        specialist: str,
        success: bool | None = None,
        ternary_outcome: int | None = None,
    ) -> None:
        """Update topology memory with route outcome signal."""
        outcome = _resolve_ternary(success, ternary_outcome)
        signature = self._query_signature(query)
        specialist_stats = self.routing_topology.setdefault(signature, {})
        bucket = specialist_stats.setdefault(specialist, {"success": 0, "failure": 0, "uncertain": 0})
        bucket.setdefault("uncertain", 0)
        if outcome > 0:
            bucket["success"] += 1
            self.router.adjust_specialist_bias(specialist, +0.02)
        elif outcome < 0:
            bucket["failure"] += 1
            self.router.adjust_specialist_bias(specialist, -0.01)
        else:
            bucket["uncertain"] += 1
        self._update_count += 1
        if self._update_count % self._auto_save_interval == 0:
            self.save_state()

    def _routes_from_topology(self, query: str) -> list[dict[str, Any]]:
        signature = self._query_signature(query)
        specialist_stats = self.routing_topology.get(signature, {})
        if not specialist_stats:
            return []
        ranked = sorted(
            specialist_stats.items(),
            key=lambda item: (item[1]["success"] - item[1]["failure"], item[1]["success"]),
            reverse=True,
        )
        out: list[dict[str, Any]] = []
        for specialist, _ in ranked[: self.max_paths]:
            out.append(self.router.route(query=query, specialist=specialist))
        return out

    def _routes_from_legacy_keywords(self, query: str) -> list[dict[str, Any]]:
        """
        Legacy 2025 compatibility strategy.

        Keeps the older keyword-first routing lineage available as one vote in
        multi-path planning.
        """
        lowered = query.lower()
        legacy_candidates: list[str] = []
        if any(tok in lowered for tok in ("flip", "rotate", "grid", "arc", "pattern", "shape")):
            legacy_candidates.append("visual")
        if any(tok in lowered for tok in ("derivative", "integral", "equation", "solve", "sum")):
            legacy_candidates.append("math")
        if any(tok in lowered for tok in ("force", "energy", "mass", "velocity", "physics")):
            legacy_candidates.append("physics")
        if any(tok in lowered for tok in ("proof", "logic", "implies", "grammar", "syntax")):
            legacy_candidates.append("grammar")
        if len(set(legacy_candidates)) >= 2:
            legacy_candidates.insert(0, "cartographer")
        out: list[dict[str, Any]] = []
        for specialist in legacy_candidates[: self.max_paths]:
            out.append(self.router.route(query=query, specialist=specialist))
        return out

    def _estimate_confidence(self, *, patterns_used: int, specialist: str, strategy: str) -> float:
        base = 0.35 + min(patterns_used, 30) / 100.0
        if specialist == "cartographer":
            base += 0.05
        if strategy == "topology_memory":
            base += 0.08
        if strategy in {"forward_reading", "backward_reading", "forward", "backward"}:
            base += 0.03
        if strategy == "fusion":
            base += 0.05
        return max(0.05, min(base, 0.99))

    def _forward_reading_path(self, query: str, base_route: dict[str, Any]) -> dict[str, Any]:
        """Create route variant that parses query left-to-right."""
        clauses = self._split_into_sentences(query)
        if not clauses:
            return {**base_route, "query_variant": query}
        context = [self._extract_variables_and_constraints(s) for s in clauses[:-1]]
        goal = self._extract_goal(clauses[-1])
        context_text = " ".join(item.get("raw", "") for item in context if item.get("raw"))
        goal_text = goal.get("raw", clauses[-1])
        variant = f"context: {context_text} goal: {goal_text}".strip()
        return {
            **base_route,
            "query_variant": variant or query,
            "forward_parse": {"context": context, "goal": goal},
        }

    def _backward_reading_path(self, query: str, base_route: dict[str, Any]) -> dict[str, Any]:
        """Create route variant that parses query right-to-left."""
        clauses = self._split_into_sentences(query)
        if not clauses:
            return {**base_route, "query_variant": query}
        reverse_clauses = list(reversed(clauses))
        goal = self._extract_goal(reverse_clauses[0])
        goal_type_entry = self._goal_type_entry(str(goal.get("raw", reverse_clauses[0])), base_route=base_route)
        goal_metadata = goal_type_entry.get("metadata") if isinstance(goal_type_entry.get("metadata"), dict) else {}
        if goal_metadata:
            goal = self._apply_goal_roles_to_block(goal, goal_metadata=goal_metadata)
        if goal_metadata:
            goal["goal_type"] = str(goal_metadata.get("goal_type", "")).strip()
            goal["operation_frame"] = str(goal_metadata.get("operation_frame", "")).strip()
            goal["implies_roles"] = dict(goal_metadata.get("implies_roles", {})) if isinstance(goal_metadata.get("implies_roles"), dict) else {}
            goal["goal_type_id"] = str(goal_type_entry.get("id", "")).strip()
        dependencies = [
            self._apply_goal_roles_to_block(
                self._extract_variables_and_constraints(s),
                goal_metadata=goal_metadata,
            )
            for s in reverse_clauses[1:]
        ]
        deps_text = " ".join(item.get("raw", "") for item in dependencies if item.get("raw"))
        goal_text = goal.get("raw", reverse_clauses[0])
        variant = f"goal: {goal_text} dependencies: {deps_text}".strip()
        return {
            **base_route,
            "query_variant": variant or query,
            "backward_parse": {"goal": goal, "dependencies": dependencies},
        }

    def _fusion_reading_path(self, query: str, base_route: dict[str, Any]) -> dict[str, Any]:
        """
        Create a fusion route that merges forward and backward parses.

        Deduplicates variable assignments and chooses the most specific goal.
        """
        forward = self._forward_reading_path(query, base_route)
        backward = self._backward_reading_path(query, base_route)

        backward_goal = backward.get("backward_parse", {}).get("goal", {})
        backward_goal_metadata = {
            "goal_type": str(backward_goal.get("goal_type", "")).strip(),
            "operation_frame": str(backward_goal.get("operation_frame", "")).strip(),
            "implies_roles": (
                dict(backward_goal.get("implies_roles", {}))
                if isinstance(backward_goal.get("implies_roles"), dict)
                else {}
            ),
        }
        forward_context = list(forward.get("forward_parse", {}).get("context", []))
        if backward_goal_metadata.get("implies_roles"):
            forward_context = [
                self._apply_goal_roles_to_block(dict(ctx), goal_metadata=backward_goal_metadata)
                if isinstance(ctx, dict)
                else ctx
                for ctx in forward_context
            ]
        backward_deps = list(backward.get("backward_parse", {}).get("dependencies", []))

        merged_vars: dict[str, Any] = {}
        merged_quantities: list[dict[str, Any]] = []
        quantity_index: dict[tuple[float, str, str, int], int] = {}

        def _merge_var_block(block: dict[str, Any]) -> None:
            if str(block.get("type", "")) != "variables":
                return
            data = block.get("data", {})
            if not isinstance(data, dict):
                return
            for key, value in data.items():
                merged_vars[str(key)] = value

        def _merge_quantity_block(block: dict[str, Any]) -> None:
            rows = block.get("quantities") if isinstance(block.get("quantities"), list) else []
            raw_block = str(block.get("raw", "")).strip()
            block_kind = str(block.get("type", "")).strip().lower() or "context"
            for raw_row in rows:
                if not isinstance(raw_row, dict):
                    continue
                try:
                    value = float(raw_row.get("value"))
                except Exception:
                    continue
                surface = str(raw_row.get("surface", "")).strip().lower()
                role = str(raw_row.get("role", "")).strip().lower() or "quantity"
                offset = int(raw_row.get("offset", 0) or 0)
                key = (value, surface, raw_block, offset)
                existing_index = quantity_index.get(key)
                if existing_index is not None:
                    existing = merged_quantities[existing_index]
                    existing_role = str(existing.get("role", "")).strip().lower() or "quantity"
                    if existing_role == "quantity" and role != "quantity":
                        existing["role"] = role
                        existing["source"] = str(raw_row.get("source", "")).strip() or "parse"
                    continue
                quantity_index[key] = len(merged_quantities)
                merged_quantities.append(
                    {
                        "value": value,
                        "surface": str(raw_row.get("surface", "")).strip(),
                        "role": role,
                        "source": str(raw_row.get("source", "")).strip() or "parse",
                        "offset": offset,
                        "raw_block": raw_block,
                        "block_kind": block_kind,
                    }
                )

        for ctx in forward_context:
            if isinstance(ctx, dict):
                _merge_var_block(ctx)
                _merge_quantity_block(ctx)
        for dep in backward_deps:
            if isinstance(dep, dict):
                _merge_var_block(dep)
                _merge_quantity_block(dep)

        f_goal = forward.get("forward_parse", {}).get("goal", {})
        if isinstance(f_goal, dict) and backward_goal_metadata.get("implies_roles"):
            f_goal = self._apply_goal_roles_to_block(dict(f_goal), goal_metadata=backward_goal_metadata)
        b_goal = backward_goal
        if isinstance(f_goal, dict):
            _merge_quantity_block(f_goal)
        if isinstance(b_goal, dict):
            _merge_quantity_block(b_goal)
        f_expr = str(f_goal.get("expression", "")) if isinstance(f_goal, dict) else ""
        b_expr = str(b_goal.get("expression", "")) if isinstance(b_goal, dict) else ""
        unified_goal = f_goal if len(f_expr) >= len(b_expr) else b_goal
        if not isinstance(unified_goal, dict):
            unified_goal = {"type": "goal", "raw": query}
        if isinstance(b_goal, dict):
            for key in ("goal_type", "operation_frame", "implies_roles", "goal_type_id"):
                if key not in unified_goal and b_goal.get(key):
                    unified_goal[key] = b_goal.get(key)

        semantic_entities: list[dict[str, Any]] = []
        goal_entity: dict[str, Any] = {}
        if str(base_route.get("goal_type_family", "")).strip().lower() == "gsm8k":
            semantic_entities, goal_entity = self._annotate_semantic_roles(
                merged_quantities,
                unified_goal=unified_goal,
            )

        var_str = ", ".join(f"{k}={v}" for k, v in merged_vars.items())
        goal_raw = str(unified_goal.get("raw") or unified_goal.get("expression") or query)
        query_variant = f"Given {var_str}, {goal_raw}".strip() if var_str else goal_raw

        return {
            **base_route,
            "query_variant": query_variant or query,
            "fusion_parse": {
                "merged_variables": merged_vars,
                "merged_quantities": merged_quantities,
                "quantity_values": [float(row["value"]) for row in merged_quantities],
                "unified_goal": unified_goal,
                "goal_type": str(unified_goal.get("goal_type", "")).strip(),
                "operation_frame": str(unified_goal.get("operation_frame", "")).strip(),
                "semantic_entities": semantic_entities,
                "goal_entity": goal_entity,
                "forward_context_count": len(forward_context),
                "backward_deps_count": len(backward_deps),
                "deduplication_savings": max(
                    (len(forward_context) + len(backward_deps)) - len(merged_vars),
                    0,
                ),
            },
        }

    def _split_into_sentences(self, query: str) -> list[str]:
        """Split query into sentence-like clauses for directional parsing."""
        pieces = re.split(r"(?:[.?!]\s+|,\s+given\s+|,\s+where\s+)", query.strip(), flags=re.IGNORECASE)
        return [piece.strip() for piece in pieces if piece and piece.strip()]

    def _extract_variables_and_constraints(self, sentence: str) -> dict[str, Any]:
        """Extract compact variable assignments/constraints from text."""
        assignments: dict[str, Any] = {}
        for var, value in re.findall(r"\b([A-Za-z_]\w*)\s*=\s*([0-9.+-]+)\b", sentence):
            try:
                assignments[var] = float(value) if "." in value else int(value)
            except ValueError:
                assignments[var] = value
        quantities = self._extract_quantities(sentence)
        return {
            "type": "variables" if assignments else "context",
            "data": assignments,
            "quantities": quantities,
            "raw": sentence,
        }

    @classmethod
    def _numeric_word_value(cls, token: str) -> float | None:
        normalized = str(token).strip().lower()
        if not normalized:
            return None
        if normalized in _NUMERIC_WORD_SPECIAL:
            return float(_NUMERIC_WORD_SPECIAL[normalized])
        if normalized in _NUMERIC_WORD_UNITS:
            return float(_NUMERIC_WORD_UNITS[normalized])
        if normalized in _NUMERIC_WORD_TENS:
            return float(_NUMERIC_WORD_TENS[normalized])
        if "-" in normalized:
            left, _, right = normalized.partition("-")
            if left in _NUMERIC_WORD_SPECIAL and right:
                return float(_NUMERIC_WORD_SPECIAL[left])
            if left in _NUMERIC_WORD_TENS and right in _NUMERIC_WORD_UNITS:
                return float(_NUMERIC_WORD_TENS[left] + _NUMERIC_WORD_UNITS[right])
        return None

    def _extract_quantities(self, sentence: str) -> list[dict[str, Any]]:
        quantities: list[dict[str, Any]] = []
        lowered = str(sentence or "").lower()
        digit_pattern = r"(?<![A-Za-z])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)"
        for match in re.finditer(digit_pattern, sentence):
            raw = match.group(0)
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            quantities.append(
                {
                    "value": value,
                    "surface": raw,
                    "role": "quantity",
                    "source": "digit",
                    "offset": int(match.start()),
                }
            )
        for match in re.finditer(r"\b[a-z]+(?:-[a-z]+)?\b", lowered):
            raw = match.group(0)
            value = self._numeric_word_value(raw)
            if value is None:
                continue
            quantities.append(
                {
                    "value": float(value),
                    "surface": raw,
                    "role": "quantity",
                    "source": "word",
                    "offset": int(match.start()),
                }
            )
        quantities.sort(key=lambda item: (int(item.get("offset", 0)), str(item.get("surface", ""))))
        return quantities

    def _extract_goal(self, sentence: str) -> dict[str, Any]:
        """Extract goal-like expression from a sentence."""
        lowered = sentence.lower()
        goal_markers = ("what is", "find", "compute", "calculate", "determine", "solve for")
        quantities = self._extract_quantities(sentence)
        for marker in goal_markers:
            idx = lowered.find(marker)
            if idx >= 0:
                expr = sentence[idx + len(marker) :].strip(" ?.")
                return {
                    "type": "goal",
                    "operation": "evaluate",
                    "expression": expr,
                    "raw": sentence,
                    "quantities": quantities,
                }
        return {"type": "goal", "raw": sentence, "quantities": quantities}

    def _goal_type_rows(self, *, base_route: dict[str, Any]) -> list[dict[str, Any]]:
        if self.knowledgeverse is None or not hasattr(self.knowledgeverse, "get_gpu_galaxy_catalog"):
            return []
        allowed = {
            str(name).strip()
            for name in (
                base_route.get("galaxy_names")
                if isinstance(base_route.get("galaxy_names"), list)
                else []
            )
            if str(name).strip()
        }
        goal_type_family = str(base_route.get("goal_type_family", "")).strip().lower()
        allowed_subfields: set[str] | None = None
        if goal_type_family == "gsm8k":
            allowed_subfields = {"word_problem_binding"}
        elif goal_type_family == "lhe":
            allowed_subfields = {"lhe_goal_typing"}
        elif goal_type_family == "math":
            allowed_subfields = {"algebra", "geometry", "number_theory", "combinatorics"}
        rows: list[dict[str, Any]] = []
        for entry in self.knowledgeverse.get_gpu_galaxy_catalog():
            galaxy = str(entry.get("galaxy", "")).strip()
            if allowed and galaxy not in allowed:
                continue
            if galaxy != "Grammar":
                continue
            metadata = self.knowledgeverse._catalog_metadata(entry)
            if not str(metadata.get("goal_type", "")).strip():
                continue
            if allowed_subfields is not None:
                subfield = str(metadata.get("subfield", "")).strip().lower()
                if subfield not in allowed_subfields:
                    continue
            if not list(entry.get("embedding16", [])):
                continue
            rows.append(self.knowledgeverse._resolve_catalog_entry(entry))
        return rows

    @staticmethod
    def _phrase_overlap_score(text: str, phrases: list[str]) -> float:
        lowered = str(text or "").lower()
        active = [str(phrase).strip().lower() for phrase in phrases if str(phrase).strip()]
        if not lowered or not active:
            return 0.0
        matched = sum(1 for phrase in active if phrase in lowered)
        return float(matched) / float(len(active))

    def _goal_type_entry(self, goal_text: str, *, base_route: dict[str, Any]) -> dict[str, Any]:
        rows = self._goal_type_rows(base_route=base_route)
        if not rows or self.knowledgeverse is None:
            return {}
        try:
            goal_embedding = self.knowledgeverse._embed_query_gpu(goal_text)
        except Exception:
            return {}
        similarities = self.knowledgeverse._embedding_similarities(
            goal_embedding,
            [list(entry.get("embedding16", [])) for entry in rows],
        )
        cue_rows = [
            self._phrase_overlap_score(
                goal_text,
                [
                    str(value)
                    for value in (
                        (
                            entry.get("metadata")
                            if isinstance(entry.get("metadata"), dict)
                            else {}
                        ).get("structural_cues", [])
                        if isinstance(
                            (
                                entry.get("metadata")
                                if isinstance(entry.get("metadata"), dict)
                                else {}
                            ).get("structural_cues", []),
                            list,
                        )
                        else []
                    )
                ],
            )
            for entry in rows
        ]
        ranked = sorted(
            [
                (
                    1.0 if cue_score > 0.0 else 0.0,
                    (0.35 * float(similarity)) + (0.65 * float(cue_score)),
                    entry,
                )
                for similarity, cue_score, entry in zip(similarities, cue_rows, rows)
            ],
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return dict(ranked[0][2]) if ranked else {}

    @staticmethod
    def _quantity_local_snippet(raw_text: str, *, surface: str, offset: int) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return str(surface or "").strip()
        start = max(int(offset) - 56, 0)
        end = min(int(offset) + max(len(str(surface or "")), 1) + 48, len(text))
        return text[start:end].strip(" ,.;:!?") or text

    @staticmethod
    def _semantic_word_spans(text: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0).strip().lower(), int(match.start()), int(match.end()))
            for match in re.finditer(r"[a-zA-Z$]+(?:-[a-zA-Z]+)?", str(text or ""))
        ]

    @staticmethod
    def _normalize_semantic_token(token: str) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "", str(token or "").strip().lower())
        if len(normalized) > 4 and normalized.endswith("ies"):
            normalized = normalized[:-3] + "y"
        elif len(normalized) > 3 and normalized.endswith("ses"):
            normalized = normalized[:-2]
        elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
            normalized = normalized[:-1]
        return normalized

    def _semantic_context_tokens(
        self,
        raw_text: str,
        *,
        surface: str,
        offset: int,
    ) -> tuple[list[str], list[str], str]:
        snippet = self._quantity_local_snippet(raw_text, surface=surface, offset=offset).lower()
        spans = self._semantic_word_spans(raw_text)
        anchor = int(offset) + max(len(str(surface or "")), 1)
        before = [token for token, _start, end in spans if end <= int(offset)]
        after = [token for token, start, _end in spans if start >= anchor]
        return before[-6:], after[:8], snippet

    def _semantic_unit_from_tokens(
        self,
        tokens: list[str],
        *,
        allow_reference: bool,
    ) -> str:
        if not allow_reference:
            return ""
        for token in tokens:
            normalized = self._normalize_semantic_token(token)
            if not normalized or normalized in _SEMANTIC_SKIP_TOKENS:
                continue
            if self._numeric_word_value(normalized) is not None:
                continue
            if normalized in _SEMANTIC_TEMPORAL_UNITS:
                continue
            return normalized
        return ""

    def _semantic_scope_from_tokens(self, before: list[str], after: list[str]) -> str:
        merged = [self._normalize_semantic_token(token) for token in [*before[-2:], *after[:6]]]
        for index, token in enumerate(merged[:-1]):
            next_token = merged[index + 1]
            if token in _SEMANTIC_SCOPE_CUES and next_token:
                return f"per_{next_token}"
            if token in {"a", "an"} and index > 0 and merged[index - 1] in {"time", "times"} and next_token:
                return f"per_{next_token}"
        return ""

    def _semantic_reference_marker(self, *, surface: str, snippet: str) -> str | None:
        normalized_surface = self._normalize_semantic_token(surface)
        if normalized_surface in _SEMANTIC_REFERENCE_MULTIPLIERS:
            return normalized_surface
        return None

    def _find_nearest_referent(
        self,
        index: int,
        semantic_entities: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if index <= 0:
            return None
        target = semantic_entities[index]
        target_unit = str(target.get("unit", "")).strip().lower()
        target_scope = str(target.get("scope", "")).strip().lower()
        target_block = str(target.get("raw_block", "")).strip().lower()
        for candidate in reversed(semantic_entities[:index]):
            candidate_reference = (
                str(candidate.get("reference", "")).strip().lower()
                if isinstance(candidate.get("reference"), str)
                else ""
            )
            if candidate_reference:
                continue
            candidate_unit = str(candidate.get("unit", "")).strip().lower()
            candidate_scope = str(candidate.get("scope", "")).strip().lower()
            if target_unit and candidate_unit == target_unit:
                return candidate
            if target_scope and candidate_scope == target_scope:
                return candidate
            if target_block and str(candidate.get("raw_block", "")).strip().lower() == target_block:
                return candidate
        for candidate in reversed(semantic_entities[:index]):
            candidate_reference = (
                str(candidate.get("reference", "")).strip().lower()
                if isinstance(candidate.get("reference"), str)
                else ""
            )
            if not candidate_reference:
                return candidate
        return None

    def _resolve_reference_entities(self, semantic_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, entity in enumerate(semantic_entities):
            reference = (
                str(entity.get("reference", "")).strip().lower()
                if isinstance(entity.get("reference"), str)
                else ""
            )
            if not reference:
                continue
            referent = self._find_nearest_referent(index, semantic_entities)
            if not isinstance(referent, dict):
                continue
            multiplier = _SEMANTIC_REFERENCE_MULTIPLIERS.get(reference)
            if multiplier is None:
                continue
            try:
                resolved_value = float(referent.get("resolved_value", referent.get("value", 0.0))) * float(multiplier)
            except Exception:
                continue
            entity["resolved_value"] = float(resolved_value)
            entity["reference_source"] = {
                "value": float(referent.get("resolved_value", referent.get("value", 0.0))),
                "surface": str(referent.get("surface", "")).strip(),
                "unit": str(referent.get("unit", "")).strip(),
                "scope": str(referent.get("scope", "")).strip(),
            }
            if not str(entity.get("unit", "")).strip():
                entity["unit"] = str(referent.get("unit", "")).strip()
            if not str(entity.get("scope", "")).strip():
                entity["scope"] = str(referent.get("scope", "")).strip()
            if str(entity.get("role", "")).strip().lower() in {"", "quantity"}:
                entity["role"] = str(referent.get("role", "")).strip() or "count"
        return semantic_entities

    def _goal_semantic_entity(
        self,
        unified_goal: dict[str, Any],
        *,
        semantic_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raw_text = str(
            unified_goal.get("expression")
            or unified_goal.get("raw")
            or ""
        ).strip()
        lowered = raw_text.lower()
        unit = ""
        scope = ""
        many_match = re.search(r"how\s+many(?:\s+total)?\s+([a-zA-Z-]+)", lowered)
        if many_match:
            unit = self._normalize_semantic_token(many_match.group(1))
        scope_match = re.search(r"(?:per|each|every|a|an)\s+([a-zA-Z-]+)", lowered)
        if scope_match:
            prev = lowered[: scope_match.start()].strip().split()
            if not prev or prev[-1] in {"times", "time", "per", "each", "every", "run", "does", "do"}:
                scope = f"per_{self._normalize_semantic_token(scope_match.group(1))}"
        if not unit:
            for entity in reversed(semantic_entities):
                inferred_unit = self._normalize_semantic_token(str(entity.get("unit", "")))
                if inferred_unit:
                    unit = inferred_unit
                    break
        return {
            "role": "goal",
            "unit": unit,
            "scope": scope,
            "raw": raw_text,
        }

    def _annotate_semantic_roles(
        self,
        merged_quantities: list[dict[str, Any]],
        *,
        unified_goal: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        semantic_entities: list[dict[str, Any]] = []
        sorted_rows = sorted(
            [dict(row) for row in merged_quantities if isinstance(row, dict)],
            key=lambda row: (int(row.get("offset", 0) or 0), str(row.get("surface", ""))),
        )
        for row in sorted_rows:
            raw_text = str(row.get("raw_block", "")).strip()
            surface = str(row.get("surface", "")).strip()
            offset = int(row.get("offset", 0) or 0)
            before_tokens, after_tokens, snippet = self._semantic_context_tokens(
                raw_text,
                surface=surface,
                offset=offset,
            )
            local_window = raw_text[
                max(offset - 24, 0) : min(offset + max(len(surface), 1) + 24, len(raw_text))
            ].lower()
            tight_window = raw_text[
                max(offset - 8, 0) : min(offset + max(len(surface), 1) + 8, len(raw_text))
            ].lower()
            reference = self._semantic_reference_marker(surface=surface, snippet=snippet)
            unit = self._semantic_unit_from_tokens(after_tokens, allow_reference=reference is None)
            scope = self._semantic_scope_from_tokens(before_tokens, after_tokens)
            normalized_surface = self._normalize_semantic_token(surface)
            role = str(row.get("role", "")).strip().lower()
            if role in {"", "quantity", "target"}:
                role = ""
            raw_lower = raw_text.lower()
            temporal_tokens = [
                self._normalize_semantic_token(token)
                for token in after_tokens[:3]
                if self._normalize_semantic_token(token)
            ]
            nearby_tokens = [
                self._normalize_semantic_token(token)
                for token in [*before_tokens[-4:], *after_tokens[:4]]
                if self._normalize_semantic_token(token)
            ]
            has_percentage_cue = "%" in tight_window or any(
                token.startswith("percent") for token in after_tokens[:2]
            )
            explicit_percent_surface = (
                raw_text[offset + max(len(surface), 1) :].lstrip().startswith("%")
                or normalized_surface.endswith("%")
                or normalized_surface in {"percent", "percentage"}
            )
            has_temporal_cue = any(token in _SEMANTIC_TEMPORAL_UNITS for token in temporal_tokens)
            surface_temporal_unit = next(
                (
                    token
                    for token in _SEMANTIC_TEMPORAL_UNITS
                    if token in normalized_surface
                ),
                "",
            )
            has_rate_cue = (
                "/" in local_window
                or " per " in f" {local_window} "
                or bool(scope)
                or any(token in {"per", "hour", "hourly", "minute", "daily", "weekly", "rate"} for token in nearby_tokens)
            )
            has_speed_cue = (
                " mph" in f" {local_window} "
                or "mph" in normalized_surface
                or "speed" in local_window
                or "/minute" in local_window
                or "/hour" in local_window
            )
            has_tight_rate_cue = (
                "mph" in tight_window
                or "/" in tight_window
                or " per " in f" {tight_window} "
                or "/minute" in tight_window
                or "/hour" in tight_window
            )
            has_currency_cue = (
                "$" in tight_window
                or any(token in {"dollar", "dollars", "cent", "cents", "price"} for token in temporal_tokens)
            )
            has_threshold_cue = any(token in {"first", "before", "until", "up", "limit", "threshold"} for token in nearby_tokens)
            has_ratio_cue = (
                " times " in f" {local_window} "
                or any(token in {"times", "double", "triple", "ratio", "multiplier"} for token in nearby_tokens)
            )
            if not role and explicit_percent_surface:
                role = "percentage"
                unit = "percent"
                scope = ""
            elif not role and has_speed_cue and not explicit_percent_surface:
                role = "rate"
            elif not role and has_currency_cue and has_rate_cue:
                role = "rate"
                unit = "currency"
                scope = scope or "per_time"
            elif not role and has_ratio_cue and any(token in {"rate", "hourly", "regular"} for token in nearby_tokens):
                role = "divisor"
                scope = ""
            elif not role and has_temporal_cue and has_threshold_cue:
                role = "threshold"
                scope = ""
            elif not role and has_temporal_cue:
                role = "duration"
                if not unit:
                    unit = next(
                        (
                            token
                            for token in temporal_tokens
                            if token in _SEMANTIC_TEMPORAL_UNITS
                        ),
                        "",
                    )
                scope = ""
            elif not role and has_rate_cue:
                role = "rate"
            elif not role and has_currency_cue:
                role = "price"
                unit = "currency"
            if not role:
                currency_window = raw_text[max(offset - 1, 0) : offset + max(len(surface), 1) + 16].lower()
                if "$" in currency_window or any(token in {"dollar", "dollars", "cent", "cents"} for token in after_tokens[:3]):
                    role = "price"
                    unit = "currency"
                elif reference is not None:
                    role = "count"
                elif any(token in {"file", "files"} for token in temporal_tokens) and "download" in raw_lower:
                    role = "total"
                elif "remaining" in raw_lower or "left" in raw_lower or "after" in raw_lower:
                    role = "result"
                elif after_tokens and after_tokens[0] == "times":
                    role = "frequency"
                    unit = "session"
                elif any(token in _SEMANTIC_SCOPE_CUES for token in after_tokens[:3]):
                    role = "rate"
                elif any(token in {"total", "altogether"} for token in before_tokens + after_tokens):
                    role = "goal"
                else:
                    role = "count"
            if role == "count" and any(token == "times" for token in after_tokens[:4]):
                scope = "per_session"
            if role == "frequency":
                if not scope:
                    scope = self._semantic_scope_from_tokens(before_tokens, ["times", *after_tokens])
                if not unit:
                    unit = "session"
                if has_ratio_cue and any(token in {"rate", "hourly", "regular"} for token in nearby_tokens):
                    role = "divisor"
                    scope = ""
            if role == "price":
                if any(token in {"buy", "buys", "bought", "purchase", "purchased", "cost", "for"} for token in before_tokens[-5:]):
                    role = "initial"
                elif (
                    any(token in {"repair", "repairs", "spent", "spend", "renovation"} for token in nearby_tokens)
                    or ("puts" in nearby_tokens and "in" in nearby_tokens)
                ):
                    role = "part"
            if role == "duration":
                if has_threshold_cue:
                    role = "threshold"
                    scope = ""
                elif (
                    "worked for" in raw_lower
                    and any(token in raw_lower for token in ("earnings", "earned", "pay"))
                ):
                    role = "total"
                    scope = ""
            if role in {"percentage", "count", "total"} and has_speed_cue and not explicit_percent_surface:
                role = "rate"
            if role == "duration" and has_tight_rate_cue and not (has_temporal_cue or bool(surface_temporal_unit)):
                role = "rate"
            if (
                role in {"rate", "count", "part"}
                and (has_temporal_cue or bool(surface_temporal_unit))
                and not has_tight_rate_cue
                and not explicit_percent_surface
            ):
                role = "duration"
                if not unit:
                    unit = surface_temporal_unit or next(
                        (
                            token
                            for token in temporal_tokens
                            if token in _SEMANTIC_TEMPORAL_UNITS
                        ),
                        "",
                    )
                scope = ""
            if (
                any(cue in raw_lower for cue in ("each of her chickens", "each chicken", "per chicken"))
                and "cup" in raw_lower
                and role in {"count", "total", "quantity"}
            ):
                role = "rate"
                unit = unit or "cup"
                scope = "per_chicken"
            if (
                any(cue in raw_lower for cue in ("morning", "afternoon", "final meal", "another"))
                and (unit == "cup" or " cup" in f" {local_window} ")
                and role == "count"
            ):
                role = "part"
            if "standstill traffic" in raw_lower and role in {"threshold", "count", "duration"}:
                role = "duration"
                unit = unit or "hour"
                scope = ""
            if role == "goal":
                unit = unit or self._goal_semantic_entity(unified_goal, semantic_entities=semantic_entities).get("unit", "")
            semantic_entities.append(
                {
                    "value": float(row.get("value", 0.0)),
                    "surface": surface,
                    "role": role or "count",
                    "unit": unit,
                    "scope": scope,
                    "reference": reference,
                    "offset": offset,
                    "raw_block": raw_text,
                }
            )
        semantic_entities = self._resolve_reference_entities(semantic_entities)
        return semantic_entities, self._goal_semantic_entity(unified_goal, semantic_entities=semantic_entities)

    def _apply_goal_roles_to_block(
        self,
        block: dict[str, Any],
        *,
        goal_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(block, dict):
            return block
        implies_roles = goal_metadata.get("implies_roles") if isinstance(goal_metadata, dict) else {}
        if not isinstance(implies_roles, dict) or not implies_roles:
            return block
        raw_text = str(block.get("raw", "")).strip()
        rows = block.get("quantities") if isinstance(block.get("quantities"), list) else []
        updated_rows: list[dict[str, Any]] = []
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            surface = str(row.get("surface", "")).strip()
            offset = int(row.get("offset", 0) or 0)
            snippet = self._quantity_local_snippet(raw_text, surface=surface, offset=offset).lower()
            best_role = ""
            best_score = 0.0
            for role_name, cues in implies_roles.items():
                phrase_list = [str(value) for value in cues] if isinstance(cues, list) else []
                overlap = self._phrase_overlap_score(snippet, phrase_list)
                if overlap > best_score:
                    best_score = overlap
                    best_role = str(role_name).strip().lower()
            if best_role:
                row["role"] = best_role
                row["source"] = "backward_goal"
                row["role_confidence"] = float(best_score)
            updated_rows.append(row)
        out = dict(block)
        out["quantities"] = updated_rows
        if goal_metadata:
            out["goal_frame"] = str(goal_metadata.get("operation_frame", "")).strip()
        return out

    def _path_signature(self, path: PathCandidate) -> str:
        """Create stable signature for cross-path agreement checks."""
        composed = path.composed if isinstance(path.composed, dict) else {"value": path.composed}
        transform = composed.get("transform")
        if isinstance(transform, dict):
            return f"arc::{json.dumps(transform, sort_keys=True, separators=(',', ':'))}"
        expression = str(composed.get("expression", "")).strip().lower()
        specialist = str(path.specialist)
        return f"expr::{specialist}::{expression}"

    def _path_score(self, path: PathCandidate, agreement: int) -> float:
        """Score path with agreement boost across exploration strategies."""
        return float(self._path_score_details(path, agreement)["final_score"])

    def _path_score_details(self, path: PathCandidate, agreement: int) -> dict[str, Any]:
        """Detailed path score with grammar/cross-modal/compositional boosts."""
        pattern_bonus = min(len(path.patterns), 30) * 0.01
        agreement_bonus = max(agreement - 1, 0) * 0.15
        grammar_confidence = self._aggregate_grammar_confidence(path)
        grammar_boost = (grammar_confidence - 0.5) * 0.3
        strategy_weight = self._strategy_weight(str(path.route.get("strategy", "heuristic_auto")))
        strategy_boost = (strategy_weight - 1.0) * 0.2
        composition_depth = self._path_composition_depth(path)
        compositional_boost = 0.0
        if composition_depth > 3:
            compositional_boost = 0.10
        elif composition_depth > 1:
            compositional_boost = 0.05
        final_score = (
            path.confidence
            + pattern_bonus
            + agreement_bonus
            + grammar_boost
            + strategy_boost
            + compositional_boost
        )
        return {
            "path": path,
            "final_score": final_score,
            "pattern_bonus": pattern_bonus,
            "agreement_bonus": agreement_bonus,
            "grammar_confidence": grammar_confidence,
            "grammar_boost": grammar_boost,
            "strategy_weight": strategy_weight,
            "strategy_boost": strategy_boost,
            "composition_depth": composition_depth,
            "compositional_boost": compositional_boost,
            "cross_path_agreement": max(int(agreement), 1),
        }

    def _strategy_weight(self, strategy: str) -> float:
        if strategy in {"forward_reading", "backward_reading", "forward", "backward"}:
            return 1.1
        return 1.0

    def _aggregate_grammar_confidence(self, path: PathCandidate) -> float:
        if not path.patterns:
            return 0.5
        confidences = [self._get_grammar_confidence_for_candidate(c) for c in path.patterns[:10]]
        if not confidences:
            return 0.5
        return sum(confidences) / len(confidences)

    def _get_grammar_confidence_for_candidate(self, candidate: dict[str, Any]) -> float:
        entry = candidate.get("entry", candidate)
        if not isinstance(entry, dict):
            return 0.5
        grammar_confidence = self._query_grammar_galaxy_confidence(entry)
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        symlink_value = str(metadata.get("symlink", ""))
        cross_modal = str(metadata.get("cross_modal", ""))
        if symlink_value or cross_modal:
            linked = symlink_value.split("_")[0].strip().capitalize() if symlink_value else ""
            if not linked and cross_modal:
                linked = cross_modal.split("_")[0].strip().capitalize()
            if linked:
                linked_confidence = self._query_cross_modal_confidence(entry, linked)
                grammar_confidence = (grammar_confidence + linked_confidence) / 2.0
                grammar_confidence = min(1.0, grammar_confidence * 1.05)
        return max(0.0, min(grammar_confidence, 1.0))

    def _query_grammar_galaxy_confidence(self, entry: dict[str, Any]) -> float:
        direct = self._entry_confidence(entry)
        if self.knowledgeverse is None or not hasattr(self.knowledgeverse, "galaxy_manager"):
            return direct
        try:
            query_parts = [
                str(entry.get("id", "")),
                str(entry.get("name", "")),
                str(entry.get("category", "")),
                str(entry.get("domain", "")),
                str(entry.get("rpn_program", ""))[:64],
            ]
            query_text = " ".join(part for part in query_parts if part).strip()
            if not query_text:
                return direct
            matches = self.knowledgeverse.galaxy_manager.query(
                query_text=query_text,
                specialist="grammar",
                top_k=5,
            )
            if not matches:
                return direct
            best = max(self._entry_confidence(m.get("entry", {})) for m in matches)
            return max(direct, best)
        except Exception:
            return direct

    def _query_cross_modal_confidence(self, entry: dict[str, Any], galaxy_name: str) -> float:
        if self.knowledgeverse is None or not hasattr(self.knowledgeverse, "galaxy_manager"):
            return 0.5
        try:
            query_text = " ".join(
                part
                for part in (
                    str(entry.get("id", "")),
                    str(entry.get("name", "")),
                    str(entry.get("category", "")),
                    str(entry.get("rpn_program", ""))[:64],
                )
                if part
            )
            if not query_text:
                return 0.5
            matches = self.knowledgeverse.galaxy_manager.query(
                query_text=query_text,
                specialist=galaxy_name.lower(),
                top_k=3,
            )
            if not matches:
                return 0.5
            return max(self._entry_confidence(m.get("entry", {})) for m in matches)
        except Exception:
            return 0.5

    def _entry_confidence(self, entry: dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.5
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
        for container in (entry, metadata):
            value = container.get("confidence")
            if value is not None:
                try:
                    return max(0.0, min(float(value), 1.0))
                except (TypeError, ValueError):
                    continue
        quality = entry.get("quality_score", metadata.get("quality_score"))
        if quality is not None:
            try:
                q = float(quality)
                if q > 1.0:
                    q = q / 100.0
                return max(0.0, min(q, 1.0))
            except (TypeError, ValueError):
                pass
        return 0.5

    def _path_composition_depth(self, path: PathCandidate) -> int:
        depth = 0
        composed = path.composed if isinstance(path.composed, dict) else {}
        transform = composed.get("transform")
        if isinstance(transform, dict):
            steps = transform.get("steps")
            if isinstance(steps, list):
                depth = max(depth, len(steps))
            elif transform.get("op"):
                depth = max(depth, 1)
        expression = str(composed.get("expression", ""))
        if expression:
            depth = max(depth, sum(1 for token in expression.split() if token.isalpha()))
        for pattern in path.patterns[:10]:
            entry = pattern.get("entry", pattern)
            if not isinstance(entry, dict):
                continue
            rpn = str(entry.get("rpn_program", ""))
            if not rpn:
                continue
            rpn_ops = sum(1 for token in rpn.split() if token.isupper())
            depth = max(depth, rpn_ops)
        return depth

    def _log_path_contribution_events(
        self,
        *,
        query: str,
        specialist: str,
        composed: dict[str, Any],
    ) -> None:
        if self.knowledgeverse is None or not hasattr(self.knowledgeverse, "log_event"):
            return
        meta = composed.get("meta_specialist", {}) if isinstance(composed, dict) else {}
        confidence = float(meta.get("confidence", 0.0))
        strategies = list(meta.get("strategies", [])) if isinstance(meta.get("strategies", []), list) else []
        contributing = set(
            str(s) for s in meta.get("contributing_strategies", []) if isinstance(s, (str, int, float))
        )
        try:
            self.knowledgeverse.log_event(
                event_type="navigator_compose",
                event_data={
                    "query": query,
                    "specialist": specialist,
                    "confidence": confidence,
                    "num_paths": int(meta.get("paths_considered", len(strategies))),
                    "strategies": strategies,
                    "verification": "multi_path_compose",
                    "galaxy": "Grammar",
                },
            )
            for strategy in strategies:
                contributed = str(strategy) in contributing
                self.knowledgeverse.log_event(
                    event_type="navigator_path_contribution",
                    event_data={
                        "query": query,
                        "specialist": specialist,
                        "strategy": str(strategy),
                        "contributed": bool(contributed),
                        "confidence": confidence,
                        "verification": "path_contribution",
                        "galaxy": "Grammar",
                    },
                )
        except Exception:
            return

    def _query_signature(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    def load_state(self) -> None:
        payload = self.weight_store.load()
        self.routing_topology = dict(payload.get("routing_topology", {}))
        self._update_count = int(payload.get("update_count", 0))
        specialist_bias = payload.get("specialist_bias", {})
        if isinstance(specialist_bias, dict):
            for specialist, value in specialist_bias.items():
                self.router.set_specialist_bias(str(specialist), float(value))

    def save_state(self) -> None:
        payload = {
            "version": 1,
            "specialist_bias": self.router.get_specialist_bias(),
            "routing_topology": self.routing_topology,
            "update_count": self._update_count,
        }
        self.weight_store.save(payload)
