"""Navigator meta-specialist for multi-path routing and composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .specialist_router import SpecialistRouter
from .trm_weight_store import TRMWeightStore


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
        success: bool,
    ) -> None:
        """Update topology memory with route outcome signal."""
        signature = self._query_signature(query)
        specialist_stats = self.routing_topology.setdefault(signature, {})
        bucket = specialist_stats.setdefault(specialist, {"success": 0, "failure": 0})
        if success:
            bucket["success"] += 1
            self.router.adjust_specialist_bias(specialist, +0.02)
        else:
            bucket["failure"] += 1
            self.router.adjust_specialist_bias(specialist, -0.01)
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
        dependencies = [self._extract_variables_and_constraints(s) for s in reverse_clauses[1:]]
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

        forward_context = list(forward.get("forward_parse", {}).get("context", []))
        backward_deps = list(backward.get("backward_parse", {}).get("dependencies", []))

        merged_vars: dict[str, Any] = {}

        def _merge_var_block(block: dict[str, Any]) -> None:
            if str(block.get("type", "")) != "variables":
                return
            data = block.get("data", {})
            if not isinstance(data, dict):
                return
            for key, value in data.items():
                merged_vars[str(key)] = value

        for ctx in forward_context:
            if isinstance(ctx, dict):
                _merge_var_block(ctx)
        for dep in backward_deps:
            if isinstance(dep, dict):
                _merge_var_block(dep)

        f_goal = forward.get("forward_parse", {}).get("goal", {})
        b_goal = backward.get("backward_parse", {}).get("goal", {})
        f_expr = str(f_goal.get("expression", "")) if isinstance(f_goal, dict) else ""
        b_expr = str(b_goal.get("expression", "")) if isinstance(b_goal, dict) else ""
        unified_goal = f_goal if len(f_expr) >= len(b_expr) else b_goal
        if not isinstance(unified_goal, dict):
            unified_goal = {"type": "goal", "raw": query}

        var_str = ", ".join(f"{k}={v}" for k, v in merged_vars.items())
        goal_raw = str(unified_goal.get("raw") or unified_goal.get("expression") or query)
        query_variant = f"Given {var_str}, {goal_raw}".strip() if var_str else goal_raw

        return {
            **base_route,
            "query_variant": query_variant or query,
            "fusion_parse": {
                "merged_variables": merged_vars,
                "unified_goal": unified_goal,
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
        return {
            "type": "variables" if assignments else "context",
            "data": assignments,
            "raw": sentence,
        }

    def _extract_goal(self, sentence: str) -> dict[str, Any]:
        """Extract goal-like expression from a sentence."""
        lowered = sentence.lower()
        goal_markers = ("what is", "find", "compute", "calculate", "determine", "solve for")
        for marker in goal_markers:
            idx = lowered.find(marker)
            if idx >= 0:
                expr = sentence[idx + len(marker) :].strip(" ?.")
                return {"type": "goal", "operation": "evaluate", "expression": expr, "raw": sentence}
        return {"type": "goal", "raw": sentence}

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
