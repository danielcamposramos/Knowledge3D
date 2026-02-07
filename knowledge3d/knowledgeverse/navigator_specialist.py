"""Navigator meta-specialist for multi-path routing and composition."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
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
        max_paths: int = 3,
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
        _append(base, "heuristic_auto")

        for route in self._routes_from_topology(query):
            _append(route, "topology_memory")

        for route in self._routes_from_legacy_keywords(query):
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
        )

        candidates: list[PathCandidate] = []
        for route in routes:
            patterns = trm_navigator.query(
                query=query,
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

        ranked = sorted(
            paths,
            key=lambda p: (p.confidence, len(p.patterns)),
            reverse=True,
        )
        best = ranked[0]
        final_program = dict(best.composed)
        final_program["specialist"] = best.specialist
        final_program["route"] = dict(best.route)
        final_program["meta_specialist"] = {
            "mode": "multi_path",
            "paths_considered": len(paths),
            "component_specialists": [p.specialist for p in ranked],
            "primary_specialist": best.specialist,
            "confidence": best.confidence,
        }
        return final_program

    def navigate_and_compose(
        self,
        trm_navigator: Any,
        query: str,
        *,
        use_enriched: bool = True,
        specialist: str = "auto",
        domain_hint: str | None = None,
        galaxy_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Full meta-specialist flow: plan -> explore -> compose."""
        paths = self.explore_multi_path(
            trm_navigator=trm_navigator,
            query=query,
            use_enriched=use_enriched,
            specialist=specialist,
            domain_hint=domain_hint,
            galaxy_names=galaxy_names,
        )
        return self.compose_paths(query=query, paths=paths)

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
        return max(0.05, min(base, 0.99))

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
