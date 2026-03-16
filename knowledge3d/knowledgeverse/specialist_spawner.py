"""Autonomous specialist spawning from usage/performance signals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .specialist_base import SpecialistBase, _resolve_ternary


@dataclass
class SpawnDecision:
    parent: str
    child: str
    domain: str
    reason: str
    query_count: int
    low_conf_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "child": self.child,
            "domain": self.domain,
            "reason": self.reason,
            "query_count": self.query_count,
            "low_conf_count": self.low_conf_count,
        }


class SpecialistSpawner:
    """Tracks subdomain pressure and spawns workers when triggers are met."""

    def __init__(
        self,
        *,
        root: SpecialistBase,
        storage_path: str | Path | None = None,
        frequency_threshold: int = 100,
        low_confidence_threshold: float = 0.6,
        low_confidence_min_samples: int = 20,
        max_children_per_parent: int = 16,
    ):
        self.root = root
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.frequency_threshold = int(frequency_threshold)
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.low_confidence_min_samples = int(low_confidence_min_samples)
        self.max_children_per_parent = int(max_children_per_parent)
        self.stats: dict[str, dict[str, dict[str, int]]] = {}
        self.decisions: list[dict[str, Any]] = []
        self._load()

    def observe(
        self,
        *,
        parent: SpecialistBase,
        query: str,
        confidence: float,
        success: bool | None = None,
        ternary_outcome: int | None = None,
        domain_hint: str | None = None,
    ) -> SpawnDecision | None:
        outcome = _resolve_ternary(success, ternary_outcome)
        bucket = self._infer_bucket(
            parent=parent,
            query=query,
            domain_hint=domain_hint,
        )
        parent_stats = self.stats.setdefault(parent.name, {})
        metrics = parent_stats.setdefault(
            bucket,
            {"count": 0, "low_conf": 0, "success": 0, "failure": 0, "uncertain": 0},
        )
        metrics["count"] += 1
        if float(confidence) < self.low_confidence_threshold:
            metrics["low_conf"] += 1
        if outcome > 0:
            metrics["success"] += 1
        elif outcome < 0:
            metrics["failure"] += 1
        else:
            metrics["uncertain"] += 1

        decision = self._evaluate(parent=parent, bucket=bucket, metrics=metrics)
        if decision is not None:
            self.decisions.append(decision.as_dict())
            self._save()
        return decision

    def persist(self) -> None:
        """Persist current stats/decisions snapshot to disk (if enabled)."""
        self._save()

    def _evaluate(
        self,
        *,
        parent: SpecialistBase,
        bucket: str,
        metrics: dict[str, int],
    ) -> SpawnDecision | None:
        if len(parent.children) >= self.max_children_per_parent:
            return None
        if any(child.domain.lower() == bucket for child in parent.children.values()):
            return None

        should_spawn = False
        reason = ""
        if metrics["count"] >= self.frequency_threshold:
            should_spawn = True
            reason = "frequency_threshold"
        elif (
            metrics["count"] >= self.low_confidence_min_samples
            and metrics["low_conf"] >= max(1, int(0.5 * metrics["count"]))
        ):
            should_spawn = True
            reason = "performance_gap"

        if not should_spawn:
            return None

        child_name = self._child_name(bucket=bucket)
        child = parent.spawn_child(name=child_name, domain=bucket)
        parent.routing_bias.setdefault(child.name, 0.55)
        return SpawnDecision(
            parent=parent.name,
            child=child.name,
            domain=bucket,
            reason=reason,
            query_count=int(metrics["count"]),
            low_conf_count=int(metrics["low_conf"]),
        )

    def _infer_bucket(
        self,
        *,
        parent: SpecialistBase,
        query: str,
        domain_hint: str | None,
    ) -> str:
        if domain_hint:
            return self._sanitize_bucket(domain_hint)
        q = query.lower()
        domain = parent.domain.lower()
        if domain == "math":
            if any(tok in q for tok in ("topology", "manifold", "homology")):
                return "topology"
            if any(tok in q for tok in ("number", "prime", "mod", "divisor")):
                return "number_theory"
            if any(tok in q for tok in ("derivative", "integral", "calculus", "matrix", "vector")):
                return "calculus_linear_algebra"
            if any(tok in q for tok in ("add", "subtract", "multiply", "divide", "arithmetic")):
                return "basic_math"
            return "general_math"
        if domain == "visual":
            if any(tok in q for tok in ("mesh", "vertex", "face", "ray", "3d", "voxel")):
                return "3d_visual"
            if any(tok in q for tok in ("arc", "grid", "pattern", "transform")):
                return "spatial_reasoning"
            return "2d_visual"
        if domain == "reality":
            if any(tok in q for tok in ("force", "mass", "energy", "velocity", "acceleration")):
                return "physics"
            if any(tok in q for tok in ("atom", "molecule", "reaction")):
                return "chemistry"
            if any(tok in q for tok in ("cell", "dna", "gene", "evolution")):
                return "biology"
            return "procedural_systems"
        if domain == "language":
            if any(tok in q for tok in ("syntax", "grammar", "parse")):
                return "grammar"
            if any(tok in q for tok in ("semantic", "meaning", "context")):
                return "semantics"
            return "pragmatics"
        return self._sanitize_bucket(parent.domain)

    def _child_name(self, *, bucket: str) -> str:
        words = [w.capitalize() for w in re.split(r"[^a-z0-9]+", bucket.lower()) if w]
        stem = "".join(words) or "Adaptive"
        return f"{stem}Specialist"

    def _sanitize_bucket(self, raw: str) -> str:
        return "_".join(tok for tok in re.split(r"[^a-z0-9]+", raw.lower()) if tok) or "general"

    def _load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        raw_stats = payload.get("stats", {})
        raw_decisions = payload.get("decisions", [])
        if isinstance(raw_stats, dict):
            self.stats = {
                str(parent): {
                    str(bucket): {
                        "count": int(metrics.get("count", 0)),
                        "low_conf": int(metrics.get("low_conf", 0)),
                        "success": int(metrics.get("success", 0)),
                        "failure": int(metrics.get("failure", 0)),
                        "uncertain": int(metrics.get("uncertain", 0)),
                    }
                    for bucket, metrics in buckets.items()
                    if isinstance(metrics, dict)
                }
                for parent, buckets in raw_stats.items()
                if isinstance(buckets, dict)
            }
        if isinstance(raw_decisions, list):
            self.decisions = [dict(item) for item in raw_decisions if isinstance(item, dict)]

    def _save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stats": self.stats,
            "decisions": self.decisions[-500:],
        }
        self.storage_path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
