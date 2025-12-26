"""
Tsinghua-inspired Galaxy explorer (Phase 5B).

This is a lightweight, Python-side adaptation of the "cluster → scout → expand"
idea for K3D math reading:
- Cluster tokens into semantic buckets (quantity/rate/operation/aggregation/...)
- Scout for intersection "hub" concepts across candidate rules (top-K, no full sort)
- Expand from hubs by selecting a focused subset of word_sequence rules

The goal is not perfect shortest paths yet; it is to:
1) reduce irrelevant rule checks / spurious matches
2) surface exploration traces for learning / diagnostics
3) provide a stable interface we can later accelerate (PTX/RPN opcodes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ExplorationResult:
    buckets: Dict[str, List[Tuple[str, int]]]
    hub_concepts: List[str]
    selected_rule_ids: List[str]
    selected_rules: List[Any]
    rule_checks: int


def _top_k_counts(counts: Dict[str, int], k: int) -> List[str]:
    """
    Return top-k keys by count using O(n*k) insertion (no full sort).
    """
    if k <= 0 or not counts:
        return []
    top: List[Tuple[str, int]] = []
    for key, val in counts.items():
        inserted = False
        for i, (_, cur) in enumerate(top):
            if val > cur:
                top.insert(i, (key, val))
                inserted = True
                break
        if not inserted:
            top.append((key, val))
        if len(top) > k:
            top.pop()
    return [k_ for k_, _v in top]


class TsinghuaGalaxyExplorer:
    """
    Minimal explorer that selects a small subset of word_sequence rules.
    """

    _BUCKET_TO_DOMAINS = {
        "quantity": {"math_extraction"},
        "rate": {"math_rate", "math_operation"},
        "operation": {"math_operation", "math_arithmetic"},
        "aggregation": {"math_aggregation"},
        "comparison": {"math_operation"},
        "temporal": {"math_operation", "math_extraction"},
    }

    def __init__(
        self,
        *,
        max_rules: int = 40,
        hub_k: int = 5,
        reality_galaxy: Any | None = None,
        drawing_galaxy: Any | None = None,
        generic_equations_galaxy: Any | None = None,
    ) -> None:
        self.max_rules = int(max(5, max_rules))
        self.hub_k = int(max(1, hub_k))
        self.reality_galaxy = reality_galaxy
        self.drawing_galaxy = drawing_galaxy
        self.generic_equations_galaxy = generic_equations_galaxy

    def cluster_concepts(self, entries: Sequence[Any]) -> Dict[str, List[Tuple[str, int]]]:
        buckets: Dict[str, List[Tuple[str, int]]] = {
            "quantity": [],
            "rate": [],
            "operation": [],
            "aggregation": [],
            "comparison": [],
            "temporal": [],
        }

        for idx, entry in enumerate(entries):
            norm = str(getattr(entry, "normalized", "") or "").lower()
            cat = str(getattr(entry, "category", "") or "").lower()
            if cat == "number":
                buckets["quantity"].append((norm, idx))
                continue

            if norm in {"per", "each", "every"}:
                buckets["rate"].append((norm, idx))
                continue

            if norm in {"total", "altogether", "sum", "combined", "in"} or cat == "aggregation":
                buckets["aggregation"].append((norm, idx))
                continue

            if norm in {"more", "less", "difference", "than"}:
                buckets["comparison"].append((norm, idx))
                continue

            if norm in {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"}:
                buckets["temporal"].append((norm, idx))
                continue
            if norm in {"before", "after", "then", "next", "yesterday", "today", "tomorrow"}:
                buckets["temporal"].append((norm, idx))
                continue

            if norm in {"+", "-", "*", "/", "plus", "minus", "times", "divided", "divide", "split", "shared", "share", "half", "twice", "double", "triple"}:
                buckets["operation"].append((norm, idx))
                continue

        return buckets

    def _rule_concepts(self, rule: Any) -> List[str]:
        rid = str(getattr(rule, "rule_id", "") or "")
        dom = str(getattr(rule, "domain", "") or "")
        concepts: List[str] = []

        if dom == "math_extraction":
            concepts.append("quantity")
        if dom == "math_aggregation":
            concepts.append("aggregation")
        if dom == "math_rate":
            concepts.append("rate")
        if dom in {"math_operation", "math_arithmetic"}:
            concepts.append("operation")

        # Refine by rule_id cues (cheap tags).
        for key, tag in (
            ("rate", "rate"),
            ("each_cost", "rate"),
            ("shared", "division"),
            ("divide", "division"),
            ("divided", "division"),
            ("percent", "percent"),
            ("more_than", "comparison"),
            ("less_than", "comparison"),
            ("twice", "multiplication"),
            ("n_times", "multiplication"),
            ("times", "multiplication"),
            ("half", "fraction"),
            ("gave", "subtraction"),
            ("spent", "subtraction"),
            ("lost", "subtraction"),
            ("received", "addition"),
            ("plus", "addition"),
            ("minus", "subtraction"),
        ):
            if key in rid and tag not in concepts:
                concepts.append(tag)

        return concepts

    def scout_intersections(self, candidate_rules: Sequence[Any]) -> List[str]:
        counts: Dict[str, int] = {}
        for rule in candidate_rules:
            for c in self._rule_concepts(rule):
                counts[c] = counts.get(c, 0) + 1

        # Cross-domain bump: if Reality/Drawing galaxies are present, lightly bias hubs
        # toward concepts that are shared across modalities.
        if self.reality_galaxy is not None:
            counts["physics"] = counts.get("physics", 0) + 1
        if self.drawing_galaxy is not None:
            counts["geometry"] = counts.get("geometry", 0) + 1
        if self.generic_equations_galaxy is not None:
            counts["generic"] = counts.get("generic", 0) + 1
        return _top_k_counts(counts, self.hub_k)

    def expand_from_hubs(self, candidate_rules: Sequence[Any], hubs: Sequence[str]) -> List[Any]:
        if not hubs:
            return list(candidate_rules)
        hub_set = {str(h) for h in hubs if h}
        selected: List[Any] = []
        for rule in candidate_rules:
            cset = set(self._rule_concepts(rule))
            if cset.intersection(hub_set):
                selected.append(rule)
        return selected

    def explore(self, *, entries: Sequence[Any], rule_bank: Sequence[Any]) -> ExplorationResult:
        buckets = self.cluster_concepts(entries)
        active_buckets = [k for k, v in buckets.items() if v]

        # Candidate rules: domain-filtered (clustering).
        domains: set[str] = set()
        for b in active_buckets:
            domains |= set(self._BUCKET_TO_DOMAINS.get(b, set()))
        if not domains:
            domains = {"math_extraction", "math_operation", "math_arithmetic", "math_rate", "math_aggregation"}

        candidates: List[Any] = []
        for rule in rule_bank:
            dom = str(getattr(rule, "domain", "") or "")
            if dom in domains:
                candidates.append(rule)

        hubs = self.scout_intersections(candidates)
        expanded = self.expand_from_hubs(candidates, hubs)

        # Cap without sorting: keep first N in stable order, but prefer rules whose rule_id mentions hubs.
        preferred: List[Any] = []
        fallback: List[Any] = []
        hub_words = set(hubs)
        for rule in expanded:
            rid = str(getattr(rule, "rule_id", "") or "")
            if any(h in rid for h in hub_words):
                preferred.append(rule)
            else:
                fallback.append(rule)
        selected_rules = (preferred + fallback)[: self.max_rules]
        selected_rule_ids = [str(getattr(r, "rule_id", "") or "") for r in selected_rules if getattr(r, "rule_id", None)]

        return ExplorationResult(
            buckets=buckets,
            hub_concepts=list(hubs),
            selected_rule_ids=selected_rule_ids,
            selected_rules=selected_rules,
            rule_checks=len(selected_rules),
        )


__all__ = ["TsinghuaGalaxyExplorer", "ExplorationResult"]
