"""Matryoshka specialist base primitives for fractal specialist hierarchies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SpecialistDelta:
    """Compact LoRA-like delta descriptor (metadata, not a full tensor copy)."""

    rank: int
    alpha: float
    seed: int
    scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "alpha": float(self.alpha),
            "seed": int(self.seed),
            "scale": float(self.scale),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpecialistDelta":
        return cls(
            rank=int(payload.get("rank", 64)),
            alpha=float(payload.get("alpha", 1.0)),
            seed=int(payload.get("seed", 0)),
            scale=float(payload.get("scale", 1.0)),
        )


class SpecialistBase:
    """
    Base specialist node for self-similar master/worker specialist trees.

    This structure is intentionally lightweight and orchestration-focused.
    Hot-path reasoning remains in sovereign TRM/RPN execution.
    """

    def __init__(
        self,
        *,
        name: str,
        domain: str,
        level: int = 0,
        parent: "SpecialistBase | None" = None,
        delta: SpecialistDelta | None = None,
        storage_dir: str | Path | None = None,
    ):
        self.name = str(name)
        self.domain = str(domain)
        self.level = int(level)
        self.parent = parent
        self.children: dict[str, SpecialistBase] = {}
        self.routing_bias: dict[str, float] = {}
        self.query_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.delta = delta if delta is not None else self._default_delta(level=self.level)

    def route(self, query: str, domain_hint: str | None = None) -> "SpecialistBase":
        """
        Route to best child; if leaf specialist, return self.
        """
        if not self.children:
            return self
        scores = self._score_children(query=query, domain_hint=domain_hint)
        if not scores:
            return self
        winner = max(scores.items(), key=lambda item: item[1])[0]
        return self.children[winner]

    def spawn_child(
        self,
        *,
        name: str,
        domain: str,
        rank: int | None = None,
        alpha: float = 1.0,
    ) -> "SpecialistBase":
        if name in self.children:
            return self.children[name]
        child = SpecialistBase(
            name=name,
            domain=domain,
            level=self.level + 1,
            parent=self,
            delta=SpecialistDelta(
                rank=max(8, int(rank if rank is not None else self._child_rank())),
                alpha=float(alpha),
                seed=self._child_seed(name=name),
                scale=max(0.05, self.delta.scale * 0.75),
            ),
            storage_dir=self.storage_dir,
        )
        self.children[name] = child
        self.routing_bias.setdefault(name, 0.5)
        return child

    def iter_tree(self) -> list["SpecialistBase"]:
        out = [self]
        for child in self.children.values():
            out.extend(child.iter_tree())
        return out

    def find(self, name: str) -> "SpecialistBase | None":
        if self.name == name:
            return self
        for child in self.children.values():
            found = child.find(name)
            if found is not None:
                return found
        return None

    def update_routing_bias(self, child_name: str, success: bool, *, alpha: float = 0.1) -> None:
        current = float(self.routing_bias.get(child_name, 0.5))
        target = 1.0 if success else 0.0
        updated = (float(alpha) * target) + ((1.0 - float(alpha)) * current)
        self.routing_bias[child_name] = max(0.0, min(updated, 1.0))

    def mark_query(self, success: bool) -> None:
        self.query_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def effective_delta_chain(self) -> list[SpecialistDelta]:
        chain: list[SpecialistDelta] = []
        node: SpecialistBase | None = self
        while node is not None:
            chain.append(node.delta)
            node = node.parent
        chain.reverse()
        return chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "level": self.level,
            "delta": self.delta.as_dict(),
            "routing_bias": dict(self.routing_bias),
            "query_count": int(self.query_count),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "children": [child.to_dict() for child in self.children.values()],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        parent: "SpecialistBase | None" = None,
        storage_dir: str | Path | None = None,
    ) -> "SpecialistBase":
        node = cls(
            name=str(payload.get("name", "Specialist")),
            domain=str(payload.get("domain", "generic")),
            level=int(payload.get("level", 0)),
            parent=parent,
            delta=SpecialistDelta.from_dict(dict(payload.get("delta", {}))),
            storage_dir=storage_dir,
        )
        node.routing_bias = {
            str(key): float(value)
            for key, value in dict(payload.get("routing_bias", {})).items()
        }
        node.query_count = int(payload.get("query_count", 0))
        node.success_count = int(payload.get("success_count", 0))
        node.failure_count = int(payload.get("failure_count", 0))
        for child_payload in list(payload.get("children", [])):
            if not isinstance(child_payload, dict):
                continue
            child = cls.from_dict(
                child_payload,
                parent=node,
                storage_dir=storage_dir,
            )
            node.children[child.name] = child
            node.routing_bias.setdefault(child.name, 0.5)
        return node

    def _score_children(self, query: str, domain_hint: str | None) -> dict[str, float]:
        scores: dict[str, float] = {}
        query_l = query.lower()
        hint_l = (domain_hint or "").lower()
        for child_name, child in self.children.items():
            score = float(self.routing_bias.get(child_name, 0.5))
            if hint_l and hint_l in child.domain.lower():
                score += 0.25
            score += self._keyword_match_score(query_l=query_l, child=child)
            scores[child_name] = score
        return scores

    def _keyword_match_score(self, *, query_l: str, child: "SpecialistBase") -> float:
        tokens = {tok for tok in re.split(r"[^a-z0-9_]+", query_l) if tok}
        child_tokens = {tok for tok in re.split(r"[^a-z0-9_]+", child.name.lower()) if tok}
        domain_tokens = {tok for tok in re.split(r"[^a-z0-9_]+", child.domain.lower()) if tok}
        overlap = len(tokens & (child_tokens | domain_tokens))
        if overlap <= 0:
            return 0.0
        return min(0.35, overlap * 0.08)

    def _default_delta(self, *, level: int) -> SpecialistDelta:
        base_rank = max(8, 64 // max(level + 1, 1))
        return SpecialistDelta(
            rank=base_rank,
            alpha=1.0,
            seed=self._child_seed(name=self.name),
            scale=max(0.1, 1.0 / max(level + 1, 1)),
        )

    def _child_rank(self) -> int:
        return max(8, int(self.delta.rank * 0.75))

    def _child_seed(self, *, name: str) -> int:
        return abs(hash((self.name, self.level, name))) % (2**31 - 1)
