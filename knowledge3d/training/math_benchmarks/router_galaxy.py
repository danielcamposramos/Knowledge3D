"""
Router Galaxy: persistent memory of routing experiences.

Stores high-confidence routing events so the router can be retrained
from accumulated experience (continual learning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


def _stable_id(text: str, *, suffix: str) -> str:
    digest = hashlib.sha256(f"{text}|{suffix}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class RouterGalaxyEntry:
    entry_id: str
    problem_text: str
    embedding: List[float]
    router_logit: float
    router_use_specialist: bool
    solver: str
    correct: bool
    dataset: str
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "problem_text": self.problem_text,
            "embedding": self.embedding,
            "router_logit": self.router_logit,
            "router_use_specialist": self.router_use_specialist,
            "solver": self.solver,
            "correct": self.correct,
            "dataset": self.dataset,
            "label": self.label,
            "metadata": self.metadata,
        }


class RouterGalaxy:
    """In-memory Router Galaxy container with JSONL export."""

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[RouterGalaxyEntry] = []

    def add_event(
        self,
        *,
        problem_text: str,
        router_logit: float,
        router_use_specialist: bool,
        solver: str,
        correct: bool,
        dataset: str,
        label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouterGalaxyEntry:
        embedding = embed_text(problem_text, dim=self.embedding_dim)
        entry_id = _stable_id(problem_text, suffix=f"{router_logit}:{solver}:{dataset}")
        entry = RouterGalaxyEntry(
            entry_id=entry_id,
            problem_text=problem_text,
            embedding=embedding,
            router_logit=float(router_logit),
            router_use_specialist=bool(router_use_specialist),
            solver=str(solver),
            correct=bool(correct),
            dataset=str(dataset),
            label=int(label) if label is not None else None,
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def to_jsonl(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")


__all__ = ["RouterGalaxy", "RouterGalaxyEntry"]
