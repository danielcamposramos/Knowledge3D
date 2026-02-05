"""
Sleep Galaxy: store memory-keeper decisions (keep/discard/uncertain).

Phase 4.0 bootstraps the Sleep Keeper with heuristic labels and then
lets the specialist curate what gets consolidated into long-term memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


@dataclass
class SleepGalaxyEntry:
    trace_id: str
    problem_text: str
    embedding: List[float]
    decision: str
    decision_score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_text": self.problem_text,
            "embedding": self.embedding,
            "decision": self.decision,
            "decision_score": self.decision_score,
            "metadata": self.metadata,
        }


class SleepGalaxy:
    """
    In-memory Sleep Galaxy container with JSONL export.

    Embeddings are generated using the sovereign router embedder to keep
    locality consistent with other navigation systems.
    """

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[SleepGalaxyEntry] = []

    def add_entry(
        self,
        *,
        trace_id: str,
        problem_text: str,
        decision: str,
        decision_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SleepGalaxyEntry:
        embedding = embed_text(problem_text, dim=self.embedding_dim)
        entry = SleepGalaxyEntry(
            trace_id=trace_id,
            problem_text=problem_text,
            embedding=embedding,
            decision=str(decision),
            decision_score=float(decision_score),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def to_jsonl(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")


__all__ = ["SleepGalaxy", "SleepGalaxyEntry"]
