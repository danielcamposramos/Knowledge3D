"""
Oracle Galaxy: store self-generated problems for Phase 5 (The Oracle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


def _stable_id(template_id: str, generated_text: str, mutation_type: str) -> str:
    digest = hashlib.sha256(f"{template_id}|{mutation_type}|{generated_text}".encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class OracleGalaxyEntry:
    entry_id: str
    template_id: str
    mutation_type: str
    generated_text: str
    embedding: List[float]
    verified: bool
    complexity_score: Optional[float] = None
    higher_order: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "template_id": self.template_id,
            "mutation_type": self.mutation_type,
            "generated_text": self.generated_text,
            "embedding": self.embedding,
            "verified": self.verified,
            "complexity_score": self.complexity_score,
            "higher_order": self.higher_order,
            "metadata": self.metadata,
        }


class OracleGalaxy:
    """
    In-memory Oracle Galaxy container with JSONL export.

    Stores self-generated problems verified by Navigation Specialist V5.
    """

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[OracleGalaxyEntry] = []

    def add_entry(
        self,
        *,
        template_id: str,
        mutation_type: str,
        generated_text: str,
        verified: bool,
        complexity_score: Optional[float] = None,
        higher_order: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OracleGalaxyEntry:
        embedding = embed_text(generated_text, dim=self.embedding_dim)
        entry_id = _stable_id(template_id, generated_text, mutation_type)
        meta = dict(metadata or {})
        meta.setdefault("label", template_id)
        meta.setdefault("geometry", "octahedron")
        meta.setdefault("color", [0.0, 1.0, 1.0])
        entry = OracleGalaxyEntry(
            entry_id=entry_id,
            template_id=str(template_id),
            mutation_type=str(mutation_type),
            generated_text=str(generated_text),
            embedding=embedding,
            verified=bool(verified),
            complexity_score=complexity_score if complexity_score is None else float(complexity_score),
            higher_order=bool(higher_order) if higher_order is not None else None,
            metadata=meta,
        )
        self.entries.append(entry)
        return entry

    def to_jsonl(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")


__all__ = ["OracleGalaxy", "OracleGalaxyEntry"]
