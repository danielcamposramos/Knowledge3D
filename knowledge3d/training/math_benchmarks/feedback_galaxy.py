"""
Feedback Galaxy: store teacher evaluations for RLWHF.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


@dataclass
class FeedbackEntry:
    trace_id: str
    problem_text: str
    teacher_score: int
    feedback_text: str
    suggested_rule: str
    embedding: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_text": self.problem_text,
            "teacher_score": self.teacher_score,
            "feedback_text": self.feedback_text,
            "suggested_rule": self.suggested_rule,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


class FeedbackGalaxy:
    """
    In-memory collection of teacher feedback with JSONL export.
    """

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[FeedbackEntry] = []

    def add_feedback(
        self,
        *,
        trace_id: str,
        problem_text: str,
        teacher_score: int,
        feedback_text: str,
        suggested_rule: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackEntry:
        embedding = embed_text(problem_text, dim=self.embedding_dim)
        entry = FeedbackEntry(
            trace_id=trace_id,
            problem_text=problem_text,
            teacher_score=int(teacher_score),
            feedback_text=feedback_text,
            suggested_rule=suggested_rule,
            embedding=embedding,
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


__all__ = ["FeedbackGalaxy", "FeedbackEntry"]
