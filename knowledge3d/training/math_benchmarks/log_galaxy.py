"""
Log Galaxy: structured execution traces for navigation supervision.

Phase 2.1 stores traces as JSONL with a VRAM-friendly layout in mind.
Planned CUDA layout (conceptual):
  - trace_offsets[N+1] -> prefix sums into flat step buffers
  - step_rule_ids[total_steps] -> uint16 rule codes (sum/product/etc.)
  - step_kind[total_steps] -> uint8 enum (decompose/base/result)
  - trace_result[N] -> float32
  - trace_success[N] -> uint8
  - problem_embeddings[N][D] -> float16/float32

Variable-length traces become contiguous ranges in the flat buffers.
This enables O(1) access per trace with two reads (offsets) and coalesced
step reads on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


def _stable_id(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class LogGalaxyEntry:
    trace_id: str
    problem_text: str
    problem_embedding: List[float]
    step_sequence: List[Dict[str, Any]]
    result: Optional[float]
    success: bool
    trace_lines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "problem_text": self.problem_text,
            "problem_embedding": self.problem_embedding,
            "step_sequence": self.step_sequence,
            "result": self.result,
            "success": self.success,
            "trace_lines": self.trace_lines,
            "metadata": self.metadata,
        }


class LogGalaxy:
    """
    In-memory Log Galaxy container with JSONL export.

    Embeddings are generated with the router embedder to keep locality.
    """

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[LogGalaxyEntry] = []

    def add_trace(
        self,
        *,
        problem_text: str,
        step_sequence: List[Dict[str, Any]],
        result: Optional[float],
        success: bool,
        trace_lines: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LogGalaxyEntry:
        embed = embed_text(problem_text, dim=self.embedding_dim)
        trace_id = _stable_id(problem_text + json.dumps(step_sequence, sort_keys=True))
        entry = LogGalaxyEntry(
            trace_id=trace_id,
            problem_text=problem_text,
            problem_embedding=embed,
            step_sequence=list(step_sequence),
            result=result,
            success=bool(success),
            trace_lines=list(trace_lines or []),
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


__all__ = ["LogGalaxy", "LogGalaxyEntry"]
