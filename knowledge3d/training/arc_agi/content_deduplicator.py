"""
Content-based deduplication for RPN programs.

Keeps a canonical set of programs keyed by content hash, and tracks usage
metadata (scores + contexts) for each hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ContentDeduplicator:
    """Deduplicate RPN programs using content hashing."""

    def __init__(self) -> None:
        self.canonical_programs: Dict[str, Dict] = {}
        self.usage_metadata: Dict[str, List[Dict]] = {}

    @staticmethod
    def compute_hash(program: str) -> str:
        """Compute SHA256 hash of normalized RPN program."""
        normalized = " ".join(program.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def add_or_reference(
        self,
        *,
        program: str,
        program_type: str,
        score: float,
        context: Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        Add a program to the canonical set or reference an existing one.

        Returns:
            (canonical_hash, is_new)
        """
        prog_hash = self.compute_hash(program)

        if prog_hash in self.canonical_programs:
            self.usage_metadata.setdefault(prog_hash, []).append({"score": score, "context": context or {}})
            return prog_hash, False

        self.canonical_programs[prog_hash] = {
            "hash": prog_hash,
            "program": program,
            "type": program_type,
            "first_seen_score": score,
            "usage_count": 1,
        }
        self.usage_metadata[prog_hash] = [{"score": score, "context": context or {}}]
        return prog_hash, True

    def get_usage_stats(self, prog_hash: str) -> Dict:
        """Return usage statistics for a program hash."""
        records = self.usage_metadata.get(prog_hash, [])
        if not records:
            return {"usage_count": 0, "avg_score": 0.0, "max_score": 0.0, "min_score": 0.0}

        scores = [r["score"] for r in records]
        return {
            "usage_count": len(records),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }

    def prune_low_quality(self, min_usage: int = 2, min_score: float = 0.65) -> int:
        """
        Remove low-usage / low-score programs.

        Returns:
            Number of programs removed.
        """
        to_remove: List[str] = []
        for prog_hash in list(self.canonical_programs.keys()):
            stats = self.get_usage_stats(prog_hash)
            if stats["usage_count"] < min_usage and stats["max_score"] < 0.8:
                to_remove.append(prog_hash)
            elif stats["avg_score"] < min_score:
                to_remove.append(prog_hash)

        for prog_hash in to_remove:
            self.canonical_programs.pop(prog_hash, None)
            self.usage_metadata.pop(prog_hash, None)

        return len(to_remove)

    def save(self, path: Path) -> None:
        """Persist deduplication index to disk."""
        state = {
            "canonical_programs": self.canonical_programs,
            "usage_metadata": self.usage_metadata,
            "total_unique": len(self.canonical_programs),
            "total_references": sum(len(v) for v in self.usage_metadata.values()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"[ContentDeduplicator] Saved {len(self.canonical_programs)} unique programs to {path}")

    def load(self, path: Path) -> None:
        """Load deduplication index if it exists."""
        if not path.exists():
            print(f"[ContentDeduplicator] No checkpoint at {path}, starting fresh")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self.canonical_programs = state.get("canonical_programs", {})
        self.usage_metadata = state.get("usage_metadata", {})
        total_refs = state.get("total_references", sum(len(v) for v in self.usage_metadata.values()))
        print(f"[ContentDeduplicator] Loaded {len(self.canonical_programs)} unique programs ({total_refs} references)")


__all__ = ["ContentDeduplicator"]
