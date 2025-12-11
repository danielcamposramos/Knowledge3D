from __future__ import annotations

import hashlib
from typing import List, Dict, Any


class TernaryWorkingMemory:
    """Bounded working memory with deduplication and utilization tracking."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._entries: List[Dict[str, Any]] = []

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, len(self._entries) / float(self.capacity))

    def _hash_entry(self, entry: Dict[str, Any]) -> str:
        # Deterministic hash to deduplicate discoveries.
        blob = repr(sorted(entry.items())).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    def add(self, entry: Dict[str, Any]) -> None:
        """Append discovery into working memory with capacity bound."""
        entry = dict(entry)
        entry["_hash"] = self._hash_entry(entry)
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity :]

    def deduplicate(self) -> List[Dict[str, Any]]:
        """Return unique entries by hash."""
        seen = set()
        unique: List[Dict[str, Any]] = []
        for item in self._entries:
            key = item.get("_hash") or self._hash_entry(item)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def clear(self) -> None:
        self._entries.clear()

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)


__all__ = ["TernaryWorkingMemory"]
