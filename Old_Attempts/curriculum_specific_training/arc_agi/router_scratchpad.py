"""
RouterScratchpad - lightweight temporary storage for multi-step reasoning.

SOVEREIGN: Stored as plain Python dict with symlink-friendly keys.
Intended to be mirrored into Galaxy (via semantic_context) if needed.
"""

from __future__ import annotations

from typing import Any, Dict


class RouterScratchpad:
    """Temporary scratch storage for intermediate results."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def clear(self) -> None:
        self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._store)


__all__ = ["RouterScratchpad"]
