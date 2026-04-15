"""File-backed Qdrant credential resolution for local ingestion tools."""

from __future__ import annotations

from pathlib import Path
import os


QDRANT_API_KEY_FILE = Path("/K3D/Knowledge3D.local/secrets/qdrant_api_key.txt")


def resolve_qdrant_api_key() -> str:
    api_key = str(os.getenv("QDRANT_API_KEY", "")).strip()
    if api_key:
        return api_key
    if QDRANT_API_KEY_FILE.exists():
        value = QDRANT_API_KEY_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise RuntimeError(
        f"QDRANT_API_KEY not set and {QDRANT_API_KEY_FILE} not found"
    )


__all__ = ["QDRANT_API_KEY_FILE", "resolve_qdrant_api_key"]
