"""Helpers for resolving the preferred local runtime storage root."""

from __future__ import annotations

import os
from pathlib import Path


PRIMARY_LOCAL_STORAGE_ROOT = Path("/K3D/Knowledge3D.local")


def repo_sibling_storage_root() -> Path:
    return Path(__file__).resolve().parents[1].parent / "Knowledge3D.local"


def iter_storage_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("K3D_STORAGE_ROOT", "K3D_LOCAL_DIR"):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.append(PRIMARY_LOCAL_STORAGE_ROOT)
    candidates.append(repo_sibling_storage_root())

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def default_storage_root() -> Path:
    for candidate in iter_storage_root_candidates():
        if candidate.exists():
            return candidate
    return PRIMARY_LOCAL_STORAGE_ROOT


def resolve_storage_root(value: str | Path | None = None) -> Path:
    if value is not None:
        raw = str(value).strip()
        if raw:
            return Path(raw).expanduser()
    return default_storage_root()
