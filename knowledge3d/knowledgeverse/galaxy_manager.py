"""Knowledgeverse Galaxy manager with resilience wrappers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .resilience import SelfHealingWrapper


@dataclass
class Galaxy:
    """Simple galaxy container with list-backed entries."""

    name: str
    entries: list[dict[str, Any]] = field(default_factory=list)


class GalaxyManager:
    """Galaxy manager with persistence and resilient query surface."""

    def __init__(self, storage_root: str | Path = "../Knowledge3D.local/galaxies"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._galaxies: dict[str, Galaxy] = {}

    @SelfHealingWrapper.with_retry(max_attempts=3, backoff_base=0.1)
    def query(self, query_text: str, specialist: str = "math", top_k: int = 10) -> Any:
        return self._query_implementation(query_text, specialist=specialist, top_k=top_k)

    def _query_implementation(self, query_text: str, specialist: str, top_k: int) -> Any:
        tokens = {tok.strip().lower() for tok in query_text.split() if tok.strip()}
        scored: list[tuple[int, dict[str, Any], str]] = []
        for name, galaxy in self._galaxies.items():
            for entry in galaxy.entries:
                haystack = json.dumps(entry, ensure_ascii=True).lower()
                score = sum(1 for tok in tokens if tok in haystack)
                if specialist and specialist != "any":
                    domain = str(entry.get("domain", "")).lower()
                    if specialist.lower() not in domain and specialist.lower() not in name.lower():
                        continue
                if score > 0:
                    scored.append((score, entry, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"galaxy": name, "score": score, "entry": entry}
            for score, entry, name in scored[: max(1, top_k)]
        ]

    def get_galaxy(self, name: str) -> Galaxy:
        galaxy = self._galaxies.get(name)
        if galaxy is not None:
            return galaxy

        path = self._galaxy_path(name)
        entries: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        galaxy = Galaxy(name=name, entries=entries)
        self._galaxies[name] = galaxy
        return galaxy

    def add_entry(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        galaxy = self.get_galaxy(galaxy_name)
        galaxy.entries.append(entry)
        self._append_entry_to_disk(galaxy_name, entry)

    def _galaxy_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self.storage_root / f"{safe}.jsonl"

    def _append_entry_to_disk(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        path = self._galaxy_path(galaxy_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
