"""Knowledgeverse Galaxy manager with resilience wrappers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .resilience import SelfHealingWrapper

try:  # pragma: no cover - optional GPU dependency
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _HAS_CUPY = False


def _env_true(name: str, default: str = "false") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


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
        self._galaxies: dict[str, Any] = {}
        self._knowledgeverse: Any | None = None
        # Sovereign enforcement: CPU O(n) scan is forbidden when PTX query is required.
        self.require_ptx_query = _env_true("K3D_REQUIRE_PTX_QUERY", "true")
        # Runtime cache for serialized entry text to reduce repeated O(entry_size)
        # json.dumps calls during benchmark query loops.
        self._entry_text_cache: dict[int, str] = {}
        # Runtime cache for hashed query vectors per entry id.
        self._entry_vector_cache: dict[int, Any] = {}
        # Cache specialist-filtered entry views per galaxy.
        self._specialist_entry_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Cache stacked vectors per specialist-filtered pool.
        self._specialist_vector_cache: dict[tuple[str, str], Any] = {}
        self._query_vector_dim = max(64, int(os.environ.get("K3D_QUERY_VECTOR_DIM", "512")))

    def set_knowledgeverse(self, knowledgeverse: Any) -> None:
        """Attach parent Knowledgeverse reference for specialized galaxy classes."""
        self._knowledgeverse = knowledgeverse

    @SelfHealingWrapper.with_retry(max_attempts=3, backoff_base=0.1)
    def query(
        self,
        query_text: str,
        specialist: str = "math",
        top_k: int = 10,
        galaxies: Sequence[str] | None = None,
        preferred_pattern_type: str | None = None,
    ) -> Any:
        return self._query_implementation(
            query_text,
            specialist=specialist,
            top_k=top_k,
            galaxies=galaxies,
            preferred_pattern_type=preferred_pattern_type,
        )

    def _query_implementation(
        self,
        query_text: str,
        specialist: str,
        top_k: int,
        galaxies: Sequence[str] | None = None,
        preferred_pattern_type: str | None = None,
    ) -> Any:
        if self.require_ptx_query:
            return self._query_ptx_implementation(
                query_text=query_text,
                specialist=specialist,
                top_k=top_k,
                galaxies=galaxies,
                preferred_pattern_type=preferred_pattern_type,
            )
        tokens = {tok for tok in re.split(r"[^a-z0-9_]+", query_text.lower()) if tok}
        specialist_key = str(specialist or "any").strip().lower()
        top_limit = max(1, int(top_k))
        target_names = self._resolve_target_galaxies(galaxies)
        scored: list[tuple[float, dict[str, Any], str]] = []
        for name in target_names:
            galaxy = self._galaxies.get(name)
            if galaxy is None:
                continue
            specialist_matches_galaxy = specialist_key in {"", "any"} or specialist_key in name.lower()
            entries = self._entries_for_specialist(
                galaxy_name=name,
                specialist_key=specialist_key,
                specialist_matches_galaxy=specialist_matches_galaxy,
            )
            for entry in entries:
                haystack = self._entry_haystack(entry)
                score = float(sum(1 for tok in tokens if tok in haystack))
                if preferred_pattern_type:
                    score += self._pattern_type_score_boost(entry, preferred_pattern_type)
                if score > 0:
                    scored.append((score, entry, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"galaxy": name, "score": float(score), "entry": entry}
            for score, entry, name in scored[:top_limit]
        ]

    def _query_ptx_implementation(
        self,
        *,
        query_text: str,
        specialist: str,
        top_k: int,
        galaxies: Sequence[str] | None,
        preferred_pattern_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if not _HAS_CUPY or cp is None:
            raise NotImplementedError(
                "PTX query kernel required but CuPy is unavailable. "
                "Install GPU runtime or run explicit non-sovereign diagnostics with "
                "K3D_REQUIRE_PTX_QUERY=false."
            )
        tokens = self._tokenize_query_tokens(query_text)
        if not tokens:
            return []
        specialist_key = str(specialist or "any").strip().lower()
        top_limit = max(1, int(top_k))
        target_names = self._resolve_target_galaxies(galaxies)
        query_vec = self._encode_tokens(tokens)
        query_gpu = cp.asarray(query_vec, dtype=cp.float32)

        scored: list[tuple[float, dict[str, Any], str]] = []
        for name in target_names:
            galaxy = self._galaxies.get(name)
            if galaxy is None:
                continue
            specialist_matches_galaxy = specialist_key in {"", "any"} or specialist_key in name.lower()
            entries = self._entries_for_specialist(
                galaxy_name=name,
                specialist_key=specialist_key,
                specialist_matches_galaxy=specialist_matches_galaxy,
            )
            if not entries:
                continue
            pool_key = (
                name,
                "__all__" if specialist_key in {"", "any"} or specialist_matches_galaxy else specialist_key,
            )
            matrix_np = self._vectors_for_pool(pool_key, entries)
            if matrix_np.shape[0] == 0:
                continue
            matrix_gpu = cp.asarray(matrix_np, dtype=cp.float32)
            scores_gpu = matrix_gpu.dot(query_gpu)
            local_k = min(top_limit, int(matrix_np.shape[0]))
            if local_k <= 0:
                continue
            if local_k < int(matrix_np.shape[0]):
                idx_gpu = cp.argpartition(scores_gpu, int(matrix_np.shape[0]) - local_k)[-local_k:]
            else:
                idx_gpu = cp.arange(int(matrix_np.shape[0]), dtype=cp.int32)
            local_idx = cp.asnumpy(idx_gpu).astype("int32", copy=False)
            local_scores = cp.asnumpy(scores_gpu[idx_gpu]).astype("float32", copy=False)
            for idx, score in zip(local_idx.tolist(), local_scores.tolist()):
                if score <= 0.0:
                    continue
                entry = entries[int(idx)]
                boosted = float(score)
                if preferred_pattern_type:
                    boosted += self._pattern_type_score_boost(entry, preferred_pattern_type)
                scored.append((boosted, entry, name))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"galaxy": name, "score": float(score), "entry": entry}
            for score, entry, name in scored[:top_limit]
        ]

    def _resolve_target_galaxies(self, galaxies: Sequence[str] | None) -> list[str]:
        if not galaxies:
            return list(self._galaxies.keys())
        wanted = [str(name).strip() for name in galaxies if str(name).strip()]
        if not wanted:
            return list(self._galaxies.keys())
        return [name for name in wanted if name in self._galaxies]

    def _tokenize_query_tokens(self, text: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-z0-9_]+", text.lower()) if tok}

    def _token_to_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) % self._query_vector_dim

    def _encode_tokens(self, tokens: set[str]) -> Any:
        vec = [0.0] * self._query_vector_dim
        for token in tokens:
            idx = self._token_to_index(token)
            vec[idx] += 1.0
        # NumPy-like dense vector is intentionally built on host once;
        # matching is offloaded to GPU.
        try:
            import numpy as np  # local import keeps module deps minimal for non-query paths

            return np.asarray(vec, dtype="float32")
        except Exception:
            return vec

    def _entry_vector(self, entry: dict[str, Any]) -> Any:
        cache_key = id(entry)
        cached = self._entry_vector_cache.get(cache_key)
        if cached is not None:
            return cached
        tokens = self._tokenize_query_tokens(self._entry_haystack(entry))
        vec = self._encode_tokens(tokens)
        self._entry_vector_cache[cache_key] = vec
        return vec

    def _vectors_for_pool(self, pool_key: tuple[str, str], entries: list[dict[str, Any]]) -> Any:
        cached = self._specialist_vector_cache.get(pool_key)
        if cached is not None and int(getattr(cached, "shape", [0])[0]) == len(entries):
            return cached
        if not entries:
            try:
                import numpy as np

                empty = np.empty((0, self._query_vector_dim), dtype="float32")
            except Exception:
                empty = []
            self._specialist_vector_cache[pool_key] = empty
            return empty
        try:
            import numpy as np

            matrix = np.vstack([self._entry_vector(entry) for entry in entries]).astype("float32", copy=False)
        except Exception:
            matrix = [self._entry_vector(entry) for entry in entries]
        self._specialist_vector_cache[pool_key] = matrix
        return matrix

    def _entry_haystack(self, entry: dict[str, Any]) -> str:
        cache_key = id(entry)
        cached = self._entry_text_cache.get(cache_key)
        if cached is not None:
            return cached
        text = json.dumps(entry, ensure_ascii=True).lower()
        self._entry_text_cache[cache_key] = text
        return text

    def _pattern_type_score_boost(self, entry: dict[str, Any], preferred_pattern_type: str) -> float:
        target = str(preferred_pattern_type or "").strip().lower()
        if not target:
            return 0.0
        entry_pattern = str(entry.get("pattern_type", "")).strip().lower()
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if not entry_pattern:
            entry_pattern = str(metadata.get("pattern_type", "")).strip().lower()
        source = str(metadata.get("source", "")).strip().lower()
        boost = 0.0
        if entry_pattern == target:
            boost += 0.30
        if source == "math_specialist_bootstrap":
            boost += 0.20
        return boost

    def _entries_for_specialist(
        self,
        *,
        galaxy_name: str,
        specialist_key: str,
        specialist_matches_galaxy: bool,
    ) -> list[dict[str, Any]]:
        galaxy = self._galaxies.get(galaxy_name)
        if galaxy is None:
            return []
        if specialist_key in {"", "any"} or specialist_matches_galaxy:
            return list(galaxy.entries)
        cache_key = (galaxy_name, specialist_key)
        cached = self._specialist_entry_cache.get(cache_key)
        if cached is not None:
            return cached
        filtered: list[dict[str, Any]] = []
        for entry in galaxy.entries:
            domain = str(entry.get("domain", "")).lower()
            category = str(entry.get("category", "")).lower()
            if specialist_key in domain or specialist_key in category:
                filtered.append(entry)
        self._specialist_entry_cache[cache_key] = filtered
        return filtered

    def get_galaxy(self, name: str) -> Any:
        galaxy = self._galaxies.get(name)
        if galaxy is not None:
            return galaxy

        if name == "Drawing":
            from .drawing_galaxy import DrawingGalaxy

            galaxy = DrawingGalaxy(knowledgeverse=self._knowledgeverse)
            self._hydrate_specialized_galaxy(name, galaxy)
            self._galaxies[name] = galaxy
            return galaxy

        if name == "Grammar":
            from .grammar_galaxy import GrammarGalaxy

            galaxy = GrammarGalaxy(knowledgeverse=self._knowledgeverse)
            self._hydrate_specialized_galaxy(name, galaxy)
            self._galaxies[name] = galaxy
            return galaxy

        entries = self._read_entries_from_disk(name)
        galaxy = Galaxy(name=name, entries=entries)
        self._galaxies[name] = galaxy
        return galaxy

    def add_entry(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        galaxy = self.get_galaxy(galaxy_name)
        if hasattr(galaxy, "add_entry"):
            galaxy.add_entry(entry, record_event=True)
        else:
            galaxy.entries.append(entry)
        # Entry list changed; clear cache to avoid stale pointers.
        self._entry_text_cache.clear()
        self._entry_vector_cache.clear()
        self._specialist_entry_cache.clear()
        self._specialist_vector_cache.clear()
        self._append_entry_to_disk(galaxy_name, entry)

    def _read_entries_from_disk(self, name: str) -> list[dict[str, Any]]:
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
        return entries

    def _hydrate_specialized_galaxy(self, galaxy_name: str, galaxy: Any) -> None:
        """Apply persisted entries to specialized galaxies on first load."""
        for entry in self._read_entries_from_disk(galaxy_name):
            try:
                galaxy.add_entry(entry, record_event=False)
            except Exception:
                continue

    def _galaxy_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self.storage_root / f"{safe}.jsonl"

    def _append_entry_to_disk(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        path = self._galaxy_path(galaxy_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
