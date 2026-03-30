"""Knowledgeverse Galaxy manager with resilience wrappers."""

from __future__ import annotations

import ast
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .meaning_star import MeaningCentricStar
from .resilience import SelfHealingWrapper


def _env_true(name: str, default: str = "false") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _decode_legacy_literal(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, tuple, int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped[:1] in {"{", "[", "("}:
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value
    return value


def _coerce_legacy_refs(value: Any) -> list[str]:
    decoded = _decode_legacy_literal(value)
    if decoded is None:
        return []
    if isinstance(decoded, tuple):
        decoded = list(decoded)
    if not isinstance(decoded, list):
        return []
    return [str(item).strip() for item in decoded if str(item).strip()]


def _coerce_legacy_position(value: Any) -> list[float]:
    decoded = _decode_legacy_literal(value)
    if isinstance(decoded, tuple):
        decoded = list(decoded)
    if not isinstance(decoded, list):
        decoded = [0.0, 0.0, 0.0]
    resolved: list[float] = []
    for item in decoded[:3]:
        try:
            resolved.append(float(item))
        except Exception:
            resolved.append(0.0)
    while len(resolved) < 3:
        resolved.append(0.0)
    return resolved


def _coerce_legacy_optional_text(value: Any) -> str | None:
    decoded = _decode_legacy_literal(value)
    if decoded is None:
        return None
    text = str(decoded).strip()
    return text or None


def _coerce_legacy_surface_forms(value: Any) -> dict[str, Any]:
    decoded = _decode_legacy_literal(value)
    if not isinstance(decoded, dict):
        return {}
    return {
        str(language).strip().lower(): raw_value
        for language, raw_value in decoded.items()
        if str(language).strip()
    }


def normalize_meaning_layer_entry(
    entry: dict[str, Any],
    *,
    galaxy_name: str = "meaning_layer_stars",
) -> dict[str, Any]:
    raw = dict(entry)
    star_payload = {
        "star_id": str(raw.get("star_id") or raw.get("id") or "").strip(),
        "meaning_class": str(raw.get("meaning_class") or raw.get("category") or "meaning_star").strip(),
        "meaning_rpn": str(raw.get("meaning_rpn") or raw.get("rpn_program") or raw.get("content") or "").strip(),
        "domain": str(raw.get("domain") or raw.get("galaxy") or galaxy_name).strip(),
        "taxonomy_refs": _coerce_legacy_refs(raw.get("taxonomy_refs")),
        "surface_forms": _coerce_legacy_surface_forms(raw.get("surface_forms")),
        "visual_rpn": _coerce_legacy_optional_text(raw.get("visual_rpn")),
        "visual_refs": _coerce_legacy_refs(raw.get("visual_refs")),
        "audio_rpn": _coerce_legacy_optional_text(raw.get("audio_rpn")),
        "audio_refs": _coerce_legacy_refs(raw.get("audio_refs")),
        "pronunciations": _decode_legacy_literal(raw.get("pronunciations")) or {},
        "behavior_rpn": _coerce_legacy_optional_text(raw.get("behavior_rpn")),
        "reality_refs": _coerce_legacy_refs(raw.get("reality_refs")),
        "grammar_refs": _coerce_legacy_refs(raw.get("grammar_refs")),
        "meta_refs": _coerce_legacy_refs(raw.get("meta_refs")),
        "house_position": _coerce_legacy_position(raw.get("house_position")),
        "house_room": str(raw.get("house_room") or "").strip(),
        "galaxy_ref": str(raw.get("galaxy_ref") or galaxy_name).strip(),
        "confidence": raw.get("confidence"),
        "polarity": raw.get("polarity"),
        "component_refs": _coerce_legacy_refs(raw.get("component_refs")),
        "composite_of": _coerce_legacy_refs(raw.get("composite_of")),
    }
    star = MeaningCentricStar.from_dict(star_payload)
    name = star._preferred_name()
    rpn_text = str(star.meaning_rpn or "").strip()
    metadata = {
        "meaning_star": star.to_dict(),
        "meaning_star_id": star.star_id,
        "meaning_class": star.meaning_class,
        "house_room": star.house_room,
        "surface_form_languages": sorted(star.surface_forms.keys()),
        "surface_forms": {
            language: surface_form.to_dict()
            for language, surface_form in star.surface_forms.items()
        },
        "house_position": [float(value) for value in star.house_position],
        "confidence": int(star.confidence),
        "polarity": int(star.polarity),
        "taxonomy_refs": list(star.taxonomy_refs),
        "component_refs": list(star.component_refs),
        "grammar_refs": list(star.grammar_refs),
        "reality_refs": list(star.reality_refs),
    }
    if star.behavior_rpn:
        metadata["behavior_rpn"] = str(star.behavior_rpn)
    if star.visual_rpn:
        metadata["visual_rpn"] = str(star.visual_rpn)
    if star.audio_rpn:
        metadata["audio_rpn"] = str(star.audio_rpn)
    if star.domain:
        metadata["domain"] = str(star.domain)
    if star.galaxy_ref:
        metadata["galaxy_ref"] = str(star.galaxy_ref)
    return {
        "id": star.star_id,
        "name": name,
        "galaxy": str(galaxy_name),
        "domain": str(star.domain or galaxy_name).strip().lower(),
        "category": str(star.meaning_class or "meaning_star").strip().lower(),
        "layer": 2,
        "content": rpn_text or name,
        "summary": name or rpn_text,
        "description": rpn_text or name,
        "rpn_program": rpn_text,
        "metadata": metadata,
    }


def normalize_disk_entry(
    galaxy_name: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    raw = dict(entry)
    if "star_id" in raw and "id" not in raw:
        return normalize_meaning_layer_entry(raw, galaxy_name=galaxy_name)
    return raw


@dataclass
class Galaxy:
    """Simple galaxy container with list-backed entries."""

    name: str
    entries: list[dict[str, Any]] = field(default_factory=list)


class GalaxyManager:
    """Galaxy manager with persistence and resilient query surface."""

    def __init__(
        self,
        storage_root: str | Path = "../Knowledge3D.local/galaxies",
        *,
        extra_storage_roots: Sequence[str | Path] | None = None,
    ):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.extra_storage_roots = [Path(path) for path in list(extra_storage_roots or [])]
        for root in self.extra_storage_roots:
            root.mkdir(parents=True, exist_ok=True)
        self._galaxies: dict[str, Any] = {}
        self._knowledgeverse: Any | None = None
        # Sovereign query is currently token/rule matching only. Any future
        # embedding query must arrive as a real PTX kernel, not a wrapper library.
        self.require_ptx_query = _env_true("K3D_REQUIRE_PTX_QUERY", "true")
        # Runtime cache for serialized entry text to reduce repeated O(entry_size)
        # json.dumps calls during benchmark query loops.
        self._entry_text_cache: dict[int, str] = {}
        # Cache specialist-filtered entry views per galaxy.
        self._specialist_entry_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Batch persistence during foundational bootstrap to avoid rewriting
        # the full galaxy file once per upsert.
        self._disk_sync_depth = 0
        self._dirty_galaxies: set[str] = set()

    def _iter_storage_roots(self) -> list[Path]:
        ordered = [self.storage_root, *self.extra_storage_roots]
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in ordered:
            resolved = root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(root)
        return unique

    def iter_storage_jsonl_paths(self) -> list[Path]:
        paths: list[Path] = []
        for root in self._iter_storage_roots():
            paths.extend(sorted(root.glob("*.jsonl")))
        return paths

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
        return self._query_token_implementation(
            query_text=query_text,
            specialist=specialist,
            top_k=top_k,
            galaxies=galaxies,
            preferred_pattern_type=preferred_pattern_type,
        )

    def _query_token_implementation(
        self,
        *,
        query_text: str,
        specialist: str,
        top_k: int,
        galaxies: Sequence[str] | None = None,
        preferred_pattern_type: str | None = None,
    ) -> list[dict[str, Any]]:
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
                score += self._math_core_score_boost(entry, tokens)
                if score > 0:
                    scored.append((score, entry, name))
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
        resolved: list[str] = []
        for name in wanted:
            try:
                self.get_galaxy(name)
            except Exception:
                continue
            if name in self._galaxies:
                resolved.append(name)
        return resolved

    def _tokenize_query_tokens(self, text: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-z0-9_]+", text.lower()) if tok}

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
        boost = 0.0
        if entry_pattern == target:
            boost += 0.30
        return boost

    def _math_core_score_boost(self, entry: dict[str, Any], query_tokens: set[str]) -> float:
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        math_core = metadata.get("math_core") if isinstance(metadata.get("math_core"), dict) else {}
        preferred_tier = math_core.get("preferred_tier")
        tool_kind = str(metadata.get("tool_kind", "")).strip().lower()
        if preferred_tier is None and tool_kind != "math_core_allocator":
            return 0.0

        tokens = {str(token).strip().lower() for token in query_tokens if str(token).strip()}
        boost = 0.0
        if preferred_tier is not None:
            tier = int(preferred_tier)
            tier1_tokens = {
                "scalar",
                "ternary",
                "cheap",
                "fast",
                "gradient",
                "palette",
                "worker_worker",
                "tier1",
            }
            tier2_tokens = {
                "grid",
                "mesh",
                "signal",
                "video",
                "audio",
                "spectrogram",
                "geometry",
                "profile",
                "worker",
                "tier2",
            }
            tier3_tokens = {
                "matrix",
                "tensor",
                "trm",
                "master",
                "high",
                "tier3",
                "complex",
            }
            if tier == 1 and not tier1_tokens.isdisjoint(tokens):
                boost += 0.16
            if tier == 2 and not tier2_tokens.isdisjoint(tokens):
                boost += 0.18
            if tier == 3 and not tier3_tokens.isdisjoint(tokens):
                boost += 0.22
        if tool_kind == "math_core_allocator" and {"pool", "spawn", "cascade", "allocate"} & tokens:
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
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            modality_tokens = self._entry_specialist_tokens(entry, metadata)
            if (
                specialist_key in domain
                or specialist_key in category
                or specialist_key in modality_tokens
                or not self._specialist_aliases(specialist_key).isdisjoint(modality_tokens)
            ):
                filtered.append(entry)
        self._specialist_entry_cache[cache_key] = filtered
        return filtered

    def _specialist_aliases(self, specialist_key: str) -> set[str]:
        key = str(specialist_key or "").strip().lower()
        aliases = {
            "visual": {"visual", "drawing", "image", "texture", "signal"},
            "audio": {"audio", "sound", "signal", "spectrogram"},
            "physics": {"physics", "reality", "3dobjects", "3d", "spatial"},
            "cartographer": {"drawing", "reality", "3dobjects", "spatial", "visual"},
            "math": {"math"},
            "grammar": {"grammar", "language"},
        }
        return aliases.get(key, {key})

    def _entry_specialist_tokens(self, entry: dict[str, Any], metadata: dict[str, Any]) -> set[str]:
        raw_values: list[str] = []
        for key in ("tool_kind", "cross_modal", "modalities"):
            value = metadata.get(key)
            if isinstance(value, str):
                raw_values.append(value)
            elif isinstance(value, list):
                raw_values.extend(str(item) for item in value)
        for key in ("pattern_type", "type", "domain", "category"):
            value = entry.get(key)
            if isinstance(value, str):
                raw_values.append(value)

        tokens: set[str] = set()
        for value in raw_values:
            for token in re.split(r"[^a-z0-9_]+", value.lower()):
                if token:
                    tokens.add(token)
        return tokens

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

    def _coerce_entry(self, galaxy_name: str, entry: dict[str, Any] | MeaningCentricStar) -> dict[str, Any]:
        if isinstance(entry, MeaningCentricStar):
            return entry.to_galaxy_entry(galaxy_name=galaxy_name)
        return dict(entry)

    def store_meaning_star(
        self,
        galaxy_name: str,
        star: MeaningCentricStar,
        *,
        entry_id: str | None = None,
        name: str | None = None,
        category: str = "meaning_star",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry = star.to_galaxy_entry(
            entry_id=entry_id or star.star_id,
            name=name,
            galaxy_name=galaxy_name,
            category=category,
            metadata=metadata,
        )
        return self.upsert_entry(galaxy_name, entry)

    def load_meaning_star(self, galaxy_name: str, star_id: str) -> MeaningCentricStar | None:
        target = str(star_id).strip()
        if not target:
            return None
        galaxy = self.get_galaxy(galaxy_name)
        entries = list(getattr(galaxy, "entries", getattr(galaxy, "_extra_entries", [])) or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            if str(entry.get("id", "")).strip() == target or str(metadata.get("meaning_star_id", "")).strip() == target:
                return MeaningCentricStar.from_galaxy_entry(entry)
        return None

    def load_galaxy_on_demand(self, star: MeaningCentricStar) -> Any | None:
        """Load the galaxy referenced by a container object, if any."""
        if not isinstance(star, MeaningCentricStar):
            return None
        galaxy_ref = str(star.galaxy_ref).strip()
        if not galaxy_ref:
            return None
        return self.get_galaxy(galaxy_ref)

    def add_entry(self, galaxy_name: str, entry: dict[str, Any] | MeaningCentricStar) -> None:
        entry_dict = self._coerce_entry(galaxy_name, entry)
        galaxy = self.get_galaxy(galaxy_name)
        if hasattr(galaxy, "add_entry"):
            galaxy.add_entry(entry_dict, record_event=True)
        else:
            galaxy.entries.append(entry_dict)
        # Entry list changed; clear cache to avoid stale pointers.
        self._entry_text_cache.clear()
        self._specialist_entry_cache.clear()
        if self._knowledgeverse is not None and hasattr(self._knowledgeverse, "invalidate_gpu_galaxy_binding"):
            try:
                self._knowledgeverse.invalidate_gpu_galaxy_binding()
            except Exception:
                pass
        if self._disk_sync_depth > 0:
            self._dirty_galaxies.add(galaxy_name)
        else:
            self._append_entry_to_disk(galaxy_name, entry_dict)

    def upsert_entry(self, galaxy_name: str, entry: dict[str, Any] | MeaningCentricStar) -> str:
        entry_dict = self._coerce_entry(galaxy_name, entry)
        galaxy = self.get_galaxy(galaxy_name)
        entry_id = self._entry_identifier(entry_dict)
        if not entry_id:
            self.add_entry(galaxy_name, entry_dict)
            return "inserted"

        replaced = self._replace_entry_in_memory(galaxy, entry_id, entry_dict)
        if not replaced:
            self.add_entry(galaxy_name, entry_dict)
            return "inserted"

        self._entry_text_cache.clear()
        self._specialist_entry_cache.clear()
        if self._knowledgeverse is not None and hasattr(self._knowledgeverse, "invalidate_gpu_galaxy_binding"):
            try:
                self._knowledgeverse.invalidate_gpu_galaxy_binding()
            except Exception:
                pass
        if self._disk_sync_depth > 0:
            self._dirty_galaxies.add(galaxy_name)
        else:
            self._rewrite_galaxy_disk(galaxy_name, galaxy)
        return "updated"

    @contextmanager
    def bulk_disk_sync(self):
        self._disk_sync_depth += 1
        try:
            yield
        finally:
            self._disk_sync_depth = max(0, self._disk_sync_depth - 1)
            if self._disk_sync_depth != 0 or not self._dirty_galaxies:
                return
            dirty_galaxies = sorted(self._dirty_galaxies)
            self._dirty_galaxies.clear()
            for galaxy_name in dirty_galaxies:
                galaxy = self.get_galaxy(galaxy_name)
                self._rewrite_galaxy_disk(galaxy_name, galaxy)

    def _read_entries_from_disk(self, name: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for path in self._candidate_galaxy_paths(name):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                normalized = normalize_disk_entry(name, entry)
                entry_id = self._entry_identifier(normalized)
                if entry_id and entry_id in seen_ids:
                    continue
                entries.append(normalized)
                if entry_id:
                    seen_ids.add(entry_id)
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

    def _candidate_galaxy_paths(self, name: str) -> list[Path]:
        safe = name.replace("/", "_").replace(" ", "_")
        return [root / f"{safe}.jsonl" for root in self._iter_storage_roots()]

    def _append_entry_to_disk(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        path = self._galaxy_path(galaxy_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")

    @staticmethod
    def _entry_identifier(entry: dict[str, Any]) -> str:
        return str(entry.get("id") or entry.get("rule_id") or "").strip()

    def _replace_entry_in_memory(self, galaxy: Any, entry_id: str, entry: dict[str, Any]) -> bool:
        if isinstance(galaxy, Galaxy):
            for idx, current in enumerate(galaxy.entries):
                if not isinstance(current, dict):
                    continue
                if self._entry_identifier(current) == entry_id:
                    galaxy.entries[idx] = dict(entry)
                    return True
            return False

        extra_entries = getattr(galaxy, "_extra_entries", None)
        if isinstance(extra_entries, list):
            for idx, current in enumerate(extra_entries):
                if not isinstance(current, dict):
                    continue
                if self._entry_identifier(current) == entry_id:
                    extra_entries[idx] = dict(entry)
                    return True
        return False

    def _rewrite_galaxy_disk(self, galaxy_name: str, galaxy: Any) -> None:
        path = self._galaxy_path(galaxy_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(galaxy, Galaxy):
            entries = list(galaxy.entries)
        else:
            extra_entries = getattr(galaxy, "_extra_entries", None)
            entries = list(extra_entries) if isinstance(extra_entries, list) else []
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n")
