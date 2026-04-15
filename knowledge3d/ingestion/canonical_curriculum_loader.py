"""Load curriculum meaning stars from k3d_canonical into a live Knowledgeverse."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from qdrant_client import models

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from scripts.ingest_meaning_layer import target_galaxy_for_star


def _domain_token_set(star: MeaningCentricStar, *, subkind: str = "") -> set[str]:
    values = [
        str(star.domain or "").strip().lower(),
        str(star.galaxy_ref or "").strip().lower(),
        str(subkind or "").strip().lower(),
        str(star.meaning_class or "").strip().lower(),
        str(star.star_id or "").strip().lower(),
    ]
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        tokens.add(value)
        for token in value.replace("-", "_").split("_"):
            if token:
                tokens.add(token)
    return tokens


def _target_galaxy(star: MeaningCentricStar) -> str:
    return _target_galaxy_with_subkind(star)


def _target_galaxy_with_subkind(star: MeaningCentricStar, *, subkind: str = "") -> str:
    tokens = _domain_token_set(star, subkind=subkind)
    if tokens & {
        "reality",
        "history",
        "geography",
        "earth",
        "environment",
        "environmental",
        "natural",
        "natural_science",
        "science",
        "physics",
        "chemistry",
        "biology",
        "astronomy",
        "space",
        "civics",
        "economics",
        "macroeconomics",
        "microeconomics",
        "government",
        "politics",
        "social_studies",
        "health",
        "medicine",
        "clinical",
        "psychology",
        "psychiatry",
        "sociology",
        "anthropology",
    }:
        return "Reality"
    if tokens & {
        "tools",
        "tool",
        "computer",
        "cyber",
        "media",
        "research",
        "arc",
        "arc_agi",
        "pattern",
        "transform",
        "visual_reasoning",
        "grid",
        "applied",
        "computer_science",
    }:
        return "Tool"
    if tokens & {
        "language",
        "linguistic",
        "linguistics",
        "humanities",
        "crosscultural",
        "cross_cultural",
        "cross-cultural",
        "literature",
        "philosophy",
        "religion",
        "theology",
        "ethics",
        "aesthetics",
        "arts",
        "music",
        "music_theory",
        "culture",
        "calendar",
        "proverb",
    }:
        return "Language"
    if tokens & {"drawing", "visual", "shape", "glyph"}:
        return "Drawing"
    if tokens & {"math", "mathematics"}:
        return "Math"
    return target_galaxy_for_star(star)


def _iter_meaning_star_payloads(lookup: CanonicalLookup) -> list[dict[str, Any]]:
    lookup.ensure_collection()
    offset = None
    payloads: list[dict[str, Any]] = []
    while True:
        points, offset = lookup.client.scroll(
            collection_name=lookup.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="kind", match=models.MatchValue(value="meaning_star"))]
            ),
        )
        if not points:
            break
        payloads.extend(dict(point.payload or {}) for point in points)
        if offset is None:
            break
    return payloads


def _loaded_curriculum_star_ids(knowledgeverse: Any) -> set[str]:
    manager = knowledgeverse.galaxy_manager
    star_ids: set[str] = set()
    for galaxy_name in knowledgeverse._discover_live_galaxy_names():
        galaxy = manager.get_galaxy(galaxy_name)
        entries = list(getattr(galaxy, "entries", []) or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = dict(entry.get("metadata") or {})
            if str(metadata.get("ingest_source") or "") != "canonical_curriculum":
                continue
            star_id = str(metadata.get("meaning_star_id") or entry.get("id") or "").strip()
            if star_id:
                star_ids.add(star_id)
    return star_ids


def _loaded_curriculum_counts_by_galaxy(knowledgeverse: Any) -> dict[str, int]:
    manager = knowledgeverse.galaxy_manager
    counts: Counter[str] = Counter()
    for galaxy_name in knowledgeverse._discover_live_galaxy_names():
        galaxy = manager.get_galaxy(galaxy_name)
        entries = list(getattr(galaxy, "entries", []) or [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = dict(entry.get("metadata") or {})
            if str(metadata.get("ingest_source") or "") != "canonical_curriculum":
                continue
            counts[str(galaxy_name)] += 1
    return dict(sorted(counts.items()))


def assert_canonical_curriculum_loaded(
    knowledgeverse: Any,
    *,
    payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lookup = CanonicalLookup()
    rows = list(payloads or _iter_meaning_star_payloads(lookup))
    expected_by_galaxy: Counter[str] = Counter()
    expected_ids: set[str] = set()
    for payload in rows:
        metadata = dict(payload.get("metadata") or {})
        meaning_payload = dict(metadata.get("meaning_star") or {})
        if not meaning_payload:
            continue
        star = MeaningCentricStar.from_dict(meaning_payload)
        subkind = str(metadata.get("subkind") or "").strip()
        expected_ids.add(star.star_id)
        expected_by_galaxy[_target_galaxy_with_subkind(star, subkind=subkind)] += 1
    loaded_ids = _loaded_curriculum_star_ids(knowledgeverse)
    missing_ids = sorted(expected_ids - loaded_ids)
    loaded_by_galaxy = _loaded_curriculum_counts_by_galaxy(knowledgeverse)
    return {
        "status": "ok" if not missing_ids else "error",
        "expected_total": len(expected_ids),
        "loaded_total": len(loaded_ids),
        "missing_ids": missing_ids,
        "expected_by_galaxy": dict(sorted(expected_by_galaxy.items())),
        "loaded_by_galaxy": loaded_by_galaxy,
    }


def load_canonical_curriculum_into_knowledgeverse(
    knowledgeverse: Any,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    emit = progress or (lambda _message: None)
    manager = knowledgeverse.galaxy_manager
    knowledgeverse.ensure_default_galaxies_loaded()
    lookup = CanonicalLookup()
    payloads = _iter_meaning_star_payloads(lookup)
    inserted = 0
    updated = 0
    by_galaxy: Counter[str] = Counter()
    by_subkind: Counter[str] = Counter()
    for payload in payloads:
        metadata = dict(payload.get("metadata") or {})
        meaning_payload = dict(metadata.get("meaning_star") or {})
        if not meaning_payload:
            continue
        star = MeaningCentricStar.from_dict(meaning_payload)
        subkind = str(metadata.get("subkind") or "").strip()
        galaxy_name = _target_galaxy_with_subkind(star, subkind=subkind)
        entry = star.to_galaxy_entry(
            entry_id=star.star_id,
            galaxy_name=galaxy_name,
            category="meaning_star",
            metadata={
                "ingest_source": "canonical_curriculum",
                "subkind": metadata.get("subkind"),
                "source_file": metadata.get("source_file"),
                "source_line": metadata.get("source_line"),
                "rpn_sketch": metadata.get("rpn_sketch"),
                "symlink_refs": list(metadata.get("symlink_refs") or []),
                "surface_forms_raw": dict(metadata.get("surface_forms_raw") or {}),
                "curriculum_metadata": {
                    "subkind": metadata.get("subkind"),
                    "source_file": metadata.get("source_file"),
                    "source_line": metadata.get("source_line"),
                    "rpn_sketch": metadata.get("rpn_sketch"),
                    "symlink_refs": list(metadata.get("symlink_refs") or []),
                    "surface_forms_raw": dict(metadata.get("surface_forms_raw") or {}),
                    "is_a": list(metadata.get("is_a") or []),
                    "saudades": metadata.get("saudades"),
                    "domain": metadata.get("domain"),
                },
            },
        )
        status = manager.upsert_entry(galaxy_name, entry)
        if status == "inserted":
            inserted += 1
        else:
            updated += 1
        by_galaxy[galaxy_name] += 1
        subkind = str(metadata.get("subkind") or "").strip()
        if subkind:
            by_subkind[subkind] += 1
    emit(
        "Canonical curriculum load: "
        f"inserted={inserted} updated={updated} "
        f"galaxies={dict(sorted(by_galaxy.items()))}"
    )
    return {
        "payload_count": len(payloads),
        "inserted": inserted,
        "updated": updated,
        "by_galaxy": dict(sorted(by_galaxy.items())),
        "by_subkind": dict(sorted(by_subkind.items())),
    }


__all__ = [
    "assert_canonical_curriculum_loaded",
    "load_canonical_curriculum_into_knowledgeverse",
]
