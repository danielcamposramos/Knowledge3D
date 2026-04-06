"""Load disk-backed Galaxy knowledge into VRAM-ready star records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
from knowledge3d.knowledgeverse.sovereign_text_embedder import embed_text_sovereign


GALAXY_JSONL_DIR = Path("/K3D/Knowledge3D.local/galaxies")
HOUSE_JSONL_DIR = Path("/K3D/Knowledge3D.local/house")
GALAXY_NAME_TO_TYPE = {
    "drawing": 1,
    "3dobjects": 1,
    "character": 2,
    "word": 2,
    "language": 2,
    "grammar": 3,
    "reality": 4,
    "tool": 4,
    "audio": 4,
    "book_": 4,
    "math": 5,
    "number": 5,
    "meaning_layer_stars": 7,
}


def _iter_disk_jsonl_paths(search_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if (search_dir / "galaxies").exists():
        roots.append(search_dir / "galaxies")
        roots.append(search_dir / "house")
    else:
        roots.append(search_dir)
        if search_dir.name.lower() == "galaxies":
            roots.append(search_dir.parent / "house")
    ordered: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen or not root.exists():
            continue
        seen.add(resolved)
        ordered.append(root)
    paths: list[Path] = []
    for root in ordered:
        paths.extend(sorted(root.glob("*.jsonl")))
    return paths


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _galaxy_type_from_filename(filename: str) -> int:
    name = str(filename or "").strip().lower().removesuffix(".jsonl")
    for key, star_type in GALAXY_NAME_TO_TYPE.items():
        if name.startswith(key) or key in name:
            return int(star_type)
    return 6


def _coerce_embedding(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    embedding = []
    for value in list(values)[:32]:
        try:
            embedding.append(float(value))
        except Exception:
            embedding.append(0.0)
    if len(embedding) < 32:
        embedding.extend([0.0] * (32 - len(embedding)))
    return embedding[:32]


def _star_source(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        meaning_star = metadata.get("meaning_star")
        if isinstance(meaning_star, dict):
            return meaning_star
    return entry


def _surface_form_text(surface_forms: Any) -> str:
    if not isinstance(surface_forms, dict):
        return ""
    parts: list[str] = []
    for payload in surface_forms.values():
        if isinstance(payload, dict):
            word_ref = payload.get("word_ref")
            if word_ref:
                parts.append(str(word_ref))
    return " ".join(parts)


def _embedding_from_entry(entry: dict[str, Any]) -> list[float]:
    sources = [entry, _star_source(entry)]
    for source in sources:
        for key in ("embedding", "vector", "embedding_32", "embedding_64", "embedding_128", "embedding_512", "embedding_2048"):
            embedding = _coerce_embedding(source.get(key))
            if embedding and any(abs(value) > 1.0e-8 for value in embedding):
                return embedding
    return []


def _text_from_entry(entry: dict[str, Any]) -> str:
    source = _star_source(entry)
    candidates = [
        source.get("name"),
        source.get("label"),
        source.get("title"),
        source.get("text"),
        source.get("content"),
        source.get("description"),
        source.get("summary"),
        source.get("symbol"),
        source.get("meaning_rpn"),
        source.get("visual_rpn"),
        source.get("behavior_rpn"),
        source.get("law_rpn"),
        source.get("answer_text"),
        _surface_form_text(source.get("surface_forms")),
        entry.get("name"),
        entry.get("content"),
        entry.get("description"),
        entry.get("summary"),
        entry.get("answer_text"),
        entry.get("id"),
        entry.get("star_id"),
    ]
    tokens = [str(value).strip() for value in candidates if str(value or "").strip() and str(value).strip().lower() != "none"]
    return " ".join(tokens)


def _ref_ids_from_entry(entry: dict[str, Any]) -> list[str]:
    source = _star_source(entry)
    raw_refs: list[Any] = []
    for key in (
        "component_refs",
        "router_refs",
        "executor_refs",
        "validator_refs",
        "anti_pattern_refs",
        "visual_refs",
        "grammar_refs",
        "math_refs",
        "reality_refs",
        "audio_refs",
        "meta_refs",
        "composite_of",
    ):
        values = source.get(key)
        if isinstance(values, list):
            raw_refs.extend(values)
    refs: list[str] = []
    for value in raw_refs:
        text = str(value or "").strip()
        if text and text not in refs:
            refs.append(text)
    return refs


def _entry_id(entry: dict[str, Any]) -> str:
    source = _star_source(entry)
    for key in ("_id", "star_id", "id", "name"):
        value = source.get(key) if isinstance(source, dict) else None
        if str(value or "").strip():
            return str(value).strip()
        value = entry.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _entry_to_star(entry: dict[str, Any], star_type: int, galaxy_id: int, fallback_id: str) -> dict[str, Any] | None:
    embedding = _embedding_from_entry(entry)
    source_text = _text_from_entry(entry)
    if not embedding:
        if not source_text.strip():
            return None
        embedding = embed_text_sovereign(source_text)
    star_id = _entry_id(entry) or fallback_id
    flags = 0x01
    if _embedding_from_entry(entry):
        flags |= 0x02
    source = _star_source(entry)

    def _list_field(*keys: str) -> list[str]:
        values: list[str] = []
        for key in keys:
            for container in (source, entry):
                raw = container.get(key)
                if not isinstance(raw, list):
                    continue
                for item in raw:
                    text = str(item or "").strip()
                    if text and text not in values:
                        values.append(text)
        return values

    return {
        "id": star_id,
        "_id": star_id,
        "_ref_ids": _ref_ids_from_entry(entry),
        "embedding": embedding[:32],
        "galaxy_id": int(galaxy_id) & 0xFFFFFFFF,
        "star_type": int(star_type),
        "route_family": (
            str(source.get("route_family") or entry.get("route_family") or "").strip().upper()
        ),
        "selection_role": str(source.get("selection_role") or entry.get("selection_role") or "").strip().lower(),
        "layer_id": source.get("layer_id", source.get("layer", entry.get("layer_id", entry.get("layer", 0)))),
        "answer_eligible": bool(source.get("answer_eligible") or entry.get("answer_eligible")),
        "router_refs": _list_field("router_refs"),
        "executor_refs": _list_field("executor_refs"),
        "validator_refs": _list_field("validator_refs"),
        "anti_pattern_refs": _list_field("anti_pattern_refs"),
        "route_policy": dict(source.get("route_policy") or entry.get("route_policy") or {}),
        "attractive_prior": float(source.get("attractive_prior", entry.get("attractive_prior", 0.0)) or 0.0),
        "repulsive_prior": float(source.get("repulsive_prior", entry.get("repulsive_prior", 0.0)) or 0.0),
        "component_refs": [],
        "flags": flags,
    }


def load_all_galaxies_from_disk(galaxy_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Load all disk-backed galaxy entries, keeping foundational stars first."""
    stars = build_foundational_galaxy_table()
    id_to_index: dict[str, int] = {}
    for index, star in enumerate(stars):
        star_id = str(star.get("_id") or star.get("id") or "").strip()
        if star_id:
            id_to_index[star_id] = index

    search_dir = Path(galaxy_dir or GALAXY_JSONL_DIR)
    if not search_dir.exists():
        return stars

    for jsonl_path in _iter_disk_jsonl_paths(search_dir):
        star_type = _galaxy_type_from_filename(jsonl_path.name)
        galaxy_id = _fnv1a32(jsonl_path.stem)
        try:
            with jsonl_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except Exception:
                        continue
                    fallback_id = f"{jsonl_path.stem}:{line_number}"
                    star = _entry_to_star(entry, star_type, galaxy_id, fallback_id)
                    if star is None:
                        continue
                    star_id = str(star.get("_id") or "").strip()
                    if star_id and star_id in id_to_index:
                        continue
                    index = len(stars)
                    if star_id:
                        id_to_index[star_id] = index
                    stars.append(star)
        except Exception:
            continue

    def _resolve_ref_ids(values: Any) -> list[int]:
        resolved: list[int] = []
        if not isinstance(values, list):
            return resolved
        for value in values:
            ref_id = str(value or "").strip()
            if not ref_id or ref_id not in id_to_index:
                continue
            ref_index = int(id_to_index[ref_id])
            if ref_index not in resolved:
                resolved.append(ref_index)
        return resolved

    for star in stars:
        ref_ids = list(star.pop("_ref_ids", []) or [])
        star["component_refs"] = _resolve_ref_ids(ref_ids)
        for key in ("router_refs", "executor_refs", "validator_refs", "anti_pattern_refs"):
            star[key] = _resolve_ref_ids(star.get(key))
    return stars


__all__ = [
    "GALAXY_JSONL_DIR",
    "HOUSE_JSONL_DIR",
    "GALAXY_NAME_TO_TYPE",
    "load_all_galaxies_from_disk",
]
