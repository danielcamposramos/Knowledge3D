from __future__ import annotations

from typing import Any

from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table


ROLE_REF_KEYS = (
    "router_refs",
    "executor_refs",
    "validator_refs",
    "anti_pattern_refs",
)


def resolve_star_refs(stars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_to_index = {
        str(star.get("_id") or star.get("id") or "").strip(): index
        for index, star in enumerate(stars)
        if str(star.get("_id") or star.get("id") or "").strip()
    }

    def _resolve(values: Any) -> list[int]:
        resolved: list[int] = []
        if not isinstance(values, list):
            return resolved
        for value in values:
            ref_id = str(value or "").strip()
            if not ref_id:
                continue
            ref_index = id_to_index.get(ref_id)
            if ref_index is None or ref_index in resolved:
                continue
            resolved.append(int(ref_index))
        return resolved

    resolved_stars: list[dict[str, Any]] = []
    for star in stars:
        row = dict(star)
        row["component_refs"] = _resolve(row.get("_ref_ids"))
        for key in ROLE_REF_KEYS:
            row[key] = _resolve(row.get(key))
        resolved_stars.append(row)
    return resolved_stars


def build_resolved_foundational_stars() -> list[dict[str, Any]]:
    return resolve_star_refs(build_foundational_galaxy_table())
