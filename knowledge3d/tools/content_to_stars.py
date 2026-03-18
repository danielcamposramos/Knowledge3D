"""Convert augmented content into MeaningCentricStar entries."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

from knowledge3d.knowledgeverse._house_utils import char_refs
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar, SurfaceForm

from .augmentation_providers import AugmentationResult


DOMAIN_TO_ROOM = {
    "Mathematics": "House/Library",
    "Physics": "House/Library",
    "Biology": "House/Garden",
    "Language": "House/Library",
    "Tools": "House/Workshop",
    "Visual": "House/Gallery",
    "Audio": "House/Gallery",
    "General": "House/Library",
}


def _slug(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "entry"


def _word_ref(text: str, language: str) -> str:
    slug = _slug(text)
    if slug != "entry":
        return slug
    stripped = str(text or "").strip()
    if stripped:
        return f"{language}_u{ord(stripped[0]):04x}"
    return f"{language}_entry"


def _build_surface_forms(values: dict[str, str]) -> dict[str, SurfaceForm]:
    out: dict[str, SurfaceForm] = {}
    for language, text in dict(values or {}).items():
        lang = str(language).strip().lower()
        raw = str(text).strip()
        if not lang or not raw:
            continue
        out[lang] = SurfaceForm(word_ref=_word_ref(raw, lang), char_refs=char_refs(raw, lang))
    if "en" not in out:
        out["en"] = SurfaceForm(word_ref="entry", char_refs=["char_e"])
    return out


def _confidence_to_trit(value: float) -> int:
    resolved = float(value)
    if resolved >= 0.67:
        return 1
    if resolved <= 0.2:
        return -1
    return 0


def result_to_star(
    result: AugmentationResult,
    *,
    star_id: str | None = None,
    house_room: str | None = None,
    house_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    meta_refs: list[str] | None = None,
) -> MeaningCentricStar:
    """Convert an augmentation result into a meaning-centric star."""
    if house_room is None:
        house_room = DOMAIN_TO_ROOM.get(result.domain, "House/Library")
    refs = list(result.taxonomy_refs)
    if not refs and result.domain != "General":
        refs.append(f"concept_{result.domain.lower()}")
    return MeaningCentricStar(
        star_id=str(star_id or "").strip(),
        meaning_class="entry",
        meaning_rpn=result.meaning_rpn_hint or f"{result.domain.upper()} CONTENT ENTRY",
        domain=f"House/{result.domain}",
        taxonomy_refs=refs,
        surface_forms=_build_surface_forms(result.surface_forms),
        behavior_rpn="INSPECT LOAD_CONTENT",
        meta_refs=list(meta_refs or []),
        house_room=house_room,
        house_position=house_position,
        confidence=_confidence_to_trit(result.confidence),
        polarity=1,
    )


def results_to_stars(
    results: Iterable[AugmentationResult],
    *,
    id_prefix: str = "ingested",
) -> list[MeaningCentricStar]:
    """Convert a batch of augmentation results into stars."""
    stars: list[MeaningCentricStar] = []
    for index, result in enumerate(results):
        title = result.surface_forms.get("en") or result.summary or result.domain
        star = result_to_star(
            result,
            star_id=f"{id_prefix}_{_slug(title)}_{index:04d}",
        )
        stars.append(star)
    return stars


def write_stars_jsonl(stars: Iterable[MeaningCentricStar], output_path: str | Path) -> Path:
    """Persist stars as JSONL for downstream ingestion/export."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for star in stars:
            handle.write(json.dumps(star.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


__all__ = [
    "DOMAIN_TO_ROOM",
    "result_to_star",
    "results_to_stars",
    "write_stars_jsonl",
]
