"""Export House book content and seed concepts as static companion JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.book_content_biology import BIOLOGY_ATLAS_ENTRIES
from knowledge3d.knowledgeverse.book_content_language import LANGUAGE_FOUNDATIONS_ENTRIES
from knowledge3d.knowledgeverse.book_content_mathematics import MATHEMATICS_PRIMER_ENTRIES
from knowledge3d.knowledgeverse.book_content_physics import PHYSICS_HANDBOOK_ENTRIES
from knowledge3d.knowledgeverse.book_content_tools import TOOL_MANUAL_ENTRIES
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from knowledge3d.knowledgeverse.seed_stars import SEED_STARS


def _star_to_content_entry(star: MeaningCentricStar) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "star_id": star.star_id,
        "meaning_class": star.meaning_class,
        "domain": star.domain,
        "meaning_rpn": star.meaning_rpn,
        "behavior_rpn": star.behavior_rpn,
        "surface_forms": {
            language: {"word_ref": form.word_ref, "char_refs": list(form.char_refs)}
            for language, form in star.surface_forms.items()
        },
        "taxonomy_refs": list(star.taxonomy_refs),
        "grammar_refs": list(star.grammar_refs),
        "component_refs": list(star.component_refs),
    }
    if star.visual_rpn:
        entry["visual_rpn"] = star.visual_rpn
    return entry


def export_house_content(output_path: Path) -> dict[str, Any]:
    books = {
        "Book/MathematicsPrimer": MATHEMATICS_PRIMER_ENTRIES,
        "Book/LanguageFoundations": LANGUAGE_FOUNDATIONS_ENTRIES,
        "Book/PhysicsHandbook": PHYSICS_HANDBOOK_ENTRIES,
        "Book/BiologyAtlas": BIOLOGY_ATLAS_ENTRIES,
        "Book/ToolManual": TOOL_MANUAL_ENTRIES,
    }
    payload = {
        "version": 1,
        "books": {
            galaxy_ref: {
                "entries": [_star_to_content_entry(entry) for entry in entries],
            }
            for galaxy_ref, entries in books.items()
        },
        "concepts": {
            star.star_id: _star_to_content_entry(star)
            for star in SEED_STARS
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    total_entries = sum(len(book["entries"]) for book in payload["books"].values())
    return {
        "books": len(payload["books"]),
        "book_entries": total_entries,
        "concepts": len(payload["concepts"]),
        "output": str(output_path),
    }


def main() -> int:
    output = Path("viewer/public/house-content.json")
    result = export_house_content(output)
    print(
        f"Exported content: {result['books']} books, "
        f"{result['book_entries']} entries, "
        f"{result['concepts']} concepts -> {result['output']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

