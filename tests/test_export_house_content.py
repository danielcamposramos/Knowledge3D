from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.tools.export_house_content import export_house_content


def test_export_house_content_produces_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "house-content.json"
    result = export_house_content(output)
    assert result["books"] == 5
    assert result["book_entries"] == 85
    assert result["concepts"] == 10
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "Book/MathematicsPrimer" in data["books"]


def test_book_entries_have_required_fields(tmp_path: Path) -> None:
    output = tmp_path / "house-content.json"
    export_house_content(output)
    data = json.loads(output.read_text(encoding="utf-8"))
    for book in data["books"].values():
        for entry in book["entries"]:
            assert "star_id" in entry
            assert "meaning_class" in entry
            assert "surface_forms" in entry
            assert "taxonomy_refs" in entry
            assert "grammar_refs" in entry
