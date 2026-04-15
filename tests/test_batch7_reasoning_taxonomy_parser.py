from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from knowledge3d.ingestion.reasoning_taxonomy_parser import parse_catalogue


def test_parse_automated_reasoning_catalogue_matches_repo_reality() -> None:
    payload = parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md"))
    assert len(payload.stars) >= 10
    assert payload.stars[0].star_id == "concept_automated_reasoning"
    assert len({row.star_id for row in payload.stars}) == len(payload.stars)


def test_parse_full_wave_extracts_known_reasoning_taxonomy_stars() -> None:
    payloads = [
        parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_AUTOMATED_REASONING_2026-04-13.md")),
        parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_AML_AND_SOLVERS_2026-04-13.md")),
        parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_HEURISTICS_AND_METAHEURISTICS_2026-04-13.md")),
        parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md")),
    ]
    all_star_ids = [row.star_id for payload in payloads for row in payload.stars]
    assert len(all_star_ids) >= 20
    assert "concept_heuristic" in all_star_ids
    assert "aml_aimms" in all_star_ids


def test_parser_dataclasses_round_trip_through_asdict() -> None:
    payload = parse_catalogue(Path("TEMP/KIMI_KNOWLEDGE_EXTENSION_AML_HEURISTICS_REASONING_2026-04-13.md"))
    row = payload.stars[0]
    serialized = asdict(row)
    assert serialized["star_id"] == row.star_id
    assert tuple(serialized["surface_forms"]) == row.surface_forms
    assert tuple(serialized["taxonomy_refs"]) == row.taxonomy_refs


def test_malformed_star_table_raises_with_source_context(tmp_path: Path) -> None:
    path = tmp_path / "bad_catalogue.md"
    path.write_text(
        "\n".join(
            [
                "## Canonical Star Table",
                "| Star ID | Class | Domain |",
                "|---|---|---|",
                "| `concept_bad` | concept | Test |",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reasoning_taxonomy_missing_required_columns"):
        parse_catalogue(path)
