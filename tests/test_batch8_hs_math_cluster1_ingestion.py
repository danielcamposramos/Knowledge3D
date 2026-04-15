from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id
from knowledge3d.ingestion.math_symbol_builder import math_symbol_star_id

from scripts.ingest_hs_math_cluster1 import run_cluster1_ingestion
from tests._batch8_helpers import FakeCanonicalLookup, write_fixture


def test_cluster1_ingestion_dry_run_reports_forward_refs_and_no_writes(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    lookup = FakeCanonicalLookup()
    summary = run_cluster1_ingestion(lookup, source=source, write=False)
    assert summary["rows"] == 2
    assert "concept::arithmetic_precedence" in summary["forward_refs"]
    assert len(lookup.records) == 0


def test_cluster1_ingestion_write_registers_meaning_stars_and_edges(tmp_path: Path) -> None:
    source = write_fixture(tmp_path)
    lookup = FakeCanonicalLookup(
        preset_star_ids={
            canonical_char_star_id("("),
            math_symbol_star_id("+"),
        }
    )
    summary = run_cluster1_ingestion(lookup, source=source, write=True)
    assert summary["meaning_star_written"] == 2
    assert summary["math_symlink_written"] >= 1
    assert summary["confirmation"]["misses"] == []
