from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.ingestion.hs_math_parser import parse_cluster1_bullets_with_diagnostics, parse_hs_math_file
from tests._batch8_helpers import write_fixture


def test_cluster1_parser_extracts_rows_and_normalized_symlinks(tmp_path: Path) -> None:
    path = write_fixture(tmp_path)
    diagnostics = parse_cluster1_bullets_with_diagnostics(path.read_text(encoding="utf-8"), source_file=path.name)
    assert len(diagnostics.rows) == 2
    row = diagnostics.rows[1]
    assert row.canonical_id_raw == "rule_order_of_operations_pemdas"
    assert row.symlink_refs_raw[0] == "star.symbol.parenthesis"
    assert row.symlink_refs_norm[0] == "symbol::parenthesis"
    assert len(row.symlink_refs_raw) == len(row.symlink_refs_norm)


def test_dispatcher_routes_by_filename_and_cluster2_cluster3_are_deferred(tmp_path: Path) -> None:
    # cluster1 fixture still routes through the original path
    cluster1 = write_fixture(tmp_path, name="KIMI_MATH_HS_CLUSTER1_ARITHMETIC_ALGEBRA_2026-04-13.md")
    assert len(parse_hs_math_file(cluster1)) == 2

    repo_root = Path(__file__).resolve().parents[1]
    cluster2 = repo_root / "TEMP" / "KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md"
    cluster3 = repo_root / "TEMP" / "KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md"
    assert len(parse_hs_math_file(cluster2)) == 56
    assert len(parse_hs_math_file(cluster3)) == 57
