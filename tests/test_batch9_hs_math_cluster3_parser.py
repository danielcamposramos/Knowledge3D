from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.hs_math_parser import parse_cluster1_bullets_with_diagnostics, parse_hs_math_file
from knowledge3d.ingestion.math_symlink_resolver import MathSymlinkResolver
from tests._batch8_helpers import FakeCanonicalLookup


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER3 = REPO_ROOT / "TEMP" / "KIMI_MATH_HS_CLUSTER3_STATS_DISCRETE_APPLIED_2026-04-13.md"


def test_cluster3_real_file_parses_with_no_summary_tail_failure() -> None:
    diagnostics = parse_cluster1_bullets_with_diagnostics(CLUSTER3.read_text(encoding="utf-8"), source_file=CLUSTER3.name)
    assert len(diagnostics.rows) == 57
    assert diagnostics.skipped_lines == ()
    assert diagnostics.rows[-1].canonical_id_raw == "formula_celsius_to_kelvin"


def test_cluster3_dispatcher_routes_to_bullet_parser() -> None:
    assert len(parse_hs_math_file(CLUSTER3)) == 57


def test_cluster3_greek_letters_and_greater_symbol_resolve() -> None:
    resolver = MathSymlinkResolver(FakeCanonicalLookup(), allowlist_path=None)
    assert resolver.resolve("letter::mu") == "char_u03bc"
    assert resolver.resolve("letter::sigma") == "char_u03c3"
    assert resolver.resolve("letter::lambda") == "char_u03bb"
    assert resolver.resolve("letter::rho") == "char_u03c1"
    assert resolver.resolve("symbol::greater") == "math_symbol_greater_than_sign"
