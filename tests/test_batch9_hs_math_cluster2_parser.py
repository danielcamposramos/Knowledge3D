from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.hs_math_parser import parse_cluster1_bullets_with_diagnostics, parse_hs_math_file
from knowledge3d.ingestion.math_symlink_resolver import MathSymlinkResolver
from tests._batch8_helpers import FakeCanonicalLookup


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER2 = REPO_ROOT / "TEMP" / "KIMI_MATH_HS_CLUSTER2_GEOMETRY_TRIG_2026-04-13.md"


def test_cluster2_real_file_parses_with_no_summary_tail_failure() -> None:
    diagnostics = parse_cluster1_bullets_with_diagnostics(CLUSTER2.read_text(encoding="utf-8"), source_file=CLUSTER2.name)
    assert len(diagnostics.rows) == 56
    assert diagnostics.skipped_lines == ()
    assert diagnostics.rows[-1].canonical_id_raw == "formula_dilation_origin"


def test_cluster2_dispatcher_routes_to_bullet_parser() -> None:
    assert len(parse_hs_math_file(CLUSTER2)) == 56


def test_cluster2_greek_letters_and_trig_symbols_resolve() -> None:
    resolver = MathSymlinkResolver(FakeCanonicalLookup(), allowlist_path=None)
    assert resolver.resolve("letter::theta") == "char_u03b8"
    assert resolver.resolve("letter::alpha") == "char_u03b1"
    assert resolver.resolve("symbol::sin") == "math_symbol_sin"
    assert resolver.resolve("symbol::cos") == "math_symbol_cos"
    assert resolver.resolve("symbol::tan") == "math_symbol_tan"
    assert resolver.resolve("symbol::greater") == "math_symbol_greater_than_sign"
