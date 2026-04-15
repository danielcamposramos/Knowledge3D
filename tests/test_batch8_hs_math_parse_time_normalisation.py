from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.hs_math_parser import MathSymlinkNormaliseError, normalise_symlink_ref, parse_cluster1_bullets
from knowledge3d.ingestion.math_semantic_aliases import SYMBOL_ALIASES, UNICODE_TO_NAME
from tests._batch8_helpers import write_fixture


def test_parse_time_normalisation_collapses_all_supported_dialects() -> None:
    assert normalise_symlink_ref("star.letter.a") == "letter::a"
    assert normalise_symlink_ref("letter::a") == "letter::a"
    assert normalise_symlink_ref("letter_a") == "letter::a"
    assert normalise_symlink_ref("star.symbol.sqrt") == "symbol::sqrt"
    assert normalise_symlink_ref("symbol::\u221a") == "symbol::sqrt"
    assert normalise_symlink_ref("constant::\u03c0") == "constant::pi"
    assert normalise_symlink_ref("concept_area") == "concept::area"


def test_unicode_to_name_round_trip_covers_symbol_aliases() -> None:
    for glyph, alias in UNICODE_TO_NAME.items():
        assert alias in SYMBOL_ALIASES or alias == "pi"


def test_unknown_unicode_glyph_raises() -> None:
    try:
        normalise_symlink_ref("symbol::\u2603")
    except MathSymlinkNormaliseError:
        pass
    else:
        raise AssertionError("expected MathSymlinkNormaliseError")


def test_parser_preserves_raw_and_normalised_refs(tmp_path: Path) -> None:
    source = write_fixture(
        tmp_path,
        text="\n".join(
            [
                "#### rule_demo",
                "- **canonical_id**: `rule_demo`",
                "- **is_a**: `concept_demo`",
                "- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.a]`",
                "- **symlinks**: `star.letter.a, symbol::\u221a, constant::\u03c0, concept_area`",
                "- **surface_forms**:",
                '  - en: "demo"',
                '  - pt: "demo"',
                '  - es: "demo"',
                '  - fr: "demo"',
                '  - de: "demo"',
                '  - it: "demo"',
                '  - ja: "demo"',
                '  - zh: "demo"',
                '  - ru: "demo"',
            ]
        ),
    )
    row = parse_cluster1_bullets(source.read_text(encoding="utf-8"), source_file=source.name)[0]
    assert row.symlink_refs_raw == ("star.letter.a", "symbol::\u221a", "constant::\u03c0", "concept_area")
    assert row.symlink_refs_norm == ("letter::a", "symbol::sqrt", "constant::pi", "concept::area")
