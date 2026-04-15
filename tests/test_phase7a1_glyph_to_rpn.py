from pathlib import Path

from knowledge3d.ingestion.fonts.glyph_to_rpn import (
    TARGET_EM_SQUARE,
    build_system_font_manifest,
    extract_glyph_rpn,
    font_glyph_metadata,
    glyph_key,
    glyph_star_id,
    is_symbol_font,
    mark_unreadable_codepoints,
    register_font_glyph,
    script_for_codepoint,
)


DEJAVU_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
STIX_TWO_MATH = Path("/usr/share/fonts/opentype/stix/STIXTwoMath-Regular.otf")
FONTAWESOME = Path("/usr/share/fonts/opentype/font-awesome/FontAwesome.otf")


class FakeCanonicalLookup:
    def __init__(self):
        self.calls = []

    def register(self, *, kind, key, star_id, metadata):
        self.calls.append(
            {
                "kind": kind,
                "key": key,
                "star_id": star_id,
                "metadata": metadata,
            }
        )


def test_symbol_exclusion_keeps_stix_math_and_drops_pictogram_fonts():
    assert not is_symbol_font("STIX Two Math", STIX_TWO_MATH)
    assert is_symbol_font("FontAwesome", FONTAWESOME)
    assert is_symbol_font("Noto Emoji", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")
    assert is_symbol_font("Symbol", "/usr/share/fonts/truetype/msttcorefonts/Symbol.ttf")


def test_system_manifest_uses_text_font_entries(tmp_path):
    assert DEJAVU_SANS.exists(), "required_system_font_missing:DejaVuSans.ttf"
    manifest_path = tmp_path / "MANIFEST.json"
    manifest = build_system_font_manifest(font_paths=[DEJAVU_SANS], output_path=manifest_path)

    assert manifest_path.exists()
    assert manifest["font_count"] == 1
    entry = manifest["fonts"][0]
    assert entry["family"] == "DejaVu Sans"
    assert entry["style"]
    assert "latn" in entry["scripts"]
    assert entry["codepoint_count"] > 100
    assert entry["codepoint_ranges"]


def test_extract_glyph_rpn_emits_existing_drawing_ops():
    assert DEJAVU_SANS.exists(), "required_system_font_missing:DejaVuSans.ttf"
    glyph = extract_glyph_rpn(DEJAVU_SANS, ord("A"))

    assert glyph.contour_count >= 1
    assert glyph.opcode_count >= 4
    assert glyph.rpn_bytes == glyph.rpn_program.encode("utf-8")
    assert "MOVE" in glyph.rpn_program
    assert "LINE" in glyph.rpn_program
    assert glyph.rpn_program.endswith("STROKE")
    assert "DRAW_" not in glyph.rpn_program
    assert glyph.metrics.em_square == TARGET_EM_SQUARE
    assert glyph.metrics.advance_width > 0
    assert glyph.metrics.xmax > glyph.metrics.xmin


def test_extract_glyph_rpn_handles_curve_glyphs():
    assert DEJAVU_SANS.exists(), "required_system_font_missing:DejaVuSans.ttf"
    glyph = extract_glyph_rpn(DEJAVU_SANS, ord("ã"))

    assert glyph.contour_count >= 1
    assert any(op in glyph.rpn_program for op in ("QUAD", "CUBIC", "LINE"))
    assert glyph.metrics.ymax > glyph.metrics.ymin


def test_font_glyph_canonical_contract():
    glyph = extract_glyph_rpn(DEJAVU_SANS, ord("A"))
    metadata = font_glyph_metadata("DejaVu Sans", "Book", ord("A"), glyph)
    lookup = FakeCanonicalLookup()
    star_id = register_font_glyph(
        lookup,
        family="DejaVu Sans",
        style="Book",
        codepoint=ord("A"),
        glyph=glyph,
    )

    assert glyph_key("DejaVu Sans", "Book", ord("A")) == "DejaVu Sans::Book::U+0041"
    assert glyph_star_id("DejaVu Sans", "Book", ord("A")) == "font_glyph_dejavu_sans_book_u0041"
    assert metadata["script"] == "latn"
    assert metadata["em_square"] == TARGET_EM_SQUARE
    assert star_id == "font_glyph_dejavu_sans_book_u0041"
    assert lookup.calls == [
        {
            "kind": "font_glyph",
            "key": "DejaVu Sans::Book::U+0041",
            "star_id": "font_glyph_dejavu_sans_book_u0041",
            "metadata": metadata,
        }
    ]


def test_script_for_codepoint_covers_phase7_target_scripts():
    assert script_for_codepoint(ord("A")) == "latn"
    assert script_for_codepoint(ord("π")) == "grek"
    assert script_for_codepoint(ord("Ж")) == "cyrl"
    assert script_for_codepoint(ord("ع")) == "arab"
    assert script_for_codepoint(ord("ह")) == "deva"
    assert script_for_codepoint(ord("漢")) == "hani"
    assert script_for_codepoint(ord("あ")) == "hira"
    assert script_for_codepoint(ord("ก")) == "thai"


def test_mark_unreadable_codepoints_updates_per_font_manifest_entry():
    manifest = {
        "fonts": [
            {"path": "/fonts/a.ttf", "font_index": 0, "unreadable_codepoints": ["U+0041"]},
            {"path": "/fonts/b.ttf", "font_index": 1},
        ]
    }

    updated = mark_unreadable_codepoints(
        manifest,
        [
            {"font": "/fonts/a.ttf", "font_index": "0", "codepoint": "U+0042"},
            {"font": "/fonts/b.ttf", "font_index": "1", "codepoint": "U+03C0"},
        ],
    )

    assert updated["fonts"][0]["unreadable_codepoints"] == ["U+0041", "U+0042"]
    assert updated["fonts"][1]["unreadable_codepoints"] == ["U+03C0"]
