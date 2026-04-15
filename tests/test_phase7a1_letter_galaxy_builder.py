import json
from pathlib import Path

from knowledge3d.ingestion.letter_galaxy_builder import (
    build_letter_galaxy,
    canonical_letter_entry,
    register_mathematical_role_symlink_kind,
)
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


DEJAVU_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEJAVU_SERIF = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")


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
        return star_id


def _manifest(*paths: Path) -> dict:
    fonts = []
    for path in paths:
        family = "DejaVu Sans" if "Sans" in path.name else "DejaVu Serif"
        fonts.append(
            {
                "path": str(path),
                "file": path.name,
                "font_index": 0,
                "family": family,
                "style": "Book",
                "scripts": ["latn", "grek", "number"],
                "codepoint_count": 256,
                "codepoint_ranges": [["U+0030", "U+0039"], ["U+0041", "U+005A"], ["U+0061", "U+007A"], ["U+03C0", "U+03C0"]],
            }
        )
    return {"fonts": fonts}


def test_letter_star_contains_all_manifest_font_drawings():
    assert DEJAVU_SANS.exists(), "required_system_font_missing:DejaVuSans.ttf"
    assert DEJAVU_SERIF.exists(), "required_system_font_missing:DejaVuSerif.ttf"
    build = build_letter_galaxy(_manifest(DEJAVU_SANS, DEJAVU_SERIF), codepoints=[ord("A")])

    star = build.stars["char_a"]
    assert star["star_id"] == "char_a"
    assert star["meaning_class"] == "form"
    assert star["lod_class"] == "LOD_ICON"
    assert star["script"] == "latn"
    assert star["unicode_category"] == "Lu"
    assert len(star["font_glyphs"]) == 2
    assert {row["family"] for row in star["font_glyphs"]} == {"DejaVu Sans", "DejaVu Serif"}
    assert all(row["rpn_program"].endswith("STROKE") for row in star["font_glyphs"])
    assert all(row["font_glyph_star_id"].startswith("font_glyph_") for row in star["font_glyphs"])
    assert build.glyph_failures == []


def test_digit_mathematical_role_links_only_when_target_exists():
    target = MeaningCentricStar(star_id="concept_digit_zero", meaning_rpn="0 INTEGER DIGIT", domain="Math")
    build = build_letter_galaxy(
        _manifest(DEJAVU_SANS),
        existing_targets={"concept_digit_zero": target},
        codepoints=[ord("0"), ord("1")],
    )

    assert build.stars["char_0"]["taxonomy_refs"] == ["concept_digit_zero"]
    assert build.target_updates["concept_digit_zero"]["component_refs"] == ["char_0"]
    assert build.stars["char_1"]["taxonomy_refs"] == []
    assert build.skipped_links == [{"source": "char_1", "target": "concept_digit_one", "reason": "target_missing"}]


def test_greek_mathematical_role_uses_existing_registry_targets():
    target = MeaningCentricStar(star_id="concept_math_pi", meaning_rpn="PI CONSTANT", domain="Math")
    build = build_letter_galaxy(
        _manifest(DEJAVU_SANS),
        existing_targets={"concept_math_pi": target},
        codepoints=[ord("π")],
    )

    assert build.stars["char_u03c0"]["taxonomy_refs"] == ["concept_math_pi"]
    assert build.target_updates["concept_math_pi"]["component_refs"] == ["char_u03c0"]


def test_canonical_letter_entry_and_mathematical_role_registration():
    entry = canonical_letter_entry(ord("A"), font_count=2)
    lookup = FakeCanonicalLookup()
    registered = register_mathematical_role_symlink_kind(lookup)

    assert entry["kind"] == "letter_star"
    assert entry["key"] == "U+0041"
    assert entry["star_id"] == "char_a"
    assert entry["metadata"]["script"] == "latn"
    assert entry["metadata"]["font_count"] == 2
    assert registered == "mathematical_role"
    assert lookup.calls == [
        {
            "kind": "symlink_kind",
            "key": "mathematical_role",
            "star_id": "mathematical_role",
            "metadata": {"forward_field": "taxonomy_refs", "backward_field": "component_refs"},
        }
    ]


def test_letter_galaxy_build_can_write_jsonl(tmp_path):
    build = build_letter_galaxy(_manifest(DEJAVU_SANS), codepoints=[ord("A")])
    out = tmp_path / "letters.jsonl"

    from knowledge3d.ingestion.letter_galaxy_builder import write_letter_galaxy_build

    written = write_letter_galaxy_build(build, out)
    rows = [json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()]

    assert written == out
    assert rows[0]["star_id"] == "char_a"
    assert out.with_suffix(".jsonl.meta.json").exists()


def test_letter_builder_skips_manifest_unreadable_codepoint():
    manifest = _manifest(DEJAVU_SANS, DEJAVU_SERIF)
    manifest["fonts"][0]["unreadable_codepoints"] = ["U+0041"]

    build = build_letter_galaxy(manifest, codepoints=[ord("A")])

    star = build.stars["char_a"]
    assert len(star["font_glyphs"]) == 1
    assert star["font_glyphs"][0]["family"] == "DejaVu Serif"
