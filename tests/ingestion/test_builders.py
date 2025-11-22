import json
from pathlib import Path

from knowledge3d.ingestion.atomic.letter_meaning_builder import build_letter_stars
from knowledge3d.ingestion.atomic.word_meaning_builder import build_word_stars
from knowledge3d.ingestion.atomic.math_symbol_builder import build_math_symbols
from knowledge3d.ingestion.atomic.math_grammar_builder import build_math_grammar
from knowledge3d.ingestion.atomic.segmenter import syllabify, morph_segment


def _write_tmp(tmp_path: Path, lines):
    p = tmp_path / "data.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return p


def test_letter_builder_groups_case_variants(tmp_path):
    lines = [
        {"character": "A", "visual_rpn": "A_UP", "font_metadata": {"family": "Arial"}},
        {"character": "a", "visual_rpn": "A_LOW", "font_metadata": {"family": "Arial"}},
    ]
    path = _write_tmp(tmp_path, lines)
    stars = build_letter_stars(path)
    assert len(stars) == 1
    star = stars[0]
    assert star["letter_concept"] == "LETTER_A_LATIN"
    cases = {g["case"] for g in star["glyph_variants"]}
    assert cases == {"uppercase", "lowercase"}


def test_word_builder_sense_and_letter_refs(tmp_path):
    lines = [
        {"lang": "en", "lemma": "apple", "sense": "fruit", "meaning_id": "WORD_en_apple_fruit", "morph_rpn": "MORPH", "meaning_program": "MEAN"},
    ]
    path = _write_tmp(tmp_path, lines)
    stars = build_word_stars(path)
    assert len(stars) == 1
    star = stars[0]
    assert star["meaning_id"] == "WORD_en_apple_fruit"
    assert star["letter_refs"][0]["letter_concept"].startswith("LETTER_A_LATIN")


def test_math_symbol_builder_defaults(tmp_path):
    lines = [{"symbol": "+", "visual_rpn": "PLUS"}]
    symbols = build_math_symbols(_write_tmp(tmp_path, lines))
    assert symbols[0]["symbol_concept"] == "ADDITION_OPERATOR"
    assert "math_rpn" in symbols[0]["procedural_programs"]


def test_math_grammar_precedence():
    grammar = build_math_grammar()
    assert grammar and grammar[0]["precedence"][0]["group"] == "parentheses"


def test_segmenter_syllables_pt():
    syls = syllabify("casa", "pt")
    assert len(syls) >= 2
    assert syls[0]["syllable"].startswith("c")


def test_segmenter_morpheme_pt():
    segs = morph_segment("reconstrução", "pt")
    assert any("prefix" in s["morpheme_id"] for s in segs)


def test_segmenter_multilang_fr_it_de():
    assert syllabify("réseau", "fr")
    assert morph_segment("reimpostare", "it")  # ri- prefix likely
    assert morph_segment("unabhängig", "de")  # un- prefix likely


def test_segmenter_ar_cjk_fallbacks():
    ar = syllabify("سلام", "ar")
    assert ar and isinstance(ar[0]["syllable"], str)
    cjk = syllabify("你好", "zh")
    assert len(cjk) == 2


def test_drawing_grammar_builder(tmp_path):
    from knowledge3d.ingestion.atomic.drawing_grammar_builder import build_primitives, build_strokes, build_shapes, build_scenes, build_collections

    prims = build_primitives()
    strokes = build_strokes(prims)
    shapes = build_shapes(strokes)
    scenes = build_scenes(shapes)
    cols = build_collections(scenes)
    assert prims and strokes and shapes and scenes and cols
