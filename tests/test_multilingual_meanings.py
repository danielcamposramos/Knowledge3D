from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (
    OMW_LANG_MAP,
    POS_MAP,
    build_meaning_layer_stars,
    iter_meaning_stars,
    load_all_omw,
    meaning_layer_stats,
    parse_omw_tab,
    synset_to_star,
)


OMW_PATH = Path("/K3D/K3D_llama_cpp/datasets/omw-data/omw-data-main/wns")
HAS_OMW = OMW_PATH.exists()


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_english() -> None:
    synsets = parse_omw_tab(OMW_PATH / "eng" / "wn-data-eng.tab", "en")

    assert len(synsets) > 1000
    assert "able" in synsets["00001740-a"].lemmas["en"]


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_parse_omw_tab_portuguese() -> None:
    synsets = parse_omw_tab(OMW_PATH / "por" / "wn-data-por.tab", "pt")

    assert "capaz" in synsets["00001740-a"].lemmas["pt"]


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_load_all_omw_merges() -> None:
    synsets = load_all_omw(OMW_PATH)

    assert len(synsets) > 10000
    entry = synsets["00001740-a"]
    assert len(entry.lemmas) >= 3


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_synset_to_star_english_primary() -> None:
    synsets = load_all_omw(OMW_PATH)
    star = synset_to_star(synsets["00001740-a"])

    assert star.star_id == "synset_00001740_a"
    assert "ABLE" in star.meaning_rpn
    assert "en" in star.surface_forms
    assert star.surface_forms["en"].word_ref == "en_able"


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_surface_forms_have_char_refs() -> None:
    synsets = load_all_omw(OMW_PATH)
    star = synset_to_star(synsets["00001740-a"])

    for lang, surface_form in star.surface_forms.items():
        assert len(surface_form.char_refs) > 0, f"Missing char_refs for {lang}"


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_min_languages_filter() -> None:
    stars = list(iter_meaning_stars(OMW_PATH, min_languages=5, limit=20))

    for star in stars:
        assert len(star.surface_forms) >= 5


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_pos_filter() -> None:
    nouns = list(iter_meaning_stars(OMW_PATH, min_languages=2, pos_filter={"n"}, limit=20))

    assert nouns
    assert all(star.meaning_class == "noun" for star in nouns)


@pytest.mark.skipif(not HAS_OMW, reason="OMW data not available")
def test_build_meaning_layer_stats() -> None:
    stars = build_meaning_layer_stars(OMW_PATH, min_languages=3, limit=100)
    stats = meaning_layer_stats(stars)

    assert stats["total_stars"] == 100
    assert stats["avg_languages_per_star"] >= 3.0


def test_pos_map_values() -> None:
    assert POS_MAP["n"] == "noun"
    assert POS_MAP["v"] == "verb"
    assert POS_MAP["a"] == "adjective"
    assert POS_MAP["r"] == "adverb"


def test_lang_map_key_languages() -> None:
    assert OMW_LANG_MAP["eng"] == "en"
    assert OMW_LANG_MAP["por"] == "pt"
    assert OMW_LANG_MAP["jpn"] == "ja"
