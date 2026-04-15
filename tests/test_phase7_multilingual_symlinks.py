from __future__ import annotations

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_word_star_id
from knowledge3d.ingestion.symlink_helpers import append_ref
from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import SynsetEntry, synset_to_star, synset_to_star_bundle
from knowledge3d.knowledgeverse._house_utils import char_refs, surface_forms
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar


def test_phase7_canonical_char_ids_are_language_agnostic() -> None:
    assert canonical_char_star_id("a") == "char_a"
    assert canonical_char_star_id("ã") == "char_u00e3"
    assert char_refs("ação", "pt") == ["char_a", "char_u00e7", "char_u00e3", "char_o"]


def test_phase7_house_surface_forms_use_canonical_word_ids() -> None:
    forms = surface_forms("Living Room", "Sala de Estar", "図書館")

    assert forms["en"].word_ref == "word_en_living_room"
    assert forms["pt"].word_ref == "word_pt_sala_de_estar"
    assert forms["ja"].word_ref == canonical_word_star_id("ja", "図書館")
    assert forms["ja"].word_ref != "word_ja_unknown"


def test_phase7_synset_to_star_uses_canonical_word_refs() -> None:
    entry = SynsetEntry(
        synset_id="00001740-a",
        pos="a",
        lemmas={"en": ["able"], "pt": ["capaz"], "ja": ["有能"]},
        definitions={"en": "having the power to perform"},
    )

    star = synset_to_star(entry)

    assert star.surface_forms["en"].word_ref == "word_en_able"
    assert star.surface_forms["pt"].word_ref == "word_pt_capaz"
    assert star.surface_forms["ja"].word_ref == canonical_word_star_id("ja", "有能")
    assert star.surface_forms["ja"].word_ref != "word_ja_unknown"
    assert star.lod_class == "lod_summary"


def test_phase7_bundle_builds_bidirectional_meaning_word_character_links() -> None:
    entry = SynsetEntry(
        synset_id="12345678-n",
        pos="n",
        lemmas={"en": ["water"], "pt": ["agua"]},
        definitions={"en": "transparent liquid"},
    )

    stars = synset_to_star_bundle(entry)
    by_id = {star.star_id: star for star in stars}

    meaning = by_id["synset_12345678_n"]
    word_en = by_id["word_en_water"]
    word_pt = by_id["word_pt_agua"]
    char_a = by_id["char_a"]

    assert meaning.surface_forms["en"].word_ref == word_en.star_id
    assert meaning.surface_forms["pt"].word_ref == word_pt.star_id
    assert meaning.surface_forms["en"].char_refs == ["char_w", "char_a", "char_t", "char_e", "char_r"]
    assert meaning.surface_forms["pt"].char_refs == ["char_a", "char_g", "char_u", "char_a"]
    assert meaning.star_id in word_en.taxonomy_refs
    assert meaning.star_id in word_pt.taxonomy_refs
    assert "char_a" in word_en.component_refs
    assert "char_a" in word_pt.component_refs
    assert word_en.star_id in char_a.composite_of
    assert word_pt.star_id in char_a.composite_of


def test_phase7_meaning_star_supports_untranslatable_languages_and_lod() -> None:
    star = MeaningCentricStar(
        star_id="concept_saudades_pt",
        meaning_class="concept",
        meaning_rpn="LONGING PAST IRRECOVERABLE",
        domain="Library/Linguistics/Emotion/Untranslatable",
        untranslatable_languages=["en", "ja"],
        lod_class="lod_full",
    )

    payload = star.to_dict()

    assert payload["untranslatable_languages"] == ["en", "ja"]
    assert payload["lod_class"] == "lod_full"


def test_phase7_symlink_helpers_cover_modal_reference_lists() -> None:
    star = MeaningCentricStar(star_id="concept_signal", meaning_class="concept", meaning_rpn="SIGNAL", domain="Test")

    append_ref(star, "visual_refs", "drawing_wave")
    append_ref(star, "audio_refs", "audio_tone")
    append_ref(star, "reality_refs", "physics_oscillator")

    assert star.visual_refs == ["drawing_wave"]
    assert star.audio_refs == ["audio_tone"]
    assert star.reality_refs == ["physics_oscillator"]
