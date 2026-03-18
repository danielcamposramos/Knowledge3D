from __future__ import annotations

from knowledge3d.tools.augmentation_providers import AugmentationResult
from knowledge3d.tools.content_to_stars import result_to_star


def test_result_to_star_produces_valid_star() -> None:
    result = AugmentationResult(
        summary="Test content",
        entities=[],
        relationships=[],
        domain="Mathematics",
        meaning_rpn_hint="MATH CONTENT TEST",
        taxonomy_refs=["concept_mathematics"],
        surface_forms={"en": "Test Entry", "pt": "Entrada Teste"},
        confidence=0.9,
        provider="ollama",
        raw_response="{}",
    )

    star = result_to_star(result, star_id="test_entry_001")

    assert star.star_id == "test_entry_001"
    assert star.house_room == "House/Library"
    assert "concept_mathematics" in star.taxonomy_refs
    assert star.behavior_rpn == "INSPECT LOAD_CONTENT"
    assert star.confidence == 1
