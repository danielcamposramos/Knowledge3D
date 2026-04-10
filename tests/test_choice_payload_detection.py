from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_choice_payload_detection_accepts_common_choice_field_names() -> None:
    for field in ("options", "choices", "answers", "candidates", "alternatives"):
        assert Knowledgeverse._looks_like_choice_payload({field: ["A", "B", "C"]}) is True


def test_choice_payload_detection_accepts_explicit_options_argument() -> None:
    assert Knowledgeverse._looks_like_choice_payload({}, options=["A", "B"]) is True


def test_choice_payload_detection_rejects_missing_or_empty_lists() -> None:
    assert Knowledgeverse._looks_like_choice_payload({}) is False
    assert Knowledgeverse._looks_like_choice_payload({"choices": []}) is False
