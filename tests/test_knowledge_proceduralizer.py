from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.tools.knowledge_proceduralizer import (
    GSM8K_DEFAULT_PATH,
    MMLU_DEFAULT_PATH,
    PROCEDURALIZATION_SYSTEM_PROMPT,
    _extract_json,
    _parse_response,
    _subject_to_domain,
    build_rag_context,
    load_gsm8k_entries,
    load_mmlu_entries,
)


MMLU_PATH = MMLU_DEFAULT_PATH
GSM8K_PATH = GSM8K_DEFAULT_PATH
HAS_MMLU = (MMLU_PATH / "val").exists()
HAS_GSM8K = (GSM8K_PATH / "grade_school_math" / "data" / "train.jsonl").exists()


@pytest.mark.skipif(not HAS_MMLU, reason="MMLU data not available")
def test_load_mmlu_val_entries() -> None:
    entries = list(load_mmlu_entries(MMLU_PATH, "val", subjects=["astronomy"], limit_per_subject=2))

    assert len(entries) >= 1
    assert "Correct Answer:" in entries[0]["content"]
    assert entries[0]["subject"] == "astronomy"


@pytest.mark.skipif(not HAS_GSM8K, reason="GSM8K data not available")
def test_load_gsm8k_entries() -> None:
    entries = list(load_gsm8k_entries(GSM8K_PATH, limit=2))

    assert len(entries) >= 1
    assert "Step-by-step" in entries[0]["content"]
    assert entries[0]["domain_hint"] == "Mathematics"


def test_rag_context_chemistry() -> None:
    context = build_rag_context("Physics", "college_chemistry", "What is the atomic number of carbon?")

    assert "element_" in context


def test_rag_context_physics() -> None:
    context = build_rag_context("Physics", "astronomy", "What is the speed of light?")

    assert "constant_" in context


def test_rag_context_always_has_taxonomy() -> None:
    context = build_rag_context("General", "philosophy", "What is virtue?")

    assert "concept_" in context
    assert "synset_" in context


def test_rag_context_has_h19_reference() -> None:
    context = build_rag_context("General", "any", "any question")

    assert "synset_" in context


def test_extract_json_clean() -> None:
    result = _extract_json('{"meaning_class": "fact", "domain": "Physics"}')

    assert result is not None
    assert result["meaning_class"] == "fact"


def test_extract_json_fenced() -> None:
    result = _extract_json('Result:\n```json\n{"meaning_class": "rule"}\n```')

    assert result is not None
    assert result["meaning_class"] == "rule"


def test_extract_json_with_thinking() -> None:
    raw = '<think>analysis...</think>\n{"meaning_class": "fact", "summary": "test"}'
    result = _parse_response(raw, {"domain_hint": "General", "subject": "test"})

    assert result.summary == "test"


def test_subject_to_domain() -> None:
    assert _subject_to_domain("college_physics") == "Physics"
    assert _subject_to_domain("abstract_algebra") == "Mathematics"
    assert _subject_to_domain("college_biology") == "Biology"
    assert _subject_to_domain("world_religions") == "General"


def test_system_prompt_has_symlink_principle() -> None:
    assert "symlink" in PROCEDURALIZATION_SYSTEM_PROMPT.lower() or "REFERENCE" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "star_refs" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "English" in PROCEDURALIZATION_SYSTEM_PROMPT
