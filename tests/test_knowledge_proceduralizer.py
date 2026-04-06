from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.ingestion.proceduralizer_contract import (
    ProceduralizerBundle,
    ProceduralizerPacket,
    ProceduralizerReceipt,
)
from knowledge3d.tools.augmentation_providers import AugmentationResult
from knowledge3d.tools.knowledge_proceduralizer import (
    GSM8K_DEFAULT_PATH,
    MMLU_DEFAULT_PATH,
    MODEL_OPTIONS,
    PROCEDURALIZATION_SYSTEM_PROMPT,
    PROCEDURALIZER_CHUNK_OVERLAP_CHARS,
    PROCEDURALIZER_MAX_CONTENT_CHARS,
    _merge_receipts,
    _extract_json,
    _payload_contract,
    _parse_response,
    _subject_to_domain,
    build_rag_context,
    chunk_source_content,
    load_math_entries,
    load_mmlu_entries,
    receipt_is_usable,
    result_to_payload_row,
)


MMLU_PATH = MMLU_DEFAULT_PATH
GSM8K_PATH = GSM8K_DEFAULT_PATH
SPEC_PATH = Path("docs/vocabulary/KNOWLEDGE_PROCEDURALIZER_SPECIFICATION.md")
HAS_MMLU = (MMLU_PATH / "val").exists()
HAS_GSM8K = (GSM8K_PATH / "grade_school_math" / "data" / "train.jsonl").exists()


@pytest.mark.skipif(not HAS_MMLU, reason="MMLU data not available")
def test_load_mmlu_val_entries() -> None:
    entries = list(load_mmlu_entries(MMLU_PATH, "val", subjects=["astronomy"], limit_per_subject=2))

    assert len(entries) >= 1
    assert "Correct Answer:" in entries[0]["content"]
    assert entries[0]["subject"] == "astronomy"


@pytest.mark.skipif(not HAS_GSM8K, reason="GSM8K data not available")
def test_load_math_entries() -> None:
    entries = list(load_math_entries(GSM8K_PATH, limit=2))

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
    lowered = PROCEDURALIZATION_SYSTEM_PROMPT.lower()
    assert "symlink" in lowered or "reference" in lowered
    assert "form -> meaning -> rules -> meta-rules" in lowered
    assert "knowledge_packets" in PROCEDURALIZATION_SYSTEM_PROMPT
    assert "strict json only" in lowered


def test_spec_mentions_context_reset_overlap_and_retry_window() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "clear model context between distinct sources" in lowered
    assert "preserve overlap between adjacent chunks" in lowered
    assert "5 hours + 1 minute" in lowered


def test_model_options_define_bounded_context_and_disable_thinking() -> None:
    for options in MODEL_OPTIONS.values():
        assert int(options["num_ctx"]) > 0
        assert bool(options["think"]) is False


def test_chunk_source_content_uses_overlap() -> None:
    content = "a" * (PROCEDURALIZER_MAX_CONTENT_CHARS + 500)
    chunks = chunk_source_content(content)

    assert len(chunks) >= 2
    assert chunks[0][-PROCEDURALIZER_CHUNK_OVERLAP_CHARS:] == chunks[1][:PROCEDURALIZER_CHUNK_OVERLAP_CHARS]


def test_merge_receipts_supports_long_hash_ids() -> None:
    packet = ProceduralizerPacket(
        layer_kind="meaning",
        meaning_class="definition",
        meaning_rpn="GENERAL FACT ENTRY",
        summary="chunk anchor",
        domain="General",
        surface_forms={"en": "chunk anchor"},
    )
    receipt_a = ProceduralizerReceipt(
        status="completed",
        provider="ollama",
        model="qwen3.5:397b-cloud",
        latency_ms=10,
        request_hash="req_a",
        response_hash="resp_a",
        raw_response_path="a.txt",
        schema_ok=True,
        failure_code="",
        parsed_bundle=ProceduralizerBundle(ingest_action="augment", knowledge_packets=[packet]),
    )
    receipt_b = ProceduralizerReceipt(
        status="completed",
        provider="ollama",
        model="qwen3.5:397b-cloud",
        latency_ms=12,
        request_hash="req_b",
        response_hash="resp_b",
        raw_response_path="b.txt",
        schema_ok=True,
        failure_code="",
        parsed_bundle=ProceduralizerBundle(ingest_action="augment", knowledge_packets=[packet]),
    )

    merged = _merge_receipts([receipt_a, receipt_b], provider="ollama", model="qwen3.5:397b-cloud")

    assert merged.request_hash
    assert merged.response_hash
    assert len(merged.request_hash) == 16
    assert len(merged.response_hash) == 16
    assert len(merged.parsed_bundle.knowledge_packets) == 2


def test_receipt_is_usable_requires_schema_clean_augment_packets() -> None:
    packet = ProceduralizerPacket(
        layer_kind="meaning",
        meaning_class="definition",
        meaning_rpn="GENERAL FACT ENTRY",
        summary="usable anchor",
        domain="General",
        surface_forms={"en": "usable anchor"},
    )
    good = ProceduralizerReceipt(
        status="completed",
        provider="ollama",
        model="glm-5:cloud",
        latency_ms=10,
        request_hash="req_a",
        response_hash="resp_a",
        raw_response_path="a.txt",
        schema_ok=True,
        failure_code="",
        parsed_bundle=ProceduralizerBundle(ingest_action="augment", knowledge_packets=[packet]),
    )
    bad = ProceduralizerReceipt(
        status="invalid_json",
        provider="ollama",
        model="glm-5:cloud",
        latency_ms=10,
        request_hash="req_b",
        response_hash="resp_b",
        raw_response_path="b.txt",
        schema_ok=False,
        failure_code="timeout",
        parsed_bundle=ProceduralizerBundle(ingest_action="augment", knowledge_packets=[packet]),
    )

    assert receipt_is_usable(good) is True
    assert receipt_is_usable(bad) is False


def test_payload_contract_maps_domains_to_meaning_families() -> None:
    math_contract = _payload_contract(
        {"domain_hint": "Mathematics"},
        AugmentationResult("m", [], [], "Mathematics", "MATH ENTRY", [], {"en": "m"}, 0.9, "test", "{}"),
    )
    grammar_contract = _payload_contract(
        {"domain_hint": "Language"},
        AugmentationResult("g", [], [], "Language", "GRAMMAR ENTRY", [], {"en": "g"}, 0.9, "test", "{}"),
    )
    general_contract = _payload_contract(
        {"domain_hint": "Physics"},
        AugmentationResult("r", [], [], "Physics", "GENERAL ENTRY", [], {"en": "r"}, 0.9, "test", "{}"),
    )

    assert math_contract["route_family"] == "MATH"
    assert grammar_contract["route_family"] == "GRAMMAR"
    assert general_contract["route_family"] == "GENERAL"


def test_result_to_payload_row_emits_route_metadata_without_benchmark_name_leakage() -> None:
    entry = {
        "entry_id": "mmlu_val_astronomy_0",
        "source": "mmlu_val",
        "subject": "astronomy",
        "question": "What planet is known as the Red Planet?",
        "domain_hint": "Physics",
    }
    result = AugmentationResult(
        summary="planetary recall anchor",
        entities=[],
        relationships=[],
        domain="Physics",
        meaning_rpn_hint="PLANET COLOR COMPARE",
        taxonomy_refs=["concept_physics"],
        surface_forms={"en": "planetary recall anchor"},
        confidence=0.91,
        provider="test",
        raw_response='{"star_refs":["planet_mars"]}',
    )

    row = result_to_payload_row(result, entry)
    payload = dict(row["entry"])

    assert row["galaxy"] == "Reality"
    assert payload["route_family"] == "GENERAL"
    assert payload["selection_role"] == "executor"
    assert payload["layer_id"] == 3
    assert payload["answer_eligible"] is False
    assert payload["validator_refs"] == ["general_consistency_validator", "general_answer_validator"]
    assert payload["anti_pattern_refs"] == [
        "anti_pattern_missing_evidence_consistency",
        "anti_pattern_generic_language_factual_winner",
    ]
    assert "mmlu" not in str(payload["id"]).lower()
    assert "mmlu" not in str(payload["category"]).lower()
