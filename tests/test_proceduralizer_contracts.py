from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.ingestion.proceduralizer_contract import (
    PROCEDURALIZER_BUNDLE_JSON_SCHEMA,
    PROCEDURALIZER_MODEL_PROFILES,
    ProceduralizerBundle,
    ProceduralizerReceipt,
    ProceduralizerRequest,
    parse_bundle,
)
from knowledge3d.ingestion.proceduralizer_wine import ProceduralizerWineBridge
from knowledge3d.knowledgeverse.proceduralizer_stargate import (
    build_row_enrichment_context,
    bundle_to_payload_rows,
    load_external_enrichment_context,
    second_pass_enrich_payload_rows,
)
from knowledge3d.tools import ingest_from_manifest


class _FakeOllama:
    def __init__(self, output: str, returncode: int = 0, stderr: str = "") -> None:
        self._output = output
        self._returncode = returncode
        self._stderr = stderr
        self.calls: list[dict[str, object]] = []

    def chat(self, **_: object):
        self.calls.append(dict(_))
        class _Result:
            def __init__(self, output: str, returncode: int, stderr: str) -> None:
                self.output = output
                self.returncode = returncode
                self.stderr = stderr

        return _Result(self._output, self._returncode, self._stderr)


def test_parse_bundle_handles_skip_action() -> None:
    request = ProceduralizerRequest(
        source_kind="pdf",
        source_id="page_1",
        source_path="paper.pdf#page=1",
        domain_hint="General",
        content="References only",
    )
    bundle, schema_ok, failure_code = parse_bundle('{"ingest_action":"skip","knowledge_packets":[]}', request)

    assert schema_ok is True
    assert failure_code == ""
    assert bundle.ingest_action == "skip"
    assert bundle.knowledge_packets == []


def test_bundle_to_payload_rows_emits_route_exempt_meaning_rows() -> None:
    request = ProceduralizerRequest(
        source_kind="text",
        source_id="entry_1",
        source_path="",
        domain_hint="General",
        content="Definition text",
    )
    bundle, schema_ok, _ = parse_bundle(
        json.dumps(
            {
                "ingest_action": "augment",
                "knowledge_packets": [
                    {
                        "layer_kind": "meaning",
                        "meaning_class": "definition",
                        "meaning_rpn": "GENERAL FACT ENTRY",
                        "summary": "definition anchor",
                        "domain": "General",
                        "surface_forms": {"en": "definition anchor"},
                        "taxonomy_refs": ["concept_general"],
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        request,
    )

    rows = bundle_to_payload_rows(bundle, request)

    assert schema_ok is True
    assert len(rows) == 1
    assert rows[0]["entry"]["selection_role"] == "unknown"
    assert rows[0]["entry"]["metadata"]["sovereign_route_exempt"] is True
    assert rows[0]["entry"]["metadata"]["foundational_layer_id"] == 2


def test_parse_bundle_invalid_json_rejects_without_packets() -> None:
    request = ProceduralizerRequest(
        source_kind="text",
        source_id="bad_json_1",
        source_path="eval://bad_json_1",
        domain_hint="General",
        content="not valid json",
    )

    bundle, schema_ok, failure_code = parse_bundle("timed out", request)

    assert schema_ok is False
    assert failure_code == "invalid_json"
    assert bundle.ingest_action == "reject"
    assert bundle.knowledge_packets == []


def test_wine_bridge_writes_receipt_and_capture(tmp_path: Path) -> None:
    fake = _FakeOllama(
        output=json.dumps(
            {
                "ingest_action": "augment",
                "knowledge_packets": [
                    {
                        "layer_kind": "rule",
                        "meaning_class": "rule",
                        "meaning_rpn": "A B ADD",
                        "summary": "addition rule anchor",
                        "domain": "Mathematics",
                        "surface_forms": {"en": "addition rule anchor"},
                        "taxonomy_refs": ["concept_mathematics"],
                        "confidence": 0.95,
                    }
                ],
            }
        )
    )
    output = json.dumps(
        {
            "ingest_action": "augment",
            "knowledge_packets": [
                {
                    "layer_kind": "rule",
                    "meaning_class": "rule",
                    "meaning_rpn": "A B ADD",
                    "summary": "addition rule anchor",
                    "domain": "Mathematics",
                    "surface_forms": {"en": "addition rule anchor"},
                    "taxonomy_refs": ["concept_mathematics"],
                    "confidence": 0.95,
                }
            ],
        }
    )
    bridge = ProceduralizerWineBridge(
        capture_dir=tmp_path,
        ollama=fake,
    )
    request = ProceduralizerRequest(
        source_kind="text",
        source_id="math_1",
        source_path="eval://math_1",
        domain_hint="Mathematics",
        content="2 plus 3 equals 5",
        quality_profile="quality",
    )

    receipt = bridge.submit(request, model_profile="quality")

    assert receipt.status == "completed"
    assert receipt.schema_ok is True
    assert receipt.model == PROCEDURALIZER_MODEL_PROFILES["quality"]
    assert Path(receipt.raw_response_path).exists()
    assert receipt.parsed_bundle.ingest_action == "augment"
    assert receipt.parsed_bundle.knowledge_packets[0].domain == "Mathematics"
    assert fake.calls
    assert fake.calls[0]["response_format"] == PROCEDURALIZER_BUNDLE_JSON_SCHEMA


def test_wine_bridge_detects_plan_limit_and_sets_retry(tmp_path: Path) -> None:
    bridge = ProceduralizerWineBridge(
        capture_dir=tmp_path,
        ollama=_FakeOllama("", returncode=1, stderr="Plan limit reached. Reset after some hours."),
    )
    request = ProceduralizerRequest(
        source_kind="text",
        source_id="limit_1",
        source_path="eval://limit_1",
        domain_hint="General",
        content="test",
    )

    receipt = bridge.submit(request, model_profile="quality")

    assert receipt.failure_code == "plan_limit_consumed"
    assert receipt.retry_after_utc


def test_wine_bridge_does_not_flag_timeout_words_inside_valid_json(tmp_path: Path) -> None:
    fake = _FakeOllama(
        output=json.dumps(
            {
                "ingest_action": "augment",
                "knowledge_packets": [
                    {
                        "layer_kind": "rule",
                        "meaning_class": "rule",
                        "meaning_rpn": "receipt write timeout artifact keep",
                        "summary": "receipts are written before cleanup on timeout",
                        "domain": "Tools",
                        "surface_forms": {"en": "timeout receipt policy"},
                        "taxonomy_refs": ["concept_tool"],
                        "meta_refs": ["policy:timeout"],
                        "confidence": 0.9,
                    }
                ],
            }
        )
    )
    bridge = ProceduralizerWineBridge(capture_dir=tmp_path, ollama=fake)
    request = ProceduralizerRequest(
        source_kind="text",
        source_id="timeout_policy",
        source_path="eval://timeout_policy",
        domain_hint="Tools",
        content="Receipts must be written before cleanup on timeout.",
    )

    receipt = bridge.submit(request, model_profile="quality")

    assert receipt.status == "completed"


def test_second_pass_enrichment_deepens_history_symlinks() -> None:
    rows = [
        {
            "galaxy": "Word",
            "entry": {
                "id": "notation_bce_dating",
                "name": "B.C.E. dating notation",
                "category": "proceduralizer_form",
                "rpn_program": "dating notation bce",
                "metadata": {
                    "surface_forms": {"en": "B.C.E."},
                    "procedural_layer_kind": "form",
                    "taxonomy_refs": [],
                    "word_refs": [],
                    "symbol_refs": [],
                    "grammar_refs": [],
                    "reality_refs": [],
                    "meta_refs": [],
                    "relationships": [],
                },
                "selection_role": "unknown",
                "layer_id": 0,
                "answer_eligible": False,
                "sovereign_route_exempt": True,
            },
        },
        {
            "galaxy": "Reality",
            "entry": {
                "id": "era_paleolithic",
                "name": "Paleolithic era from 2 million B.C.E. to 10,000 B.C.E. in prehistory",
                "category": "proceduralizer_meaning",
                "rpn_program": "paleolithic era prehistory bce",
                "metadata": {
                    "surface_forms": {"en": "Paleolithic Era"},
                    "procedural_layer_kind": "meaning",
                    "taxonomy_refs": [],
                    "word_refs": [],
                    "symbol_refs": [],
                    "grammar_refs": [],
                    "reality_refs": [],
                    "meta_refs": [],
                    "relationships": [
                        {"from": "era_paleolithic", "relation": "precedes", "to": "era_mesolithic"},
                        {"from": "era_paleolithic", "relation": "part_of", "to": "prehistory"},
                    ],
                },
                "selection_role": "unknown",
                "layer_id": 0,
                "answer_eligible": False,
                "sovereign_route_exempt": True,
            },
        },
    ]

    context = build_row_enrichment_context(rows)
    enriched = second_pass_enrich_payload_rows(rows, context=context)
    metadata = enriched[1]["entry"]["metadata"]

    assert "prehistory" in metadata["taxonomy_refs"]
    assert "high_school_world_history" in metadata["taxonomy_refs"]
    assert "notation_bce_dating" in metadata["symbol_refs"]
    assert "era_mesolithic" in metadata["reality_refs"]
    assert metadata["foundational_layer_id"] == 2
    assert "second_pass_symlink_enriched" in metadata["meta_refs"]


def test_second_pass_enrichment_keeps_form_packets_route_exempt_but_linked() -> None:
    rows = [
        {
            "galaxy": "Grammar",
            "entry": {
                "id": "script_hieroglyphic_egyptian",
                "name": "Egyptian hieroglyphic writing system",
                "category": "proceduralizer_form",
                "rpn_program": "egyptian hieroglyphic writing system",
                "metadata": {
                    "surface_forms": {"en": "hieroglyphic writing"},
                    "procedural_layer_kind": "form",
                    "taxonomy_refs": [],
                    "word_refs": [],
                    "symbol_refs": [],
                    "grammar_refs": [],
                    "reality_refs": [],
                    "meta_refs": [],
                    "relationships": [],
                },
                "selection_role": "unknown",
                "layer_id": 0,
                "answer_eligible": False,
                "sovereign_route_exempt": True,
            },
        }
    ]

    enriched = second_pass_enrich_payload_rows(rows)
    entry = enriched[0]["entry"]
    metadata = entry["metadata"]

    assert entry["selection_role"] == "unknown"
    assert entry["layer_id"] == 0
    assert entry["sovereign_route_exempt"] is True
    assert metadata["foundational_layer_kind"] == "form"
    assert metadata["foundational_layer_id"] == 1
    assert "concept_language" in metadata["taxonomy_refs"]
    assert "grammar_forward_entity_extraction" in metadata["grammar_refs"]


def test_load_external_enrichment_context_reads_live_checkpoint_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "galaxy_consolidated_latest.json"
    checkpoint.write_text(
        json.dumps(
            {
                "galaxies": {
                    "Reality": [
                        {
                            "id": "concept_history",
                            "name": "History concept anchor",
                            "rpn_program": "history concept anchor",
                            "metadata": {"surface_forms": {"en": "history"}},
                        },
                        {
                            "entry": {
                                "id": "grammar_subject_domain_alignment",
                                "name": "Grammar subject domain alignment",
                                "rpn_program": "grammar subject domain alignment",
                                "metadata": {"surface_forms": {"en": "subject domain alignment"}},
                            }
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    context = load_external_enrichment_context(checkpoint)

    assert "concept_history" in context["id_set"]
    assert "grammar_subject_domain_alignment" in context["id_set"]
    assert "history" in context["token_to_ids"]
    assert "concept_history" in context["token_to_ids"]["history"]


def test_manifest_ingest_stops_on_plan_limit(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("alpha beta gamma", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "doc_1",
                        "name": "Sample",
                        "path": str(source_path),
                        "content_type": "text",
                        "domain_hint": "General",
                    },
                    {
                        "id": "doc_2",
                        "name": "ShouldNotRun",
                        "path": str(source_path),
                        "content_type": "text",
                        "domain_hint": "General",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    def _fake_proceduralize_text_content(**_: object):
        request = ProceduralizerRequest(
            source_kind="manifest",
            source_id="doc_1",
            source_path=str(source_path),
            domain_hint="General",
            content="alpha beta gamma",
        )
        receipt = ProceduralizerReceipt(
            status="transport_error",
            provider="ollama",
            model=PROCEDURALIZER_MODEL_PROFILES["quality"],
            latency_ms=1,
            request_hash="req",
            response_hash="resp",
            raw_response_path=str(tmp_path / "resp.txt"),
            schema_ok=False,
            failure_code="plan_limit_consumed",
            retry_after_utc="2026-04-06T05:01:00+00:00",
            parsed_bundle=ProceduralizerBundle(ingest_action="reject", knowledge_packets=[]),
        )
        return receipt, request

    monkeypatch.setattr(ingest_from_manifest, "proceduralize_text_content", _fake_proceduralize_text_content)

    report = ingest_from_manifest.ingest_manifest(
        manifest_path,
        provider_name="ollama",
        model=None,
        model_profile="quality",
        timeout_seconds=20.0,
        capture_dir=None,
        output_dir=tmp_path / "out",
        galaxy_manager=None,
    )

    assert report["stopped_due_to_plan_limit"] is True
    assert report["retry_after_utc"] == "2026-04-06T05:01:00+00:00"
    assert len(report["entries"]) == 1


def test_manifest_ingest_rejects_invalid_receipt(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("alpha beta gamma", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "doc_1",
                        "name": "Sample",
                        "path": str(source_path),
                        "content_type": "text",
                        "domain_hint": "General",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def _fake_proceduralize_text_content(**_: object):
        request = ProceduralizerRequest(
            source_kind="manifest",
            source_id="doc_1",
            source_path=str(source_path),
            domain_hint="General",
            content="alpha beta gamma",
        )
        receipt = ProceduralizerReceipt(
            status="invalid_json",
            provider="ollama",
            model=PROCEDURALIZER_MODEL_PROFILES["quality"],
            latency_ms=1,
            request_hash="req",
            response_hash="resp",
            raw_response_path=str(tmp_path / "resp.txt"),
            schema_ok=False,
            failure_code="timeout",
            parsed_bundle=ProceduralizerBundle(ingest_action="reject", knowledge_packets=[]),
        )
        return receipt, request

    monkeypatch.setattr(ingest_from_manifest, "proceduralize_text_content", _fake_proceduralize_text_content)

    report = ingest_from_manifest.ingest_manifest(
        manifest_path,
        provider_name="ollama",
        model=None,
        model_profile="quality",
        timeout_seconds=20.0,
        capture_dir=None,
        output_dir=tmp_path / "out",
        galaxy_manager=None,
    )

    assert report["ingested"] == 0
    assert report["rejected"] == 1
    assert report["entries"][0]["status"] == "rejected"
