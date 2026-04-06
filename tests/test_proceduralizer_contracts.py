from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.ingestion.proceduralizer_contract import (
    PROCEDURALIZER_MODEL_PROFILES,
    ProceduralizerBundle,
    ProceduralizerReceipt,
    ProceduralizerRequest,
    parse_bundle,
)
from knowledge3d.ingestion.proceduralizer_wine import ProceduralizerWineBridge
from knowledge3d.knowledgeverse.proceduralizer_stargate import bundle_to_payload_rows
from knowledge3d.tools import ingest_from_manifest


class _FakeOllama:
    def __init__(self, output: str, returncode: int = 0, stderr: str = "") -> None:
        self._output = output
        self._returncode = returncode
        self._stderr = stderr

    def chat(self, **_: object):
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


def test_wine_bridge_writes_receipt_and_capture(tmp_path: Path) -> None:
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
        ollama=_FakeOllama(output),
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
        capture_dir=None,
        output_dir=tmp_path / "out",
        galaxy_manager=None,
    )

    assert report["stopped_due_to_plan_limit"] is True
    assert report["retry_after_utc"] == "2026-04-06T05:01:00+00:00"
    assert len(report["entries"]) == 1
