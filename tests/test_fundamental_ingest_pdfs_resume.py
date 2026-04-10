from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "fundamental_ingest_pdfs.py"
    spec = importlib.util.spec_from_file_location("fundamental_ingest_pdfs_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_overwrite_and_payload_rebuild(tmp_path: Path):
    mod = _load_script_module()

    stage_root = tmp_path / "stage"
    payload_out = tmp_path / "payload.jsonl"
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7 test")

    pdf_stage = mod._pdf_stage_dir(stage_root, "abc123")

    old_rows = [{"galaxy": "Math", "entry": {"id": "math_old", "name": "old"}}]
    mod._write_stage_page(
        pdf_stage_dir=pdf_stage,
        page_num=5,
        pdf_path=pdf,
        total_pages=10,
        decision={"classification": "knowledge"},
        rows=old_rows,
    )
    new_rows = [{"galaxy": "Math", "entry": {"id": "math_new", "name": "new"}}]
    mod._write_stage_page(
        pdf_stage_dir=pdf_stage,
        page_num=5,
        pdf_path=pdf,
        total_pages=10,
        decision={"classification": "knowledge"},
        rows=new_rows,
    )

    by_galaxy, by_classification, per_pdf, payload_rows = mod._rebuild_payload_from_stage(
        stage_root=stage_root,
        payload_output=payload_out,
    )

    assert payload_rows == 1
    assert by_galaxy["Math"] == 1
    assert by_classification["knowledge"] == 1
    assert per_pdf[str(pdf)]["rows_generated"] == 1

    lines = payload_out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["entry"]["id"] == "math_new"


def test_compose_ocr_page_text_includes_visual_sections() -> None:
    mod = _load_script_module()

    text = mod._compose_ocr_page_text(
        {
            "page_text": "Primary body text.",
            "image_descriptions": ["A labeled historical map."],
            "diagram_descriptions": ["A timeline linking empires and dynasties."],
            "table_descriptions": ["A two-column chronology table."],
        }
    )

    assert "Primary body text." in text
    assert "Image descriptions:" in text
    assert "A labeled historical map." in text
    assert "Diagram descriptions:" in text
    assert "Table descriptions:" in text


def test_proceduralize_pdf_page_retries_timeout_before_advancing(monkeypatch) -> None:
    mod = _load_script_module()

    class _FakeBundle:
        def __init__(self, ingest_action: str) -> None:
            self.ingest_action = ingest_action
            self.knowledge_packets = []

        def to_dict(self) -> dict[str, object]:
            return {"ingest_action": self.ingest_action, "knowledge_packets": []}

    class _FakeReceipt:
        def __init__(self, *, status: str, failure_code: str, schema_ok: bool, ingest_action: str) -> None:
            self.status = status
            self.failure_code = failure_code
            self.schema_ok = schema_ok
            self.provider = "ollama"
            self.model = "gemini-3-flash-preview:cloud"
            self.latency_ms = 1
            self.request_hash = "req"
            self.response_hash = "resp"
            self.raw_response_path = ""
            self.retry_after_utc = ""
            self.parsed_bundle = _FakeBundle(ingest_action)

        def to_dict(self) -> dict[str, object]:
            return {
                "status": self.status,
                "provider": self.provider,
                "model": self.model,
                "latency_ms": self.latency_ms,
                "request_hash": self.request_hash,
                "response_hash": self.response_hash,
                "raw_response_path": self.raw_response_path,
                "schema_ok": self.schema_ok,
                "failure_code": self.failure_code,
                "retry_after_utc": self.retry_after_utc,
                "parsed_bundle": self.parsed_bundle.to_dict(),
            }

    calls: list[float] = []

    def _fake_proceduralize_text_content(**kwargs):
        calls.append(float(kwargs["timeout"]))
        if len(calls) == 1:
            return _FakeReceipt(status="transport_error", failure_code="timeout", schema_ok=False, ingest_action="reject"), object()
        return _FakeReceipt(status="completed", failure_code="", schema_ok=True, ingest_action="augment"), object()

    monkeypatch.setattr(mod, "proceduralize_text_content", _fake_proceduralize_text_content)
    monkeypatch.setattr(mod, "receipt_is_usable", lambda receipt: bool(receipt.schema_ok))
    monkeypatch.setattr(mod, "bundle_to_payload_rows", lambda bundle, request: [{"galaxy": "Reality", "entry": {"id": "ok"}}])

    decision, rows, receipt_dict, bundle_dict, attempts, failure_code = mod._proceduralize_pdf_page(
        pdf_path=Path("/tmp/sample.pdf"),
        page_num=7,
        total_pages=10,
        page_text="Sample page text",
        pages={6: "Previous page", 7: "Sample page text", 8: "Next page"},
        provider="ollama",
        model="gemini-3-flash-preview:cloud",
        model_profile="quality",
        timeout=100.0,
        capture_dir=None,
        ollama=object(),
        retry_attempts=3,
        retry_timeout_multiplier=1.5,
    )

    assert calls == [100.0, 150.0]
    assert decision["classification"] == "knowledge"
    assert decision["attempt_count"] == 2
    assert len(rows) == 1
    assert receipt_dict["schema_ok"] is True
    assert bundle_dict["ingest_action"] == "augment"
    assert len(attempts) == 2
    assert failure_code == ""


def test_ocr_payload_requires_retry_for_needs_review_and_non_cloud() -> None:
    mod = _load_script_module()

    assert mod._ocr_payload_requires_retry(
        {
            "page_text": "Readable text",
            "ocr_quality": "good",
            "needs_review": False,
            "status": "completed",
            "model": "qwen3-vl:235b-instruct-cloud",
        }
    ) is False

    assert mod._ocr_payload_requires_retry(
        {
            "page_text": "Readable text",
            "ocr_quality": "good",
            "needs_review": True,
            "status": "completed",
            "model": "qwen3-vl:235b-instruct-cloud",
        }
    ) is True

    assert mod._ocr_payload_requires_retry(
        {
            "page_text": "Readable text",
            "ocr_quality": "good",
            "needs_review": False,
            "status": "completed",
            "model": "glm-ocr:latest",
        }
    ) is True


def test_delete_stage_pages_removes_only_requested_files(tmp_path: Path) -> None:
    mod = _load_script_module()

    pdf_stage = tmp_path / "stage"
    pdf_stage.mkdir(parents=True, exist_ok=True)
    keep = pdf_stage / "page_00001.json"
    drop = pdf_stage / "page_00002.json"
    keep.write_text("{}", encoding="utf-8")
    drop.write_text("{}", encoding="utf-8")

    removed = mod._delete_stage_pages(pdf_stage, {2, 9})

    assert removed == 1
    assert keep.exists() is True
    assert drop.exists() is False


def test_main_skips_staged_complete_pdf_without_reentering_ocr(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script_module()

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7 test")
    pdf_list = tmp_path / "pdfs.txt"
    pdf_list.write_text(str(pdf) + "\n", encoding="utf-8")

    stage_root = tmp_path / "stage"
    payload_out = tmp_path / "payload.jsonl"
    report_out = tmp_path / "report.json"
    storage_root = tmp_path / "storage"
    stage_root.mkdir(parents=True, exist_ok=True)
    storage_root.mkdir(parents=True, exist_ok=True)

    pdf_sha = mod._pdf_sha(pdf)
    pdf_stage = mod._pdf_stage_dir(stage_root, pdf_sha)
    mod._write_stage_page(
        pdf_stage_dir=pdf_stage,
        page_num=1,
        pdf_path=pdf,
        total_pages=1,
        decision={"classification": "knowledge"},
        rows=[{"galaxy": "Grammar", "entry": {"id": "row_1", "name": "Row 1"}}],
    )
    (stage_root / "manifest.json").write_text(
        json.dumps(
            {
                "pdfs": {
                    str(pdf): {
                        "sha256": pdf_sha,
                        "pages_total": 1,
                        "resume_from_page": 2,
                        "extraction_mode": "ocr_reconstructed",
                        "status": "staged_complete",
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def _fail_load(*_args, **_kwargs):
        raise AssertionError("staged_complete PDF should not re-enter page extraction/OCR")

    class _DummyOllama:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(mod, "_load_pdf_pages_for_ingest", _fail_load)
    monkeypatch.setattr(mod, "OllamaModelManager", _DummyOllama)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fundamental_ingest_pdfs.py",
            "--pdf-list",
            str(pdf_list),
            "--stage-dir",
            str(stage_root),
            "--payload-output",
            str(payload_out),
            "--report-output",
            str(report_out),
            "--storage-root",
            str(storage_root),
            "--disable-resume-last-page",
            "--no-repair-retryable-stage-pages",
        ],
    )

    rc = mod.main()

    assert rc == 0
    assert payload_out.exists() is True
    lines = payload_out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["entry"]["id"] == "row_1"
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["payload_rows"] == 1
    assert report["pdf_count"] == 1
