from __future__ import annotations

import importlib.util
import json
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
