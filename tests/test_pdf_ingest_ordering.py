from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_pdf_types
from scripts import fundamental_ingest_pdfs


def test_root_slug_uses_stable_overrides() -> None:
    assert analyze_pdf_types.root_slug(1, Path("/tmp/Encyclopedias")) == "01_encyclopedias"
    assert analyze_pdf_types.root_slug(2, Path("/tmp/EchoSystems Default Libraries")) == "02_default_libraries"


def test_analyze_root_emits_ordered_artifacts_and_ignores_non_pdfs(tmp_path, monkeypatch) -> None:
    root = tmp_path / "Encyclopedias"
    nested = root / "nested"
    nested.mkdir(parents=True)

    pdf_a = root / "a.pdf"
    pdf_b = nested / "b.pdf"
    pdf_c = nested / "c.pdf"
    pdf_d = root / "d.pdf"
    for path in (pdf_a, pdf_b, pdf_c, pdf_d):
        path.write_bytes(b"%PDF-1.4\n")
    (root / "sidecar.json").write_text("{}", encoding="utf-8")
    (nested / "notes.js").write_text("console.log(1)", encoding="utf-8")

    mapping = {
        str(pdf_a): {"path": str(pdf_a), "type": "vector", "pages": 3},
        str(pdf_b): {"path": str(pdf_b), "type": "mixed", "pages": 11},
        str(pdf_c): {"path": str(pdf_c), "type": "scanned_no_text", "pages": 21},
        str(pdf_d): {"path": str(pdf_d), "type": "error", "pages": 0, "error": "broken pdf"},
    }

    def fake_classify(pdf_path: Path, sample_pages: int = 5) -> dict[str, object]:
        assert sample_pages == 3
        return dict(mapping[str(pdf_path)])

    monkeypatch.setattr(analyze_pdf_types, "classify_pdf", fake_classify)

    output_dir = tmp_path / "results"
    summary = analyze_pdf_types.analyze_root(root, output_dir=output_dir, sample_pages=3)

    assert summary["inventory"] == {"pdf": 4, "json": 1, "other": 1}
    assert summary["discovered_pdf_count"] == 4
    assert summary["eligible_pdf_count"] == 2
    assert summary["ocr_needed_count"] == 1
    assert summary["error_count"] == 1

    eligible = (output_dir / "eligible_pdfs.txt").read_text(encoding="utf-8").splitlines()
    assert eligible == [str(pdf_b), str(pdf_a)]

    ocr_needed = (output_dir / "ocr_needed_pdfs.txt").read_text(encoding="utf-8").splitlines()
    assert ocr_needed == [str(pdf_c)]

    extraction_errors = (output_dir / "extraction_errors.txt").read_text(encoding="utf-8").splitlines()
    assert extraction_errors == [f"{pdf_d}\tbroken pdf"]

    inventory = json.loads((output_dir / "all_pdf_inventory.json").read_text(encoding="utf-8"))
    assert [record["path"] for record in inventory] == [str(pdf_a), str(pdf_d), str(pdf_b), str(pdf_c)]


def test_iter_pdf_paths_uses_pdf_list_order_and_filters_non_pdfs(tmp_path) -> None:
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    json_sidecar = tmp_path / "meta.json"
    pdf_a.write_bytes(b"%PDF-1.4\n")
    pdf_b.write_bytes(b"%PDF-1.4\n")
    json_sidecar.write_text("{}", encoding="utf-8")

    pdf_list = tmp_path / "eligible_pdfs.txt"
    pdf_list.write_text(
        "\n".join(
            [
                str(pdf_b),
                str(json_sidecar),
                str(tmp_path / "missing.pdf"),
                "",
                str(pdf_a),
            ]
        ),
        encoding="utf-8",
    )

    ordered = fundamental_ingest_pdfs._iter_pdf_paths(
        pdf=None,
        pdf_dir=None,
        pdf_list=pdf_list,
        pattern="**/*.pdf",
        limit=0,
    )
    assert ordered == [pdf_b, pdf_a]

    limited = fundamental_ingest_pdfs._iter_pdf_paths(
        pdf=None,
        pdf_dir=None,
        pdf_list=pdf_list,
        pattern="**/*.pdf",
        limit=1,
    )
    assert limited == [pdf_b]
