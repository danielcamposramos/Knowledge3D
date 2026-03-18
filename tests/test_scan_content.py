from __future__ import annotations

from pathlib import Path

from knowledge3d.tools.scan_content import guess_domain, scan_content


def test_scan_recognizes_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    manifest = scan_content([tmp_path / "doc.pdf", tmp_path / "notes.txt", tmp_path / "data.csv"])

    assert manifest["total_files"] == 3
    assert manifest["by_type"]["document"] == 1
    assert manifest["by_type"]["text"] == 1
    assert manifest["by_type"]["tabular"] == 1


def test_scan_skips_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")

    manifest = scan_content([tmp_path / "binary.exe"])

    assert manifest["total_files"] == 0
    assert manifest["entries"] == []


def test_domain_guess_from_path() -> None:
    assert guess_domain(Path("/books/mathematics/calculus.pdf")) == "Mathematics"
    assert guess_domain(Path("/random/stuff.txt")) == "General"
