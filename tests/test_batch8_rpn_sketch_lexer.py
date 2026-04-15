from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.rpn_sketch_lexer import classify_opcode, lex_rpn_sketch, write_coverage_report


def test_rpn_sketch_lexer_classifies_real_documentary_and_unknown(tmp_path: Path) -> None:
    tokens = lex_rpn_sketch("[TADD][GALAXY_LOOKUP star.symbol.plus][MYSTERY_OP x]")
    assert [token.opcode for token in tokens] == ["TADD", "GALAXY_LOOKUP", "MYSTERY_OP"]
    assert classify_opcode("TADD") == "real"
    assert classify_opcode("GALAXY_LOOKUP") == "documentary"
    assert classify_opcode("MYSTERY_OP") == "unknown"

    report = write_coverage_report(["[TADD][GALAXY_LOOKUP x][MYSTERY_OP y]"], output_path=tmp_path / "coverage.json")
    assert report["rows_scanned"] == 1
    assert report["opcode_histogram"]["TADD"] == 1
    assert report["unknown_opcodes"]["MYSTERY_OP"] == 1
