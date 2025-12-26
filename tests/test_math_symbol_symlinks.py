from __future__ import annotations

import json
from pathlib import Path


def test_math_symbol_galaxy_variant_lookup() -> None:
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY

    assert MATH_GALAXY.lookup("\\cos") is not None
    assert MATH_GALAXY.lookup("cos") is not None
    assert MATH_GALAXY.lookup("cos") is MATH_GALAXY.lookup("\\cos")

    assert MATH_GALAXY.lookup("\\pi") is not None
    assert MATH_GALAXY.lookup("π") is not None
    assert MATH_GALAXY.lookup("π") is MATH_GALAXY.lookup("\\pi")


def test_trm_reader_book_query_expands_math_variants(tmp_path: Path) -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps(
            [
                {
                    "page": 1,
                    "content": "Trig note\ncos angle identity\ncos(x) = adjacent / hypotenuse",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Trig Book", book_id="trig_book", domain="trig")

    wg = WordGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=None,
        math_galaxy=MATH_GALAXY,
        rule_bank=[],
        enable_book_galaxies=True,
        book_galaxy_root=str(out_dir.parent),
        book_top_k=5,
    )

    # Prompt uses LaTeX surface form; books use pdftotext surface form ("cos").
    # Retrieval should bridge the mismatch via the Math Galaxy variant registry.
    _, meta, _ = reader._generate_book_galaxy_candidates("Compute \\cos of an angle.")
    assert meta, "Expected at least one book hit via math variant expansion"
    assert any(item.get("book_id") == "trig_book" for item in meta if isinstance(item, dict))
