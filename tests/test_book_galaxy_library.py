from __future__ import annotations

import json
from pathlib import Path


def _write_demo_book(book_root: Path) -> Path:
    book_dir = book_root / "demo_book"
    book_dir.mkdir(parents=True, exist_ok=True)
    (book_dir / "metadata.json").write_text(
        json.dumps(
            {
                "book_id": "demo_book",
                "title": "Demo Book",
                "author": "Unit Test",
                "domain": "linear_algebra",
                "page_count": 1,
                "schema_version": "book_galaxy_v0",
                "artifact_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (book_dir / "token_index.json").write_text(
        json.dumps(
            {
                "determinant": [1],
                "matrix": [1],
            }
        ),
        encoding="utf-8",
    )
    (book_dir / "pages_text.jsonl").write_text(
        json.dumps(
            {
                "page_number": 1,
                "text": "Determinant of a 2x2 matrix: det([[a,b],[c,d]]) = a*d - b*c.",
                "embedding_index": 0,
                "position_3d": [0.1, 0.2, 0.3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Articulated artifacts (optional) - used by the book-aware TTC path.
    artifact = {
        "artifact_id": "demo_book_p1_formula_det2x2_0",
        "artifact_type": "formula",
        "name": "2x2 determinant",
        "domain": "linear_algebra",
        "book_id": "demo_book",
        "page_number": 1,
        "conditions": ["matrix is 2x2"],
        "lhs": "det([[a,b],[c,d]])",
        "rhs": "a*d - b*c",
        "lhs_rpn": "det",
        "rhs_rpn": "a d * b c * -",
        "rpn": "a d * b c * -",
        "derived_rpns": [],
        "var_mapping": {"a": "a", "b": "b", "c": "c", "d": "d"},
        "symbol_bindings": {},
        "source": "unit_test",
        "raw_block": "Determinant of a 2x2 matrix: det([[a,b],[c,d]]) = a*d - b*c.",
    }
    (book_dir / "artifacts.jsonl").write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    (book_dir / "artifact_index.json").write_text(
        json.dumps(
            {
                "determinant": [artifact["artifact_id"]],
                "matrix": [artifact["artifact_id"]],
                "det": [artifact["artifact_id"]],
            }
        ),
        encoding="utf-8",
    )
    return book_dir


def test_book_galaxy_library_discovers_and_searches(tmp_path):
    from knowledge3d.training.math_benchmarks.book_galaxy_library import BookGalaxyLibrary

    _write_demo_book(tmp_path)

    lib = BookGalaxyLibrary(books_root=tmp_path)
    books = lib.list_books()
    assert len(books) == 1
    assert books[0]["book_id"] == "demo_book"

    hits = lib.search(normalized_tokens=["determinant", "matrix"], top_k=3)
    assert hits
    assert hits[0].book_id == "demo_book"
    assert hits[0].page_number == 1
    assert hits[0].score >= 2
    assert "determinant" in hits[0].excerpt.lower()


def test_book_galaxy_library_searches_artifacts(tmp_path):
    from knowledge3d.training.math_benchmarks.book_galaxy_library import BookGalaxyLibrary

    _write_demo_book(tmp_path)

    lib = BookGalaxyLibrary(books_root=tmp_path)
    hits = lib.search_artifacts(normalized_tokens=["determinant", "matrix"], top_k=3)
    assert hits
    assert hits[0].book_id == "demo_book"
    assert hits[0].artifact_type == "formula"
    assert hits[0].rpn


def test_trm_galaxy_reader_generates_book_seed_candidates(tmp_path):
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
    from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES
    from knowledge3d.training.arc_agi.math_symbol_galaxy import MATH_GALAXY
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    book_root = tmp_path / "books"
    _write_demo_book(book_root)

    wg = WordGalaxy(storage_path=tmp_path / "wg")
    gg = GrammarGalaxy()

    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=gg,
        math_galaxy=MATH_GALAXY,
        rule_bank=GALAXY_AWARE_RULES,
        shadow_copy=None,
        enable_book_galaxies=True,
        book_galaxy_root=str(book_root),
        book_top_k=3,
    )

    candidates, hits = reader._generate_book_galaxy_candidates(  # noqa: SLF001
        "Compute the determinant of matrix [[1, 2], [3, 4]].",
        max_candidates=6,
    )
    assert hits
    assert "1 4 * 2 3 * -" in candidates
