from __future__ import annotations

import json
from pathlib import Path


def test_book_galaxy_ingester_writes_templates(tmp_path: Path) -> None:
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps(
            [
                {
                    "page": 1,
                    "content": "Area formula\narea = length * width\nExample: length 3 width 4",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Area Book", book_id="area_book")

    assert (out_dir / "templates.jsonl").exists()
    assert (out_dir / "template_index.json").exists()
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert int(meta.get("template_count") or 0) >= 1

    # Ensure at least one template is executable after variable binding.
    first = json.loads((out_dir / "templates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first.get("rpn"), str) and first["rpn"]


def test_book_galaxy_library_search_templates(tmp_path: Path) -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
    from knowledge3d.training.math_benchmarks.book_galaxy_library import BookGalaxyLibrary

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps([{"page": 1, "content": "area = length * width"}], ensure_ascii=False),
        encoding="utf-8",
    )
    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Area Book", book_id="area_book")

    lib = BookGalaxyLibrary(books_root=out_dir.parent)
    wg = WordGalaxy()
    toks = [t.normalized for t in wg.tokenize("area length width") if getattr(t, "normalized", None)]
    hits = lib.search_templates(normalized_tokens=toks, top_k=5)
    assert hits, "Expected at least one template hit"
    assert hits[0].rpn


def test_trm_reader_emits_book_template_candidate(tmp_path: Path) -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps([{"page": 1, "content": "area = length * width"}], ensure_ascii=False),
        encoding="utf-8",
    )
    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Area Book", book_id="area_book")

    wg = WordGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=None,
        math_galaxy=None,
        rule_bank=[],
        enable_book_galaxies=True,
        book_galaxy_root=str(out_dir.parent),
        book_top_k=5,
    )

    candidates, meta, sources = reader._generate_book_galaxy_candidates("Find the area with length 3 and width 4.")
    assert any(c.strip() == "3 4 *" or c.strip().endswith("3 4 *") for c in candidates), candidates
    assert meta
    assert isinstance(sources, dict)


def test_trm_reader_book_artifact_condition_gate_prefers_right_triangle(tmp_path: Path) -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps(
            [
                {
                    "page": 1,
                    "content": (
                        "Theorem (Pythagorean Theorem)\n"
                        "Let triangle ABC be a right triangle with legs a and b and hypotenuse c.\n"
                        "a^2 + b^2 = c^2\n"
                    ),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Demo Book", book_id="demo_book", domain="geometry")

    wg = WordGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=None,
        math_galaxy=None,
        rule_bank=[],
        enable_book_galaxies=True,
        book_galaxy_root=str(out_dir.parent),
        book_top_k=10,
    )

    # Right triangle problem: should accept the artifact and yield the derived sqrt candidate.
    ok_candidates, _, _ = reader._generate_book_galaxy_candidates("A right triangle has legs 3 and 4. Find the hypotenuse.")
    assert any(c.strip() == "3 2 pow 4 2 pow + sqrt" for c in ok_candidates), ok_candidates

    # Non-right triangle mention: should reject the artifact based on its conditions.
    bad_candidates, _, _ = reader._generate_book_galaxy_candidates("Use the Pythagorean theorem on a triangle with sides 3, 4, 5.")
    assert not any("sqrt" in c for c in bad_candidates), bad_candidates


def test_trm_reader_binds_radius_and_height_semantically(tmp_path: Path) -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    src = tmp_path / "book.json"
    src.write_text(
        json.dumps(
            [
                {
                    "page": 1,
                    "content": (
                        "Theorem (Cylinder Volume)\n"
                        "Let a cylinder have radius r and height h.\n"
                        "volume = π * r^2 * h\n"
                    ),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(json_path=src, title="Cylinder Volume", book_id="cyl_volume", domain="geometry")

    wg = WordGalaxy()
    reader = TRMGalaxyReader(
        word_galaxy=wg,
        grammar_galaxy=None,
        math_galaxy=None,
        rule_bank=[],
        enable_book_galaxies=True,
        book_galaxy_root=str(out_dir.parent),
        book_top_k=10,
    )

    # Ensure semantic binding prefers radius/height proximity over numeric order.
    # If bound incorrectly, we'd often see pi 5 2 pow * 3 * for "radius 3 height 5".
    candidates, _, _ = reader._generate_book_galaxy_candidates("A cylinder has height 5 and radius 3. Find its volume.")
    assert any("π 3 2 pow" in c and "5" in c for c in candidates), candidates


def test_trm_reader_generates_multistep_circle_circumference_to_area_candidates() -> None:
    from knowledge3d.cranium.word_galaxy import WordGalaxy
    from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader

    reader = TRMGalaxyReader(
        word_galaxy=WordGalaxy(),
        grammar_galaxy=None,
        math_galaxy=None,
        rule_bank=[],
        enable_book_galaxies=False,
    )

    candidates = reader._generate_multistep_geometry_candidates("A circle has circumference 20. Find its area.", max_candidates=6)
    assert candidates
    assert any("20 2 pi * /" in c and "2 pow pi *" in c for c in candidates), candidates
