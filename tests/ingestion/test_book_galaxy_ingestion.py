import json

import numpy as np

from knowledge3d.training.math_benchmarks.book_galaxy_ingestion import BookGalaxyIngester


def test_book_galaxy_ingester_writes_expected_artifacts(tmp_path):
    # Minimal "page dump" similar to the existing Advanced Maths JSON exports.
    pages = [
        {"page": 1, "content": "A linear transformation is a map. 2 + 2 = 4."},
        {"page": 2, "content": "det([[1,2],[3,4]]) = 1*4 - 2*3 = -2."},
    ]
    src = tmp_path / "book.json"
    src.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    ingester = BookGalaxyIngester(local_dir=tmp_path)
    out_dir = ingester.ingest_json_pages(
        json_path=src,
        title="Linear Algebra Done Right",
        author="Sheldon Axler",
        domain="linear_algebra",
    )

    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "pages.jsonl").exists()
    assert (out_dir / "embeddings_128.npy").exists()
    assert (out_dir / "positions_3d.npy").exists()
    assert (out_dir / "token_index.json").exists()

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["book_id"] == "linear_algebra_done_right"
    assert meta["page_count"] == 2
    assert meta["domain"] == "linear_algebra"

    pages_jsonl = (out_dir / "pages.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(pages_jsonl) == 2
    first = json.loads(pages_jsonl[0])
    assert first["page_number"] == 1
    assert first["token_count"] > 0
    assert isinstance(first["position_3d"], list) and len(first["position_3d"]) == 3
    assert any("char_sequence" in t and t["char_sequence"] for t in first["tokens"])

    emb = np.load(out_dir / "embeddings_128.npy")
    pos = np.load(out_dir / "positions_3d.npy")
    assert emb.shape == (2, 128)
    assert pos.shape == (2, 3)


def test_book_galaxy_ingester_rejects_empty_text(tmp_path):
    src = tmp_path / "empty.json"
    src.write_text(json.dumps([{"page": 1, "content": "   "}]), encoding="utf-8")

    ingester = BookGalaxyIngester(local_dir=tmp_path)
    try:
        ingester.ingest_json_pages(json_path=src, title="Empty Book")
        assert False, "Expected ValueError for empty pages"
    except ValueError as exc:
        assert "No text found in pages" in str(exc)


def test_book_galaxy_ingester_bulk_dir(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "Book.One.json").write_text(json.dumps([{"page": 1, "content": "Hello world"}]), encoding="utf-8")
    (src_dir / "Book.Two.json").write_text(json.dumps([{"page": 1, "content": "Another page"}]), encoding="utf-8")

    ingester = BookGalaxyIngester(local_dir=tmp_path)
    outs = ingester.ingest_json_dir(json_dir=src_dir, domain="test_domain")
    assert len(outs) == 2
    assert all((p / "metadata.json").exists() for p in outs)


def test_book_galaxy_ingester_caps_template_index_ids_per_token(tmp_path):
    # Force many distinct templates sharing the same LHS token to stress
    # the template index sets (CPU OOM regression guard).
    lines = [f"common = {i}" for i in range(1, 80)]
    pages = [{"page": 1, "content": "\n".join(lines)}]
    src = tmp_path / "book.json"
    src.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    ingester = BookGalaxyIngester(
        local_dir=tmp_path,
        build_token_index=False,
        template_index_min_token_len=1,
        max_template_index_keys=10_000,
        max_templates_per_token=3,
        artifact_index_min_token_len=1,
        max_artifacts_per_token=3,
    )
    out_dir = ingester.ingest_json_pages(json_path=src, title="Index Caps")

    idx_path = out_dir / "template_index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    assert "common" in idx
    assert len(idx["common"]) <= 3

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["template_index_truncated_template_ids"] > 0


def test_book_galaxy_ingester_caps_artifact_index_ids_per_token(tmp_path):
    # Stress artifact indexing (CPU OOM regression guard) by creating many
    # theorem blocks that all reference the same variable token.
    blocks = []
    for i in range(60):
        blocks.append(f"Theorem (T{i})\na = {i} + 1\n")
    pages = [{"page": 1, "content": "\n".join(blocks)}]
    src = tmp_path / "book.json"
    src.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    ingester = BookGalaxyIngester(
        local_dir=tmp_path,
        build_token_index=False,
        artifact_index_min_token_len=1,
        max_artifact_index_keys=10_000,
        max_artifacts_per_token=2,
        template_index_min_token_len=1,
        max_templates_per_token=2,
    )
    out_dir = ingester.ingest_json_pages(json_path=src, title="Artifact Index Caps")

    idx_path = out_dir / "artifact_index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    # The token "a" should be indexed from lhs/rhs and capped.
    assert "a" in idx
    assert len(idx["a"]) <= 2

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["artifact_index_truncated_artifact_ids"] > 0


def test_book_galaxy_ingester_disables_indices_without_deleting_content(tmp_path):
    # Use an equation that results in an executable RPN candidate (has ops).
    pages = [{"page": 1, "content": "common = 1\nTheorem (T)\na = 2 + 1\n"}]
    src = tmp_path / "book.json"
    src.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")

    ingester = BookGalaxyIngester(
        local_dir=tmp_path,
        build_token_index=False,
        build_template_index=False,
        build_artifact_index=False,
    )
    out_dir = ingester.ingest_json_pages(json_path=src, title="Disable Indices")

    assert (out_dir / "templates.jsonl").exists()
    assert not (out_dir / "template_index.json").exists()
    assert (out_dir / "artifacts.jsonl").exists()
    assert not (out_dir / "artifact_index.json").exists()
