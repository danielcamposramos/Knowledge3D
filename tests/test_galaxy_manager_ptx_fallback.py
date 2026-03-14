from __future__ import annotations

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager


def test_query_uses_sovereign_token_matching_when_ptx_query_is_required(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "true")

    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    manager.add_entry(
        "Math",
        {
            "id": "math_word_total",
            "name": "Word Problem Total",
            "category": "word_problem_template",
            "rpn_program": "{rpn_chain}",
            "metadata": {"tool_kind": "math"},
        },
    )

    rows = manager.query(
        query_text="word problem total",
        specialist="math",
        top_k=5,
        galaxies=["Math"],
    )

    assert rows
    assert rows[0]["entry"]["id"] == "math_word_total"


def test_bulk_disk_sync_batches_rewrites(tmp_path) -> None:
    manager = GalaxyManager(storage_root=tmp_path / "galaxies")
    rewrite_calls: list[str] = []
    append_calls: list[str] = []

    manager._rewrite_galaxy_disk = lambda galaxy_name, galaxy: rewrite_calls.append(str(galaxy_name))  # type: ignore[method-assign]
    manager._append_entry_to_disk = lambda galaxy_name, entry: append_calls.append(str(galaxy_name))  # type: ignore[method-assign]

    with manager.bulk_disk_sync():
        manager.add_entry("Grammar", {"id": "grammar_a"})
        manager.upsert_entry("Grammar", {"id": "grammar_a", "content": "updated"})
        manager.add_entry("Math", {"id": "math_a"})

    assert append_calls == []
    assert sorted(rewrite_calls) == ["Grammar", "Math"]
