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
