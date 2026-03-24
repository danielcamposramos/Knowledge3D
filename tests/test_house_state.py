from __future__ import annotations

from scripts.run_enriched_benchmarks import _incremental_knowledge_update
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
import pickle


def test_knowledgeverse_house_state_round_trip(tmp_path) -> None:
    storage_root = tmp_path / "kv_house_state"

    first = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    first.galaxy_manager.add_entry(
        "Math",
        {
            "id": "unit_math_entry",
            "domain": "math",
            "category": "unit_test",
            "value": 7,
        },
    )
    saved = first.save_house_state()

    assert saved["total_persisted_entries"] >= 1
    for path in (storage_root / "galaxies").glob("*.jsonl"):
        path.unlink()

    second = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)

    assert second.load_house_state() is True
    restored = second.galaxy_manager.get_galaxy("Math").entries
    assert any(entry.get("id") == "unit_math_entry" for entry in restored)
    assert second.house_state_summary()["warm_boot"] is True


def test_incremental_knowledge_update_adds_only_new_entries(tmp_path, monkeypatch) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_incremental", eager_load_default_galaxies=False)
    kv.ensure_default_galaxies_loaded()
    kv.galaxy_manager.add_entry(
        "Drawing",
        {"id": "arc_anchor_existing", "domain": "drawing", "category": "arc_transform_primitive"},
    )
    kv.galaxy_manager.add_entry(
        "Math",
        {"id": "math_formula_existing", "domain": "math", "category": "formula_fact"},
    )

    monkeypatch.setattr(
        "scripts.run_enriched_benchmarks.build_all_arc_entries",
        lambda: {
            "Drawing": [
                {"id": "arc_anchor_existing", "domain": "drawing", "category": "arc_transform_primitive"},
                {"id": "arc_anchor_new", "domain": "drawing", "category": "arc_transform_primitive"},
            ],
            "Language": [
                {"id": "lang_arc_symlink_new", "domain": "language", "category": "meaning_symlink"},
            ],
        },
    )
    monkeypatch.setattr(
        "scripts.run_enriched_benchmarks.build_math_rule_entries",
        lambda: [
            {"id": "math_formula_existing", "domain": "math", "category": "formula_fact"},
            {"id": "math_formula_new", "domain": "math", "category": "formula_fact"},
        ],
    )

    summary = _incremental_knowledge_update(kv)

    assert summary["added"] == 3
    assert summary["skipped"] == 2
    assert summary["counts"] == {"Drawing": 1, "Language": 1, "Math": 1}


def test_house_state_preserves_book_galaxy_logical_names(tmp_path) -> None:
    storage_root = tmp_path / "kv_book_state"
    first = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    first.galaxy_manager.add_entry(
        "Book/MathematicsPrimer",
        {
            "id": "unit_book_entry",
            "domain": "Book/MathematicsPrimer",
            "category": "unit_test",
            "content": "book star",
        },
    )
    first.save_house_state()

    with (storage_root / "house" / "galaxy_state.bin").open("rb") as handle:
        payload = pickle.load(handle)
    assert "Book/MathematicsPrimer" in payload["galaxies"]
    assert "Book_MathematicsPrimer" not in payload["galaxies"]

    second = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    assert second.load_house_state() is True
    restored = second.galaxy_manager.get_galaxy("Book/MathematicsPrimer").entries
    assert any(entry.get("id") == "unit_book_entry" for entry in restored)
