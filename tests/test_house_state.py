from __future__ import annotations

import re

from scripts.run_enriched_benchmarks import _incremental_knowledge_update
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
import pickle


def _prepare_build_feed(kv: Knowledgeverse) -> dict[str, object]:
    kv.ensure_default_galaxies_loaded()
    runtime = kv._get_sovereign_hot_path()
    runtime.refresh_feed_source()
    return runtime.refresh_build_feed()


def _seed_build_feed(storage_root) -> dict[str, object]:
    seed = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    try:
        return _prepare_build_feed(seed)
    finally:
        runtime = getattr(seed, "_sovereign_hot_path", None)
        if runtime is not None:
            runtime.close()
            seed._sovereign_hot_path = None


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
    _prepare_build_feed(first)

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
    _prepare_build_feed(first)

    with (storage_root / "house" / "galaxy_state.bin").open("rb") as handle:
        payload = pickle.load(handle)
    assert "Book/MathematicsPrimer" in payload["galaxies"]
    assert "Book_MathematicsPrimer" not in payload["galaxies"]

    second = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    assert second.load_house_state() is True
    restored = second.galaxy_manager.get_galaxy("Book/MathematicsPrimer").entries
    assert any(entry.get("id") == "unit_book_entry" for entry in restored)


def test_consolidated_checkpoint_warm_boot_restores_latest_state(tmp_path) -> None:
    storage_root = tmp_path / "kv_consolidated_state"
    first = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    first.galaxy_manager.add_entry(
        "Math",
        {
            "id": "unit_math_checkpoint_entry",
            "domain": "math",
            "category": "unit_test",
            "value": 11,
        },
    )

    summary = first.save_consolidated_state()

    latest_path = storage_root / "checkpoints" / "galaxy_consolidated_latest.json"
    assert latest_path.exists()
    assert summary["galaxy_consolidated"]["saved"] is True

    for path in (storage_root / "galaxies").glob("*.jsonl"):
        path.unlink()

    second = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    restored = second.galaxy_manager.get_galaxy("Math").entries
    assert any(entry.get("id") == "unit_math_checkpoint_entry" for entry in restored)
    assert second.house_state_summary()["warm_boot"] is True
    assert "galaxy_consolidated_latest.json" in second.house_state_summary()["path"]


def test_eager_boot_records_runtime_stage_summary(tmp_path) -> None:
    storage_root = tmp_path / "kv_boot_runtime"
    _seed_build_feed(storage_root)
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=True,
        start_live_loops=False,
    )

    summary = kv.boot_runtime_summary()
    stages = {str(item.get("stage")): dict(item) for item in list(summary.get("stages") or [])}

    assert summary["live_ready"] is True
    for stage_name in (
        "house_restore",
        "jarvis_restore",
        "trm_launcher_init",
        "default_galaxy_load",
        "trm_decoder_load",
        "sovereign_runtime_load",
    ):
        assert stage_name in stages
        assert float(stages[stage_name].get("elapsed_s", 0.0)) >= 0.0

    sovereign_summary = dict(stages["sovereign_runtime_load"].get("summary") or {})
    assert sovereign_summary["mode"] in {"artifact", "rebuilt", "resident"}
    assert "default_galaxy_load_s" in sovereign_summary
    assert "total_elapsed_s" in sovereign_summary
    assert "boot_runtime" in kv.house_state_summary()


def test_resident_sovereign_runtime_skips_reloading_default_galaxies(tmp_path, monkeypatch) -> None:
    storage_root = tmp_path / "kv_runtime_resident"
    _seed_build_feed(storage_root)
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    calls = 0
    original = kv.ensure_default_galaxies_loaded

    def _counted_ensure_default_galaxies_loaded(*, force: bool = False):
        nonlocal calls
        calls += 1
        return original(force=force)

    monkeypatch.setattr(kv, "ensure_default_galaxies_loaded", _counted_ensure_default_galaxies_loaded)

    first = kv._boot_sovereign_runtime(force_reload=True)
    second = kv._get_sovereign_hot_path().ensure_loaded()

    assert first["mode"] in {"artifact", "rebuilt", "resident"}
    assert second["mode"] == "resident"
    assert second["default_galaxy_load_skipped"] is True
    assert calls == 1


def test_gpu_flat_cache_retains_only_latest_signature_pair(tmp_path) -> None:
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_gpu_cache_retention",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )

    stride = int(kv.GPU_GALAXY_ENTRY_STRIDE)
    catalog = [{"id": "unit_cache_entry"}]
    kv._save_gpu_flat_cache(signature="aaaa1111", flat_entries=[0.0] * stride, catalog=catalog)
    summary = kv._save_gpu_flat_cache(signature="bbbb2222", flat_entries=[1.0] * stride, catalog=catalog)

    cache_dir = kv._gpu_cache_dir()
    remaining = sorted(path.name for path in cache_dir.iterdir())

    assert remaining == ["catalog_bbbb2222.pkl", "flat_bbbb2222.npy"]
    assert summary["removed_count"] == 2


def test_checkpoint_retention_prunes_heavy_timestamped_families(tmp_path) -> None:
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_checkpoint_retention",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    checkpoint_dir = kv._checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stamps = [
        "20260402_120000",
        "20260402_120100",
        "20260402_120200",
        "20260402_120300",
        "20260402_120400",
    ]
    families = (
        ("galaxy_consolidated", ".json"),
        ("sovereign_runtime_bundle", ".pkl"),
        ("trm_weights", ".npz"),
        ("shadow_patterns", ".json"),
    )
    for prefix, suffix in families:
        for stamp in stamps:
            path = checkpoint_dir / f"{prefix}_{stamp}{suffix}"
            if suffix == ".pkl":
                path.write_bytes(b"pickle")
            elif suffix == ".npz":
                path.write_bytes(b"npz")
            else:
                path.write_text("{}", encoding="utf-8")

    # Non-timestamped canonical pointers must survive rotation untouched.
    (checkpoint_dir / "galaxy_consolidated_latest.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "sovereign_runtime_bundle.pkl").write_bytes(b"bundle")
    (checkpoint_dir / "trm_weights.npz").write_bytes(b"weights")
    (checkpoint_dir / "shadow_patterns_latest.json").write_text("{}", encoding="utf-8")

    summary = kv._apply_checkpoint_retention()

    def _count(prefix: str, suffix: str) -> int:
        pattern = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}{re.escape(suffix)}$")
        return sum(1 for path in checkpoint_dir.iterdir() if pattern.fullmatch(path.name))

    assert _count("galaxy_consolidated", ".json") == 2
    assert _count("sovereign_runtime_bundle", ".pkl") == 2
    assert _count("trm_weights", ".npz") == 3
    assert _count("shadow_patterns", ".json") == 3
    assert summary["removed_count"] == 10
    assert (checkpoint_dir / "galaxy_consolidated_latest.json").exists()
    assert (checkpoint_dir / "sovereign_runtime_bundle.pkl").exists()
    assert (checkpoint_dir / "trm_weights.npz").exists()
    assert (checkpoint_dir / "shadow_patterns_latest.json").exists()
