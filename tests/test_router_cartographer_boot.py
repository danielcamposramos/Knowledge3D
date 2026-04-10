from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.galaxy_manager import GalaxyManager
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_router_cartographer_stars_seed_at_boot(tmp_path: Path) -> None:
    kv = Knowledgeverse.__new__(Knowledgeverse)
    kv.galaxy_manager = GalaxyManager(storage_root=tmp_path / "kv_router_boot")
    kv._sas_symbol_values = []
    kv._sas_symbol_star_ids = []
    kv._sas_bootstrap_stars = []
    kv._router_cartographer_stars = []
    kv._gpu_reasoning_engine = None

    summary = Knowledgeverse._ensure_sas_bootstrap_loaded(kv)

    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    entries = list(getattr(grammar, "entries", []))
    by_id = {
        str(entry.get("id") or "").strip(): dict(entry)
        for entry in entries
        if isinstance(entry, dict)
    }

    for star_id in (
        "routing:task_type:math",
        "routing:task_type:question",
        "routing:task_type:spatial",
    ):
        assert star_id in by_id
        behavior_rpn = (
            ((by_id[star_id].get("metadata") or {}).get("meaning_star") or {}).get("behavior_rpn")
        )
        assert float(str(behavior_rpn or "").strip()) > 0.0
    assert int(summary["router_star_count"]) == 3
