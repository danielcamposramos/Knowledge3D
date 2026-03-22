from __future__ import annotations

from collections import Counter

from scripts.ingest_arc_knowledge import build_arc_anchor_catalog, build_arc_language_symlink_entries


def test_arc_catalog_has_expected_scale_and_group_balance():
    specs = build_arc_anchor_catalog()
    counts = Counter(str(spec.get("family_group", "")) for spec in specs)

    assert len(specs) >= 330
    assert counts["object_centric"] >= 70
    assert counts["geometric_transform"] >= 50
    assert counts["color_pattern"] >= 50
    assert counts["symbolic_interpretation"] >= 50
    assert counts["spatial_reasoning"] >= 50
    assert counts["meta_reasoning"] >= 35
    assert counts["interactive_strategy"] >= 25


def test_arc_language_bridges_cover_all_arc_anchors():
    specs = build_arc_anchor_catalog()
    bridges = build_arc_language_symlink_entries()

    target_ids = {str(spec["id"]) for spec in specs}
    bridge_targets = {str(entry.get("symlink_to") or "") for entry in bridges}

    assert len(bridges) == len(specs)
    assert bridge_targets == target_ids
    assert len(specs) + len(bridges) >= 660
