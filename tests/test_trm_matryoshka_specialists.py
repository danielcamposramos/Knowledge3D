from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_trm_navigator_bootstraps_matryoshka_specialists(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_matryoshka_bootstrap")
    trm = kv.trm_navigator

    assert trm.count_specialists() >= 12
    assert trm.find_specialist("MathSpecialist") is not None
    assert trm.find_specialist("VisualSpecialist") is not None
    assert trm.find_specialist("PhysicsSpecialist") is not None
    assert trm.find_specialist("GrammarSpecialist") is not None


def test_route_includes_matryoshka_metadata(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_matryoshka_route")
    route = kv.trm_navigator.route(
        query="find derivative of x^2 at x=3",
        specialist="auto",
        domain_hint="math",
    )

    assert route["specialist"] == "math"
    assert route["matryoshka_specialist"]
    assert route["matryoshka_domain"]
    assert route["matryoshka_level"] >= 1


def test_spawner_persists_across_restarts(tmp_path):
    storage_root = tmp_path / "kv_matryoshka_persist"
    kv = Knowledgeverse(storage_root=storage_root)
    trm = kv.trm_navigator

    for _ in range(20):
        trm.learn_from_feedback(
            query="topology manifold homology proof",
            specialist="math",
            success=False,
            confidence=0.2,
            domain_hint="topology",
        )
    trm.save_weights()

    kv_reloaded = Knowledgeverse(storage_root=storage_root)
    trm_reloaded = kv_reloaded.trm_navigator
    assert trm_reloaded.count_specialists() >= 12
    assert trm_reloaded.specialist_spawner.decisions


def test_auto_feedback_persistence_is_debounced_but_explicit_save_still_forces_write(tmp_path):
    storage_root = tmp_path / "kv_matryoshka_debounce"
    kv = Knowledgeverse(storage_root=storage_root)
    trm = kv.trm_navigator

    checkpoints = storage_root / "checkpoints"
    weight_path = trm.navigator_specialist.weight_store.path
    spawner_path = checkpoints / "trm_specialist_spawner.json"
    tree_path = checkpoints / "trm_specialist_tree.json"

    trm.observe_execution_event(
        {
            "execution_mode": "tool_entrypoint_chain",
            "outcome": 1,
            "quality_signal": 0.9,
            "specialist_id": "AudioSpecialist",
            "query_context": "world ambient audio animation",
            "domain_hint": "audio",
            "tool_id": "tool_fusion_signal_surface_material_world_animation_v1",
        }
    )

    first_mtimes = (
        weight_path.stat().st_mtime_ns,
        spawner_path.stat().st_mtime_ns,
        tree_path.stat().st_mtime_ns,
    )

    trm.observe_execution_event(
        {
            "execution_mode": "tool_entrypoint_chain",
            "outcome": 1,
            "quality_signal": 0.9,
            "specialist_id": "AudioSpecialist",
            "query_context": "world ambient audio animation",
            "domain_hint": "audio",
            "tool_id": "tool_fusion_signal_surface_material_world_animation_v1",
        }
    )

    second_mtimes = (
        weight_path.stat().st_mtime_ns,
        spawner_path.stat().st_mtime_ns,
        tree_path.stat().st_mtime_ns,
    )
    assert second_mtimes == first_mtimes

    trm.save_weights()

    third_mtimes = (
        weight_path.stat().st_mtime_ns,
        spawner_path.stat().st_mtime_ns,
        tree_path.stat().st_mtime_ns,
    )
    assert third_mtimes >= second_mtimes
