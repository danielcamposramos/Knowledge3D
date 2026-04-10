from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_boot_loads_sas_bootstrap_and_validates_required_galaxies(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_decomposer_boot")

    sas_summary = dict(kv._house_state_summary.get("sas_bootstrap") or {})
    assert int(sas_summary.get("symbol_count", 0)) >= 11
    assert int(sas_summary.get("rule_count", 0)) >= 7

    validation = dict(kv._boot_validation_summary or {})
    assert validation.get("valid") is True
    assert tuple(validation.get("required", ())) == kv.REQUIRED_BOOT_GALAXIES
    assert all(int(validation.get("counts", {}).get(name, 0)) > 0 for name in kv.REQUIRED_BOOT_GALAXIES)


def test_universal_decomposer_populates_store_registers_and_results(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_decomposer_state")

    for task_type, query_text in (
        ("ARC_TASK", "grid [[1,2],[3,4]] transform"),
        ("MATH_TASK", "What is 12 plus 7 minus 3?"),
        ("MMLU_TASK", "Which option best matches the concept?"),
    ):
        state = kv._run_universal_decomposer(task_type=task_type, query_text=query_text)
        assert set(state["registers"]) == {60, 61, 62}
        assert int(state["registers"][60]) == len(state["goal_programs"])
        assert int(state["registers"][61]) > 0
        assert int(state["registers"][62]) > 0
        assert len(state["goal_results"]) == len(state["goal_programs"])
        assert int(state["pool"].get("requested_slots", 0)) == len(state["goal_programs"])


def test_specialist_swarm_attaches_decomposer_registers_to_candidates(tmp_path, monkeypatch) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_decomposer_candidates")

    for method_name in (
        "get_vector_resonator",
        "get_galaxy_resonance_engine",
        "get_graph_crystallizer",
        "get_world_model",
        "get_resonance_field",
        "get_geometry_router",
        "get_temporal_reasoning",
        "get_fractal_emitter",
        "get_cognitive_executive",
        "get_atomic_fission_fusion",
    ):
        monkeypatch.setattr(kv, method_name, lambda: None)

    candidates = [
        {
            "match": {
                "id": "candidate_a",
                "galaxy": "Math",
                "embedding16": [0.1] * 16,
            },
            "led_focus": 0.0,
            "candidate_global_idx": 0,
            "graph_neighbors": [],
            "led_path_position": 0,
        }
    ]

    kv._apply_specialist_swarm_features(
        local_candidates=candidates,
        reference_embedding=[0.2] * 16,
        task_type="MATH_TASK",
        path={"query_text": "What is 3 plus 4?"},
        selection_steps=[],
    )

    candidate = candidates[0]
    assert candidate["specialist_worker_slots"] == list(range(9))
    assert set(candidate["swarm_store_registers"]) == {60, 61, 62}
    assert int(candidate["swarm_store_registers"][60]) >= 1
    assert int(candidate["specialist_decomposer_goals"]) == int(candidate["swarm_store_registers"][60])
    assert int(candidate["micro_slots_used"]) >= 1
