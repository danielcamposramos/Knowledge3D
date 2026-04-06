from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_jarvis_entry_is_ensured_in_tool_galaxy(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    kv.ensure_default_galaxies_loaded()
    tool_entries = kv.galaxy_manager.get_galaxy("Tool").entries
    assert any(str(entry.get("id", "")) == "specialist_jarvis_coordinator" for entry in tool_entries)


def test_jarvis_compile_brief_tracks_workers_and_agreements(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    brief = kv._jarvis_compile_brief(
        task_type="ARC_TASK",
        paths=[{"program_id": "p1"}, {"program_id": "p2"}, {"program_id": "p3"}],
        options=None,
        path_best_records=[
            {
                "option_text": "answer_a",
                "path_role": "hypothesis",
                "path_score": 0.8,
                "candidate": {"path_score": 0.8, "match": {"id": "arc_a", "galaxy": "Drawing"}},
            },
            {
                "option_text": "answer_a",
                "path_role": "validation",
                "path_score": 0.7,
                "candidate": {"path_score": 0.7, "match": {"id": "arc_a", "galaxy": "Drawing"}},
            },
            {
                "option_text": "answer_b",
                "path_role": "hypothesis",
                "path_score": 0.4,
                "candidate": {"path_score": 0.4, "match": {"id": "arc_b", "galaxy": "Drawing"}},
            },
        ],
        selected_records=[
            {
                "option_text": "answer_a",
                "path_score": 0.8,
                "candidate": {"path_score": 0.8, "match": {"id": "arc_a", "galaxy": "Drawing"}},
            }
        ],
        scored_candidates=[
            {"gpu_score": 0.8},
            {"gpu_score": 0.7},
            {"gpu_score": 0.4},
        ],
    )

    assert brief["worker_count"] == 3
    assert brief["highest_confidence"] == "g1.w1"
    assert ("g1.w1", "g1.w2") in brief["agreements"]
    assert brief["planned_swarm_groups"] >= 1
    assert brief["active_swarm_groups"] == 1
    assert brief["swarm_groups"]["g1"] == ["g1.w1", "g1.w2", "g1.w3"]


def test_jarvis_compile_brief_tracks_multiple_swarm_groups(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    records = []
    for idx in range(12):
        records.append(
            {
                "option_text": f"answer_{idx % 2}",
                "path_role": "hypothesis" if idx % 2 == 0 else "validation",
                "path_score": 1.0 - (idx * 0.01),
                "candidate": {
                    "path_score": 1.0 - (idx * 0.01),
                    "match": {"id": f"arc_{idx}", "galaxy": "Drawing"},
                },
            }
        )
    brief = kv._jarvis_compile_brief(
        task_type="ARC_TASK",
        paths=[{"program_id": f"p{idx}"} for idx in range(12)],
        options=None,
        path_best_records=records,
        selected_records=records[:2],
        scored_candidates=[{"gpu_score": 1.0 - (idx * 0.01)} for idx in range(12)],
    )

    assert brief["worker_count"] == 12
    assert brief["active_swarm_groups"] == 2
    assert sorted(brief["swarm_groups"]) == ["g1", "g2"]
    assert brief["swarm_groups"]["g2"][0] == "g2.w1"


def test_jarvis_sleep_consolidation_summarizes_patterns(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    brief = kv._jarvis_compile_brief(
        task_type="MATH_TASK",
        paths=[{"program_id": "p1"}, {"program_id": "p2"}],
        options=["42"],
        path_best_records=[
            {
                "option_text": "42",
                "path_role": "hypothesis",
                "path_score": 0.9,
                "candidate": {"path_score": 0.9, "match": {"id": "math_a", "galaxy": "Math"}},
            },
            {
                "option_text": "42",
                "path_role": "validation",
                "path_score": 0.8,
                "candidate": {"path_score": 0.8, "match": {"id": "math_a", "galaxy": "Math"}},
            },
        ],
        selected_records=[
            {
                "option_text": "42",
                "path_score": 0.9,
                "candidate": {"path_score": 0.9, "match": {"id": "math_a", "galaxy": "Math"}},
            }
        ],
        scored_candidates=[{"gpu_score": 0.9}, {"gpu_score": 0.8}],
    )
    kv._jarvis_record_brief(brief)

    summary = kv.jarvis_sleep_consolidation()

    assert summary["updated"] is True
    assert summary["briefs_consolidated"] == 1
    assert summary["pending_briefs_before"] == 1
    assert summary["recommended_groups_by_task"]["MATH_TASK"] >= 1
    assert summary["top_worker_pairs"][0]["pair"] == "g1.w1|g1.w2"
    assert summary["top_cross_connections"][0]["pattern"] == "combine hypothesis and validation traces"
    assert summary["diagnostic"]["pending_recent_briefs"] == 0
    assert summary["diagnostic"]["brief_recording_active"] is True
    assert summary["checkpoint"]["galaxy_consolidated"]["saved"] is True
    assert (tmp_path / "checkpoints" / "galaxy_consolidated_latest.json").exists()


def test_jarvis_sleep_diagnostic_reports_pending_briefs_and_quality_memory(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    brief = kv._jarvis_compile_brief(
        task_type="ARC_TASK",
        paths=[{"program_id": "p1"}],
        options=None,
        path_best_records=[
            {
                "option_text": "route_a",
                "path_role": "hypothesis",
                "path_score": 0.7,
                "candidate": {"path_score": 0.7, "match": {"id": "arc_a", "galaxy": "Drawing"}},
            }
        ],
        selected_records=[
            {
                "option_text": "route_a",
                "path_score": 0.7,
                "candidate": {"path_score": 0.7, "match": {"id": "arc_a", "galaxy": "Drawing"}},
            }
        ],
        scored_candidates=[{"gpu_score": 0.7}],
    )
    kv._jarvis_record_brief(brief)
    kv.ternary_quality_memory.update(
        pattern_id="grammar_pathfind_to_target",
        outcome=1,
        confidence=0.9,
        knowledgeverse=kv,
        specialist="visual",
        galaxy="Grammar",
        source="test",
    )

    diagnostic = kv.jarvis_sleep_diagnostic()

    assert diagnostic["pending_recent_briefs"] == 1
    assert diagnostic["brief_recording_active"] is True
    assert diagnostic["last_brief_task_type"] == "ARC_TASK"
    assert diagnostic["last_brief_worker_count"] >= 1
    assert diagnostic["ternary_quality_pattern_count"] >= 1
    assert diagnostic["contrastive_learning_active"] is True


def test_jarvis_benchmark_sleep_writes_compact_delta(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    brief = kv._jarvis_compile_brief(
        task_type="MATH_TASK",
        paths=[{"program_id": "p1"}],
        options=["5"],
        path_best_records=[
            {
                "option_text": "5",
                "path_role": "hypothesis",
                "path_score": 0.8,
                "candidate": {"path_score": 0.8, "match": {"id": "math_answer", "galaxy": "Math"}},
            }
        ],
        selected_records=[
            {
                "option_text": "5",
                "path_score": 0.8,
                "candidate": {"path_score": 0.8, "match": {"id": "math_answer", "galaxy": "Math"}},
            }
        ],
        scored_candidates=[{"gpu_score": 0.8}],
    )
    kv._jarvis_record_brief(brief)

    summary = kv.jarvis_sleep_consolidation(persist=False, trigger="shutdown", profile="benchmark")

    assert summary["profile"] == "benchmark"
    assert summary["delta"]["saved"] is True
    assert summary["sovereign_sleep"]["gravity"]["skipped"] is True
    assert (tmp_path / "checkpoints" / "sovereign_sleep_delta.bin").exists()
    assert (tmp_path / "checkpoints" / "sovereign_sleep_delta.json").exists()
    assert summary["diagnostic"]["pending_recent_briefs"] == 0


def test_jarvis_sleep_delta_loads_on_boot(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)
    brief = kv._jarvis_compile_brief(
        task_type="QUESTION_TASK",
        paths=[{"program_id": "p1"}],
        options=["Paris"],
        path_best_records=[
            {
                "option_text": "Paris",
                "path_role": "hypothesis",
                "path_score": 0.7,
                "candidate": {"path_score": 0.7, "match": {"id": "capital_answer", "galaxy": "Reality"}},
            }
        ],
        selected_records=[
            {
                "option_text": "Paris",
                "path_score": 0.7,
                "candidate": {"path_score": 0.7, "match": {"id": "capital_answer", "galaxy": "Reality"}},
            }
        ],
        scored_candidates=[{"gpu_score": 0.7}],
    )
    kv._jarvis_record_brief(brief)
    kv.jarvis_sleep_consolidation(persist=False, trigger="shutdown", profile="benchmark")

    restored = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False)

    assert int(restored._jarvis_state.get("brief_count") or 0) >= 1
    assert "QUESTION_TASK" in dict(restored._jarvis_state.get("task_type_stats") or {})
    assert dict(restored._jarvis_state.get("last_brief") or {}).get("task_type") == "QUESTION_TASK"


def test_shutdown_is_idempotent_after_explicit_benchmark_shutdown(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path, eager_load_default_galaxies=False, start_live_loops=False)

    first = kv.shutdown(persist=False, profile="benchmark")
    second = kv.shutdown(persist=False, profile="service")

    assert first["status"] == "fast_exit"
    assert second["status"] == "idempotent_noop"
