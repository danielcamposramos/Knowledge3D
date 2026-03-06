from __future__ import annotations

import json

from knowledge3d.knowledgeverse import Knowledgeverse


def test_tool_quality_tracker_updates_and_biases_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_specialist_quality")

    kv.trm_navigator.observe_execution_event(
        {
            "tool_id": "tool_alpha",
            "query_context": "generic scene tool route",
            "specialist_id": "VisualSpecialist",
            "math_core_tier": 2,
            "execution_us": 1000,
            "outcome": 1,
            "quality_signal": 0.9,
            "ternary_quality": 1,
            "timestamp_us": 1,
            "chain_depth": 1,
            "promotion_pressure": False,
        }
    )
    kv.trm_navigator.observe_execution_event(
        {
            "tool_id": "tool_beta",
            "query_context": "generic scene tool route",
            "specialist_id": "VisualSpecialist",
            "math_core_tier": 2,
            "execution_us": 1000,
            "outcome": -1,
            "quality_signal": 0.1,
            "ternary_quality": -1,
            "timestamp_us": 2,
            "chain_depth": 1,
            "promotion_pressure": False,
        }
    )

    rows = [
        {"tool_id": "tool_beta", "tool_kind": "scene_fusion", "runtime_status": "ptx_bridge_available"},
        {"tool_id": "tool_alpha", "tool_kind": "scene_fusion", "runtime_status": "ptx_bridge_available"},
    ]
    ranked = kv.trm_navigator._prioritize_executable_tools(rows, query_text="scene layered composition")

    assert ranked[0]["tool_id"] == "tool_alpha"
    record = kv.trm_navigator.execution_quality_tracker.get_tool_record("tool_alpha")
    assert record is not None
    assert float(record["bayesian_quality"]) > 0.5
    source_record = kv.trm_navigator.execution_quality_tracker.get_tool_source_record(
        "tool_alpha",
        runtime_status="",
    )
    assert source_record is not None
    assert source_record["route_source"] == "recipe"


def test_specialist_learning_updates_centroid_and_logs_gap(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_specialist_gap")

    summary = kv.trm_navigator.observe_execution_event(
        {
            "tool_id": "tool_gap_probe",
            "query_context": "quux zorb flux lattice entelechy",
            "specialist_id": "GrammarSpecialist",
            "math_core_tier": 1,
            "execution_us": 500,
            "outcome": 1,
            "quality_signal": 0.82,
            "ternary_quality": 1,
            "timestamp_us": 10,
            "chain_depth": 1,
            "promotion_pressure": False,
        }
    )

    state_path = tmp_path / "kv_specialist_gap" / "checkpoints" / "execution_quality_tracker.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    grammar = state["specialists"]["GrammarSpecialist"]
    assert int(grammar["update_count"]) >= 1

    gap_log = tmp_path / "kv_specialist_gap" / "logs" / "specialist_gaps.jsonl"
    assert gap_log.exists()
    rows = [json.loads(line) for line in gap_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert bool(summary["quality"]["gap_logged"]) is True
    assert summary["quality"]["routing_gate"]["preferred_route"] in {"bridge", "kernel", "recipe"}


def test_positive_kernel_history_produces_kernel_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_specialist_kernel_gate")

    for idx in range(3):
        summary = kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_kernel_good",
                "query_context": "signal fusion kernel route",
                "specialist_id": "AudioSpecialist",
                "math_core_tier": 3,
                "execution_us": 900 + idx,
                "outcome": 1,
                "quality_signal": 0.93,
                "ternary_quality": 1,
                "timestamp_us": 100 + idx,
                "chain_depth": 1,
                "promotion_pressure": True,
                "runtime_status": "ptx_rpn_available",
            }
        )

    gate = summary["quality"]["routing_gate"]
    assert summary["quality"]["route_source"] == "kernel"
    assert gate["preferred_route"] == "kernel"
    assert gate["ternary_gate"] == 1


def test_routing_gate_demotes_unhealthy_kernel_below_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_specialist_gate_rank")

    for idx in range(3):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_kernel_bad",
                "query_context": "scene composition route",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 3,
                "execution_us": 1500 + idx,
                "outcome": -1,
                "quality_signal": 0.12,
                "ternary_quality": -1,
                "timestamp_us": 200 + idx,
                "chain_depth": 1,
                "promotion_pressure": True,
                "runtime_status": "ptx_rpn_available",
            }
        )
    for idx in range(2):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_bridge_ok",
                "query_context": "scene composition route",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 2,
                "execution_us": 1800 + idx,
                "outcome": 0,
                "quality_signal": 0.58,
                "ternary_quality": 0,
                "timestamp_us": 300 + idx,
                "chain_depth": 1,
                "promotion_pressure": False,
                "runtime_status": "ptx_bridge_available",
            }
        )

    rows = [
        {"tool_id": "tool_kernel_bad", "tool_kind": "scene_fusion", "runtime_status": "ptx_rpn_available"},
        {"tool_id": "tool_bridge_ok", "tool_kind": "scene_fusion", "runtime_status": "ptx_bridge_available"},
    ]
    ranked = kv.trm_navigator._prioritize_executable_tools(rows, query_text="scene layered composition")

    assert ranked[0]["tool_id"] == "tool_bridge_ok"
    kernel_gate = kv.trm_navigator.execution_quality_tracker.routing_gate(
        "tool_kernel_bad",
        runtime_status="ptx_rpn_available",
        tool_kind="scene_fusion",
    )
    bridge_gate = kv.trm_navigator.execution_quality_tracker.routing_gate(
        "tool_bridge_ok",
        runtime_status="ptx_bridge_available",
        tool_kind="scene_fusion",
    )
    assert kernel_gate["preferred_route"] == "recipe"
    assert kernel_gate["ternary_gate"] == -1
    assert bridge_gate["preferred_route"] == "bridge"


def test_chain_step_event_skips_specialist_gap_and_grammar(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_chain_step_light")

    summary = kv.trm_navigator.observe_execution_event(
        {
            "tool_id": "tool_chain_probe",
            "query_context": "quux zorb flux lattice entelechy",
            "specialist_id": "GrammarSpecialist",
            "math_core_tier": 2,
            "execution_us": 700,
            "outcome": 1,
            "quality_signal": 0.77,
            "ternary_quality": 1,
            "timestamp_us": 99,
            "chain_depth": 3,
            "promotion_pressure": False,
            "runtime_status": "ptx_bridge_available",
            "tool_kind": "scene_fusion",
            "execution_mode": "tool_chain_step",
        }
    )

    record = kv.trm_navigator.execution_quality_tracker.get_tool_record("tool_chain_probe")
    assert record is not None
    assert int(record["total_executions"]) == 1
    assert summary["quality"]["gap_logged"] is False
    assert summary.get("grammar", {}) == {}

    gap_log = tmp_path / "kv_chain_step_light" / "logs" / "specialist_gaps.jsonl"
    assert not gap_log.exists()
