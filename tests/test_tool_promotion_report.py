from __future__ import annotations

from pathlib import Path

from scripts.build_tool_promotion_report import build_report


def test_build_tool_promotion_report_ranks_primary_tools_and_targets(tmp_path):
    source_path = tmp_path / "tool_promotion_pressure.jsonl"
    event_source_path = tmp_path / "execution_events.jsonl"
    grammar_source_path = tmp_path / "execution_grammar_patterns.jsonl"
    quality_state_path = tmp_path / "execution_quality_tracker.json"
    rows = [
        {
            "timestamp": "2026-03-06T10:00:00Z",
            "query": "audio signal surface material fusion",
            "specialist": "audio",
            "source_galaxy": "Audio",
            "target_galaxy": "Reality",
            "primary_tool_id": "tool_fusion_signal_surface_material_v1",
            "tool_ids": [
                "tool_fusion_signal_surface_material_v1",
                "tool_signal_spectrogram_surface_v1",
            ],
            "tool_kinds": ["signal_surface_material_fusion", "signal_surface"],
            "runtime_statuses": ["ptx_bridge_available"],
            "codec_ops": ["MDCT", "TERNARY_QUANT"],
            "entrypoints": [
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface"
            ],
            "promotion_targets": ["UV_PROJECT", "TRIPLANAR_MAP"],
        },
        {
            "timestamp": "2026-03-06T10:01:00Z",
            "query": "contour textured mesh surface material projection",
            "specialist": "visual",
            "source_galaxy": "Drawing",
            "target_galaxy": "Reality",
            "primary_tool_id": "tool_fusion_surface_material_projection_v1",
            "tool_ids": [
                "tool_fusion_surface_material_projection_v1",
                "tool_geom_profile_lathe_mesh_v1",
            ],
            "tool_kinds": ["surface_material_projection", "lathe_mesh"],
            "runtime_statuses": ["ptx_bridge_available"],
            "codec_ops": ["TERNARY_QUANT"],
            "entrypoints": [
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_lathe_mesh"
            ],
            "promotion_targets": ["TRIPLANAR_MAP"],
        },
    ]
    event_rows = [
        {
            "tool_id": "tool_fusion_signal_surface_material_v1",
            "query_context": "audio signal surface material fusion",
            "specialist_id": "audio",
            "math_core_tier": 3,
            "execution_us": 4200,
            "outcome": 1,
            "quality_signal": 0.92,
            "ternary_quality": 1,
            "timestamp_us": 1,
            "chain_depth": 3,
            "promotion_pressure": True,
            "execution_mode": "tool_entrypoint_chain",
            "runtime_status": "ptx_bridge_available",
            "chain_tool_ids": [
                "tool_signal_audio_spectrogram_v1",
                "tool_signal_spectrogram_surface_v1",
                "tool_fusion_signal_surface_material_v1",
            ],
        },
        {
            "tool_id": "tool_fusion_surface_material_projection_v1",
            "query_context": "contour textured mesh surface material projection",
            "specialist_id": "visual",
            "math_core_tier": 2,
            "execution_us": 8100,
            "outcome": 0,
            "quality_signal": 0.46,
            "ternary_quality": 0,
            "timestamp_us": 2,
            "chain_depth": 2,
            "promotion_pressure": True,
            "execution_mode": "tool_entrypoint_chain",
            "runtime_status": "ptx_bridge_available",
            "chain_tool_ids": [
                "tool_geom_profile_lathe_mesh_v1",
                "tool_fusion_surface_material_projection_v1",
            ],
        },
        {
            "tool_id": "tool_fusion_surface_material_projection_v1",
            "query_context": "contour textured mesh surface material projection",
            "specialist_id": "visual",
            "math_core_tier": 2,
            "execution_us": 8400,
            "outcome": -1,
            "quality_signal": 0.28,
            "ternary_quality": -1,
            "timestamp_us": 3,
            "chain_depth": 2,
            "promotion_pressure": True,
            "execution_mode": "tool_entrypoint_chain",
            "runtime_status": "ptx_bridge_available",
            "chain_tool_ids": [
                "tool_geom_profile_lathe_mesh_v1",
                "tool_fusion_surface_material_projection_v1",
            ],
        },
    ]
    grammar_rows = [
        {
            "event": "execution_grammar_promoted",
            "rule_id": "exec_chain_deadbeef",
            "sequence": [
                "tool_geom_profile_lathe_mesh_v1",
                "tool_fusion_surface_material_projection_v1",
            ],
            "count": 3,
            "avg_quality_signal": 0.71,
        },
        {
            "event": "execution_grammar_promoted",
            "rule_id": "exec_chain_cafebabe",
            "sequence": [
                "tool_signal_audio_spectrogram_v1",
                "tool_signal_spectrogram_surface_v1",
                "tool_fusion_signal_surface_material_v1",
            ],
            "count": 4,
            "avg_quality_signal": 0.88,
        },
    ]
    quality_state = {
        "tools": {},
        "specialists": {},
        "route_sources": {
            "bridge": {
                "total_executions": 5,
                "bayesian_quality": 0.78,
                "recent_quality": [0.82, 0.74, 0.8],
                "ternary_trend": 1,
                "avg_execution_us": 5200.0,
            },
            "kernel": {
                "total_executions": 2,
                "bayesian_quality": 0.34,
                "recent_quality": [0.18, 0.22],
                "ternary_trend": -1,
                "avg_execution_us": 2900.0,
            },
        },
        "tool_sources": {
            "tool_fusion_signal_surface_material_v1::bridge": {
                "tool_id": "tool_fusion_signal_surface_material_v1",
                "route_source": "bridge",
                "total_executions": 3,
                "bayesian_quality": 0.8,
                "recent_quality": [0.9, 0.88, 0.86],
                "ternary_trend": 1,
                "avg_execution_us": 4200.0,
            },
            "tool_fusion_surface_material_projection_v1::bridge": {
                "tool_id": "tool_fusion_surface_material_projection_v1",
                "route_source": "bridge",
                "total_executions": 2,
                "bayesian_quality": 0.42,
                "recent_quality": [0.46, 0.28],
                "ternary_trend": -1,
                "avg_execution_us": 8250.0,
            },
        },
    }

    report = build_report(
        rows,
        source_path=source_path,
        event_rows=event_rows,
        event_source_path=event_source_path,
        grammar_rows=grammar_rows,
        grammar_source_path=grammar_source_path,
        quality_state=quality_state,
        quality_state_path=quality_state_path,
    )

    assert report["rows"] == 2
    assert report["event_rows"] == 3
    assert report["grammar_rows"] == 2
    assert report["stats"]["distinct_primary_tools"] == 2
    assert report["stats"]["distinct_event_tools"] == 2
    assert report["stats"]["distinct_route_sources"] == 2
    assert report["stats"]["distinct_tool_sources"] == 2
    assert report["rankings"]["promotion_targets"][0] == {
        "name": "TRIPLANAR_MAP",
        "count": 2,
    }
    assert report["rankings"]["primary_tools"][0]["name"] in {
        "tool_fusion_signal_surface_material_v1",
        "tool_fusion_surface_material_projection_v1",
    }
    assert any(
        item["name"] == "Audio->Reality"
        for item in report["rankings"]["source_target_routes"]
    )
    assert report["rankings"]["event_tools"][0]["name"] == "tool_fusion_surface_material_projection_v1"
    assert report["rankings"]["event_chains"][0]["name"] in {
        "tool_signal_audio_spectrogram_v1 -> tool_signal_spectrogram_surface_v1 -> tool_fusion_signal_surface_material_v1",
        "tool_geom_profile_lathe_mesh_v1 -> tool_fusion_surface_material_projection_v1",
    }
    top_candidate = report["candidate_summary"]["top_candidate"]
    assert top_candidate is not None
    assert top_candidate["name"] == "TRIPLANAR_MAP"
    assert top_candidate["pressure_count"] == 2
    assert top_candidate["grammar_occurrence_count"] >= 3
    assert top_candidate["promotion_priority_score"] > 0.0
    assert top_candidate["dominant_route_source"] == "bridge"
    assert top_candidate["source_quality_level"] > 0.0
    assert report["rankings"]["route_source_quality"][0]["name"] == "bridge"
    assert report["rankings"]["tool_source_quality"][0]["route_source"] == "bridge"
