from __future__ import annotations

import json

import pytest

from knowledge3d.cranium.bridges.procedural_material_bridge import SurfaceMaterialCandidate
from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.knowledgeverse import Knowledgeverse


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_successful_recurring_tool_chains_create_grammar_entries(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_grammar")

    cool = SurfaceMaterialCandidate(
        material_id="cool",
        name="Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )

    for idx in range(3):
        kv.trm_navigator.execute(
            program,
            input_data={
                "clip_id": f"signal_exec_grammar_{idx}",
                "audio_signal": TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
                "material_candidates": (cool, warm),
                "negative_materials": (warm,),
                "frame_size": 256,
                "threshold": 0.15,
                "displacement_gain": 0.4,
                "preview_size": 32,
                },
            )

    successful_chain = [
        "tool_signal_audio_spectrogram_v1",
        "tool_signal_spectrogram_surface_v1",
        "tool_fusion_surface_material_projection_v1",
    ]
    for idx in range(3):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_fusion_signal_surface_material_v1",
                "query_context": "audio signal surface material fusion",
                "specialist_id": "AudioSpecialist",
                "math_core_tier": 3,
                "execution_us": 1000,
                "outcome": 1,
                "quality_signal": 0.92,
                "ternary_quality": 1,
                "timestamp_us": 100 + idx,
                "chain_depth": len(successful_chain),
                "promotion_pressure": True,
                "chain_tool_ids": successful_chain,
                "chain_runtime_statuses": ["ptx_bridge_available"] * len(successful_chain),
            }
        )

    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    auto_rules = [
        entry for entry in grammar.entries
        if isinstance(entry, dict)
        and str(entry.get("language", "")) == "execution"
        and str((entry.get("semantics") or {}).get("source", "")) == "auto_detected"
    ]

    assert auto_rules
    target = auto_rules[0]
    assert target["semantics"]["occurrence_count"] >= 3
    assert target["semantics"]["chain_tool_ids"]
    assert target["semantics"]["ternary_confidence"] in (-1, 0, 1)

    event_path = tmp_path / "kv_exec_grammar" / "logs" / "execution_events.jsonl"
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    top_events = [row for row in rows if row.get("execution_mode") != "tool_chain_step"]
    assert top_events
    assert any(len(row.get("chain_tool_ids", [])) >= 2 for row in top_events)


@pytest.mark.cuda
def test_failing_recurring_tool_chains_create_contrastive_grammar_entries(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_antigrammar")

    failing_chain = [
        "tool_geom_profile_lathe_mesh_v1",
        "tool_fusion_surface_material_projection_v1",
    ]
    for idx in range(3):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_fusion_surface_material_projection_v1",
                "query_context": "contour textured mesh surface material projection failure case",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 2,
                "execution_us": 1400 + idx,
                "outcome": -1,
                "quality_signal": 0.18,
                "ternary_quality": -1,
                "timestamp_us": 500 + idx,
                "chain_depth": len(failing_chain),
                "promotion_pressure": True,
                "chain_tool_ids": failing_chain,
                "chain_runtime_statuses": ["ptx_bridge_available"] * len(failing_chain),
            }
        )

    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    anti_rules = [
        entry for entry in grammar.entries
        if isinstance(entry, dict)
        and str(entry.get("language", "")) == "execution"
        and str((entry.get("semantics") or {}).get("source", "")) == "auto_detected_contrastive"
    ]

    assert anti_rules
    target = anti_rules[0]
    assert target["pattern"] == "tool_chain_negative"
    assert target["semantics"]["pattern_type"] == "execution_tool_chain_antipattern"
    assert target["semantics"]["failure_count"] >= 3
    assert target["semantics"]["contrastive_recommendation"] == "avoid_or_invert"
    assert "outcome:-1" in target["usage_conditions"]


@pytest.mark.cuda
def test_successful_multimodal_patterns_create_generalized_grammar_entries(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_multimodal_grammar")

    chain = [
        "tool_signal_audio_spectrogram_v1",
        "tool_signal_spectrogram_surface_v1",
        "tool_fusion_surface_material_projection_v1",
    ]
    queries = [
        "audio signal fusion preview for spectral learning",
        "signal audio fusion preview for texture learning",
        "audio fusion signal preview for layered learning",
    ]
    for idx, query in enumerate(queries):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_fusion_signal_surface_material_v1",
                "query_context": query,
                "specialist_id": "AudioSpecialist",
                "domain_hint": "multimodal",
                "math_core_tier": 3,
                "execution_us": 1200 + idx,
                "outcome": 1,
                "quality_signal": 0.9,
                "ternary_quality": 1,
                "timestamp_us": 1200 + idx,
                "chain_depth": len(chain),
                "promotion_pressure": True,
                "chain_tool_ids": chain,
                "chain_runtime_statuses": ["ptx_bridge_available"] * len(chain),
            }
        )

    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    multimodal_rules = [
        entry for entry in grammar.entries
        if isinstance(entry, dict)
        and str(entry.get("language", "")) == "execution"
        and str((entry.get("semantics") or {}).get("source", "")) == "auto_detected_multimodal"
    ]

    assert multimodal_rules
    target = multimodal_rules[0]
    semantics = target["semantics"]
    assert target["pattern"] == "multimodal_execution_positive"
    assert semantics["pattern_type"] == "execution_multimodal_pattern"
    assert semantics["occurrence_count"] >= 3
    assert "audio" in semantics["modalities"]
    assert "signal" in semantics["modalities"]
    assert "bridge" in semantics["route_sources"]
    assert any(token in semantics["stable_query_tokens"] for token in ("audio", "signal", "fusion"))
    assert semantics["chain_examples"]
    assert semantics["contrastive_recommendation"] == "reuse_and_generalize"


@pytest.mark.cuda
def test_failing_multimodal_patterns_create_generalized_antipattern_entries(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_multimodal_antigrammar")

    chain = [
        "tool_geom_profile_lathe_mesh_v1",
        "tool_fusion_surface_material_projection_v1",
    ]
    queries = [
        "contour mesh texture failure history",
        "mesh contour texture failure archive",
        "texture contour mesh failure lesson",
    ]
    for idx, query in enumerate(queries):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_fusion_surface_material_projection_v1",
                "query_context": query,
                "specialist_id": "VisualSpecialist",
                "domain_hint": "visual",
                "math_core_tier": 2,
                "execution_us": 2200 + idx,
                "outcome": -1,
                "quality_signal": 0.1,
                "ternary_quality": -1,
                "timestamp_us": 2200 + idx,
                "chain_depth": len(chain),
                "promotion_pressure": True,
                "chain_tool_ids": chain,
                "chain_runtime_statuses": ["ptx_bridge_available"] * len(chain),
            }
        )

    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    multimodal_rules = [
        entry for entry in grammar.entries
        if isinstance(entry, dict)
        and str(entry.get("language", "")) == "execution"
        and str((entry.get("semantics") or {}).get("source", "")) == "auto_detected_multimodal_contrastive"
    ]

    assert multimodal_rules
    target = multimodal_rules[0]
    semantics = target["semantics"]
    assert target["pattern"] == "multimodal_execution_negative"
    assert semantics["pattern_type"] == "execution_multimodal_antipattern"
    assert semantics["failure_count"] >= 3
    assert "bridge" in semantics["route_sources"]
    assert any(token in semantics["stable_query_tokens"] for token in ("contour", "mesh", "failure"))
    assert semantics["contrastive_recommendation"] == "avoid_or_invert"
