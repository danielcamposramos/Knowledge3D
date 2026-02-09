from __future__ import annotations

from pathlib import Path

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.objects_3d_galaxy import bootstrap_3d_objects_galaxy
from knowledge3d.knowledgeverse.reality_galaxy import bootstrap_reality_galaxy


def _build_kv(tmp_path: Path) -> Knowledgeverse:
    storage_root = tmp_path / "kv_autogen"
    bootstrap_reality_galaxy(storage_root=storage_root)
    bootstrap_3d_objects_galaxy(storage_root=storage_root)
    return Knowledgeverse(storage_root=storage_root)


def _generated_ids(galaxy_entries: list[dict]) -> set[str]:
    out: set[str] = set()
    for entry in galaxy_entries:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if metadata.get("generated") is True:
            entry_id = str(entry.get("id", "")).strip()
            if entry_id:
                out.add(entry_id)
    return out


def test_generate_projectile_trajectory(tmp_path):
    kv = _build_kv(tmp_path)
    result = kv.trm_navigator.generate_from_procedural(
        query="simulate projectile from origin with initial velocity 10 m/s at 45 degrees",
        source_galaxy="Reality",
        target_galaxy="Grammar",
        store_result=True,
    )
    assert "error" not in result
    assert result["metadata"]["generated"] is True
    assert result["metadata"]["source_galaxy"] == "Reality"
    assert result["metadata"]["composition_depth"] >= 1
    grammar_ids = _generated_ids(kv.galaxy_manager.get_galaxy("Grammar").entries)
    assert result["id"] in grammar_ids


def test_generate_lsystem_fractal(tmp_path):
    kv = _build_kv(tmp_path)
    result = kv.trm_navigator.generate_from_procedural(
        query="generate L-system plant with axiom F and rule F->F[+F]F[-F]F for 3 iterations",
        source_galaxy="Reality",
        target_galaxy="3DObjects",
        store_result=True,
    )
    assert "error" not in result
    assert result["metadata"]["generated"] is True
    objects_ids = _generated_ids(kv.galaxy_manager.get_galaxy("3DObjects").entries)
    assert result["id"] in objects_ids


def test_generate_3d_mesh_from_procedural(tmp_path):
    kv = _build_kv(tmp_path)
    result = kv.trm_navigator.generate_from_procedural(
        query="generate UV sphere mesh with radius 5 and 16 subdivisions",
        source_galaxy="3DObjects",
        target_galaxy="3DObjects",
        store_result=True,
    )
    assert "error" not in result
    assert result["metadata"]["generated"] is True
    assert result["metadata"]["source_galaxy"] == "3DObjects"
    assert "generate" in result["metadata"]["query"].lower()


def test_cross_modal_field_visualization(tmp_path):
    kv = _build_kv(tmp_path)
    result = kv.trm_navigator.generate_from_procedural(
        query="visualize electric field for point charge at origin with magnitude 1.0",
        source_galaxy="Reality",
        target_galaxy="Drawing",
        store_result=True,
    )
    assert "error" not in result
    assert result["metadata"]["generated"] is True
    drawing_ids = _generated_ids(kv.galaxy_manager.get_galaxy("Drawing").entries)
    assert result["id"] in drawing_ids


def test_procedural_lineage_tracking(tmp_path):
    kv = _build_kv(tmp_path)
    result = kv.trm_navigator.generate_from_procedural(
        query="simulate pendulum motion with length 1m and initial angle 30 degrees",
        source_galaxy="Reality",
        target_galaxy="Grammar",
        store_result=True,
    )
    assert "error" not in result
    metadata = result["metadata"]
    assert "source_primitives" in metadata
    assert "composition_depth" in metadata
    assert "lineage" in metadata
    assert "timestamp" in metadata
    assert isinstance(metadata["source_primitives"], list)
    assert len(metadata["source_primitives"]) > 0
    assert 1 <= int(metadata["composition_depth"]) <= 10


def test_shadow_copy_learns_from_generation(tmp_path):
    kv = _build_kv(tmp_path)
    events_before = len(kv.shadow_copy.event_buffer)
    result = kv.trm_navigator.generate_from_procedural(
        query="generate cube mesh with size 2",
        source_galaxy="3DObjects",
        target_galaxy="3DObjects",
        store_result=True,
    )
    assert "error" not in result
    events_after = len(kv.shadow_copy.event_buffer)
    assert events_after > events_before
    generation_events = [
        event for event in kv.shadow_copy.event_buffer if event.get("type") == "autonomous_generation"
    ]
    assert generation_events
    event_data = generation_events[-1].get("data", {})
    assert "source_galaxy" in event_data
    assert "target_galaxy" in event_data
    assert "composition_depth" in event_data
