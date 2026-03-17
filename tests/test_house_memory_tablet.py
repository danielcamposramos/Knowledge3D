from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_memory_tablet import MEMORY_TABLET


def test_memory_tablet_exists() -> None:
    assert MEMORY_TABLET.star_id == "memory_tablet"
    assert MEMORY_TABLET.meaning_class == "tablet"
    assert MEMORY_TABLET.house_room == "House"


def test_memory_tablet_shape_constructable() -> None:
    bridge = MeshBridge()
    result = bridge.execute_rpn_program(MEMORY_TABLET.visual_rpn or "")
    assert result.mesh.vertices
    assert result.mesh.triangles


def test_memory_tablet_references_all_domains() -> None:
    refs = set(MEMORY_TABLET.taxonomy_refs)
    assert "concept_language" in refs
    assert "concept_mathematics" in refs
    assert "concept_visual_art" in refs
    assert "concept_physics" in refs
    assert "concept_biology" in refs
    assert "concept_tool" in refs
