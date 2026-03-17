from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_gallery_displays import GALLERY_DISPLAYS


def test_gallery_displays_reference_knowledge() -> None:
    for display in GALLERY_DISPLAYS:
        assert display.meaning_class == "display"
        assert display.house_room == "House/Gallery"
        assert display.taxonomy_refs


def test_gallery_display_frames_constructable() -> None:
    bridge = MeshBridge()
    for display in GALLERY_DISPLAYS:
        result = bridge.execute_rpn_program(display.visual_rpn or "")
        assert result.mesh.vertices
