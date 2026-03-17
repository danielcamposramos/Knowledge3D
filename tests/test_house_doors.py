from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_doors import HOUSE_DOORS


def test_doors_connect_adjacent_rooms() -> None:
    for door in HOUSE_DOORS:
        assert door.meaning_class == "door"
        assert door.behavior_rpn and door.behavior_rpn.startswith("DOOR_TRAVERSE")
        assert "CONNECT" in door.behavior_rpn


def test_door_visual_rpn_produces_frame() -> None:
    bridge = MeshBridge()
    for door in HOUSE_DOORS:
        result = bridge.execute_rpn_program(door.visual_rpn or "")
        assert result.mesh.vertices
        assert result.mesh.triangles
