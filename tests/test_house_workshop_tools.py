from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_workshop_tools import WORKSHOP_TOOLS


def test_workshop_tools_reference_tool_galaxy() -> None:
    for tool in WORKSHOP_TOOLS:
        assert tool.meaning_class == "tool_object"
        assert tool.house_room == "House/Workshop"
        assert any(ref.startswith("tool_") for ref in tool.taxonomy_refs)


def test_workshop_tool_shapes_constructable() -> None:
    bridge = MeshBridge()
    for tool in WORKSHOP_TOOLS:
        result = bridge.execute_rpn_program(tool.visual_rpn or "")
        assert result.mesh.vertices
