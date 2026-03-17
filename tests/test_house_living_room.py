from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS


def test_living_room_exists() -> None:
    living = [room for room in HOUSE_ROOMS if room.star_id == "room_living"]
    assert len(living) == 1
    assert living[0].house_room == "House/LivingRoom"
    assert "furniture_holodesk" in living[0].component_refs
    assert "furniture_sofa" in living[0].component_refs


def test_living_room_furniture_exists() -> None:
    furniture = {item.star_id: item for item in HOUSE_FURNITURE}
    assert furniture["furniture_sofa"].house_room == "House/LivingRoom"
    assert furniture["furniture_holodesk"].house_room == "House/LivingRoom"
    assert "concept_mathematics" in furniture["furniture_holodesk"].taxonomy_refs


def test_living_room_furniture_constructable() -> None:
    bridge = MeshBridge()
    for star_id in ("furniture_sofa", "furniture_holodesk"):
        star = next(item for item in HOUSE_FURNITURE if item.star_id == star_id)
        result = bridge.execute_rpn_program(star.visual_rpn or "")
        assert result.mesh.vertices
        assert result.mesh.triangles
