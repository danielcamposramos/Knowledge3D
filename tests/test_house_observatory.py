from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_observatory import OBSERVATORY_INSTRUMENTS


def test_observatory_instruments_in_bathtub() -> None:
    for instrument in OBSERVATORY_INSTRUMENTS:
        assert instrument.meaning_class == "instrument"
        assert instrument.house_room == "House/Bathtub"
        assert instrument.taxonomy_refs


def test_observatory_shapes_constructable() -> None:
    bridge = MeshBridge()
    for instrument in OBSERVATORY_INSTRUMENTS:
        result = bridge.execute_rpn_program(instrument.visual_rpn or "")
        assert result.mesh.vertices
        assert result.mesh.triangles
