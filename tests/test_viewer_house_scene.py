from __future__ import annotations

from pathlib import Path

import pytest


def test_house_glb_contains_viewer_nav_graph_and_tablet() -> None:
    pygltflib = pytest.importorskip("pygltflib")
    glb_path = Path("viewer/public/house.glb")
    assert glb_path.exists()
    gltf = pygltflib.GLTF2().load(str(glb_path))
    house_node = next(node for node in (gltf.nodes or []) if node.name == "House")
    assert isinstance(house_node.extras, dict)
    k3d = house_node.extras.get("k3d", {})
    assert "nav_graph" in k3d
    nav_graph = k3d["nav_graph"]
    assert "room_library" in nav_graph["nodes"]
    assert any(edge["door"] == "door_library_garden" for edge in nav_graph["edges"])
    assert any(node.name == "memory_tablet" for node in (gltf.nodes or []))
