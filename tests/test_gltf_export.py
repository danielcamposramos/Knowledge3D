from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pygltflib")

from knowledge3d.cranium.ptx_runtime.mesh_opcodes import generate_cube, generate_uv_sphere
from knowledge3d.tools.export_house import export_house_glb
from knowledge3d.tools.gltf_export import compose_scene, mesh_to_gltf_node


def test_mesh_buffer_to_gltf_bytes() -> None:
    mesh = generate_cube(1.0)
    pos, norms, uvs, indices = mesh.to_gltf_bytes()
    assert len(pos) == 24 * 3 * 4
    assert len(norms) == 24 * 3 * 4
    assert len(uvs) == 24 * 2 * 4
    assert len(indices) == 12 * 3 * 4


def test_mesh_to_gltf_node_creates_valid_structure() -> None:
    node_data = mesh_to_gltf_node(generate_cube(1.0), name="test_cube")
    assert len(node_data.buffer_views) == 4
    assert len(node_data.accessors) == 4
    assert node_data.node.name == "test_cube"


def test_compose_scene_creates_valid_gltf() -> None:
    nodes = [
        mesh_to_gltf_node(generate_cube(1.0), name="cube"),
        mesh_to_gltf_node(generate_uv_sphere(0.5, 8, 12), name="sphere", translation=(3.0, 0.0, 0.0)),
    ]
    gltf = compose_scene(nodes)
    assert len(gltf.nodes) >= 2
    assert len(gltf.meshes) >= 2
    assert gltf.binary_blob() is not None


def test_export_house_produces_valid_glb(tmp_path: Path) -> None:
    output = tmp_path / "house.glb"
    summary = export_house_glb(output)
    assert output.exists()
    assert output.with_name("house-content.json").exists()
    assert output.stat().st_size > 1000
    assert summary["rooms"] == 6
    assert summary["furniture"] >= 8
    assert summary["doors"] >= 5
    assert summary["tools"] >= 5
    assert summary["books"] >= 5
    assert summary["displays"] >= 4
    assert summary["instruments"] >= 3
    assert summary["tablet"] >= 1
    assert summary["content_books"] == 5
    assert summary["content_entries"] == 85
    assert summary["content_concepts"] == 10
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    assert len(gltf.nodes) > 10


def test_exported_nodes_carry_k3d_metadata(tmp_path: Path) -> None:
    output = tmp_path / "house.glb"
    export_house_glb(output)
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    assert any(isinstance(node.extras, dict) and "k3d" in node.extras for node in (gltf.nodes or []))


def test_exported_house_includes_observatory_and_tablet(tmp_path: Path) -> None:
    output = tmp_path / "house.glb"
    summary = export_house_glb(output)
    assert summary["instruments"] >= 3
    assert summary["tablet"] >= 1
    pygltflib = pytest.importorskip("pygltflib")
    gltf = pygltflib.GLTF2().load(str(output))
    house_node = next(node for node in (gltf.nodes or []) if node.name == "House")
    assert isinstance(house_node.extras, dict)
    assert "nav_graph" in house_node.extras["k3d"]
    nav_graph = house_node.extras["k3d"]["nav_graph"]
    assert "room_living" in nav_graph["nodes"]
    assert any(edge["door"] == "door_living_library" for edge in nav_graph["edges"])
