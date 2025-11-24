import pygltflib

from knowledge3d.cranium.reality_gltf_export import (
    EXPORT_ALL,
    export_system_to_gltf,
    generate_all_system_gltfs,
)
from knowledge3d.cranium.reality_physics_export import export_water_molecule


def test_export_water_molecule_to_gltf(tmp_path):
    system = export_water_molecule()
    output_path = tmp_path / "water.glb"

    export_system_to_gltf(system, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    gltf = pygltflib.GLTF2().load_binary(str(output_path))
    assert gltf.meshes
    assert gltf.accessors[0].count == 3
    assert gltf.extras["node_id"] == system.node_id
    assert gltf.extras["rpn_tier"] == system.rpn_tier
    assert gltf.extras["matryoshka_dim"] == system.matryoshka_dim
    assert gltf.extras["state"]


def test_generate_all_system_gltfs(tmp_path):
    generate_all_system_gltfs(tmp_path)

    produced = {p.name for p in tmp_path.iterdir() if p.is_file()}
    expected = {f"{fn().node_id.replace('system:', '')}.glb" for fn in EXPORT_ALL}

    assert produced == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)

    sample_path = tmp_path / sorted(expected)[0]
    gltf = pygltflib.GLTF2().load_binary(str(sample_path))
    assert gltf.meshes[0].primitives[0].attributes.POSITION == 0
    assert gltf.scene == 0
