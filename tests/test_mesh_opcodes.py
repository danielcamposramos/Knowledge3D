from __future__ import annotations

from pathlib import Path
import struct

import pytest

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.cranium.ptx_runtime.mesh_engine import MeshRPNEngine
from knowledge3d.cranium.ptx_runtime.mesh_opcodes import MeshBuffer
from knowledge3d.knowledgeverse.objects_3d_galaxy import default_3d_objects_entries
from knowledge3d.tools.training_pipelines.glb_decomposer import decompose_glb_to_stars


def _cube_program() -> str:
    return "1.0 GEN_CUBE"


def _bbox(mesh: MeshBuffer) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [vertex[0] for vertex in mesh.vertices]
    ys = [vertex[1] for vertex in mesh.vertices]
    zs = [vertex[2] for vertex in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _write_cube_glb(path: Path) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    GLTF2 = pygltflib.GLTF2
    Buffer = pygltflib.Buffer
    BufferView = pygltflib.BufferView
    Accessor = pygltflib.Accessor
    Primitive = pygltflib.Primitive
    Mesh = pygltflib.Mesh
    Node = pygltflib.Node
    Scene = pygltflib.Scene
    Attributes = pygltflib.Attributes

    vertices = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    indices = [
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        0, 4, 7, 0, 7, 3,
        1, 5, 6, 1, 6, 2,
        3, 2, 6, 3, 6, 7,
        0, 1, 5, 0, 5, 4,
    ]

    position_blob = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    index_blob = b"".join(struct.pack("<H", index) for index in indices)
    position_offset = 0
    index_offset = len(position_blob)
    blob = position_blob + index_blob

    gltf = GLTF2()
    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=position_offset, byteLength=len(position_blob), target=34962),
        BufferView(buffer=0, byteOffset=index_offset, byteLength=len(index_blob), target=34963),
    ]
    gltf.accessors = [
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=5126,
            count=len(vertices),
            type="VEC3",
            min=[-0.5, -0.5, -0.5],
            max=[0.5, 0.5, 0.5],
        ),
        Accessor(
            bufferView=1,
            byteOffset=0,
            componentType=5123,
            count=len(indices),
            type="SCALAR",
            min=[0],
            max=[7],
        ),
    ]
    gltf.meshes = [Mesh(name="cube", primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])]
    gltf.nodes = [Node(mesh=0)]
    gltf.scenes = [Scene(nodes=[0])]
    gltf.scene = 0
    gltf.set_binary_blob(blob)
    gltf.save_binary(str(path))


def test_mesh_begin_vertex_face_buffer_lifecycle():
    engine = MeshRPNEngine()
    mesh = engine.evaluate(
        "MESH_BEGIN "
        "0 0 0 VERTEX3 "
        "1 0 0 VERTEX3 "
        "0 1 0 VERTEX3 "
        "0 1 2 TRI_FACE "
        "MESH_END"
    )
    assert len(mesh.vertices) == 3
    assert len(mesh.triangles) == 1
    assert len(mesh.normals) == 3


def test_gen_cube_topology_is_valid():
    mesh = MeshRPNEngine().evaluate(_cube_program())
    assert len(mesh.vertices) == 24
    assert len(mesh.triangles) == 12
    assert len(mesh.normals) == 24


def test_mat4_apply_translates_cube_vertices():
    mesh = MeshRPNEngine().evaluate("1.0 GEN_CUBE 2.0 3.0 4.0 MAT4_TRANSLATE MAT4_APPLY")
    bbox_min, bbox_max = _bbox(mesh)
    assert bbox_min == pytest.approx((1.5, 2.5, 3.5))
    assert bbox_max == pytest.approx((2.5, 3.5, 4.5))


def test_csg_subtract_two_cubes_produces_hollow_shell():
    mesh = MeshRPNEngine().evaluate(
        "2.0 GEN_CUBE "
        "1.0 GEN_CUBE "
        "0.0 0.0 0.0 MAT4_TRANSLATE MAT4_APPLY "
        "CSG_SUBTRACT"
    )
    bbox_min, bbox_max = _bbox(mesh)
    assert bbox_min == pytest.approx((-1.0, -1.0, -1.0))
    assert bbox_max == pytest.approx((1.0, 1.0, 1.0))
    assert len(mesh.triangles) > 12


def test_extrude_square_profile_produces_prism():
    mesh = MeshRPNEngine().evaluate(
        "0 0 MOVE 1 0 LINE 1 1 LINE 0 1 LINE CLOSE 2 EXTRUDE"
    )
    bbox_min, bbox_max = _bbox(mesh)
    assert bbox_min == pytest.approx((0.0, 0.0, -1.0))
    assert bbox_max == pytest.approx((1.0, 1.0, 1.0))
    assert len(mesh.triangles) >= 12


def test_lathe_profile_produces_revolved_mesh():
    mesh = MeshRPNEngine().evaluate(
        "0.4 0.0 MOVE 0.5 0.4 LINE 0.25 1.0 LINE 16 LATHE"
    )
    assert len(mesh.vertices) == 48
    assert len(mesh.triangles) > 0


def test_glb_decomposer_emits_meaning_star_for_cube(tmp_path: Path):
    glb_path = tmp_path / "cube.glb"
    _write_cube_glb(glb_path)
    stars = decompose_glb_to_stars(glb_path)
    assert len(stars) == 1
    assert stars[0].visual_rpn == "1.000000 GEN_CUBE"
    assert stars[0].surface_forms["en"].word_ref == "cube"


def test_3d_objects_bootstrap_uses_real_h1_tokens():
    entries = default_3d_objects_entries()
    cube = next(entry for entry in entries if entry["id"] == "obj3d_gen_cube")
    lathe = next(entry for entry in entries if entry["id"] == "obj3d_gen_lathe_profile")
    assert cube["rpn_program"] == "SIZE GEN_CUBE"
    assert lathe["rpn_program"] == "SEGMENTS LATHE"
    assert MeshBridge().is_mesh_program(cube["rpn_program"])
