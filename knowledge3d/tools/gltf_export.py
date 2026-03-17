"""GLTF export helpers for procedural mesh buffers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from knowledge3d.cranium.ptx_runtime.mesh_opcodes import MeshBuffer

try:  # pragma: no cover - optional at import, required at runtime
    from pygltflib import Accessor, Asset, Buffer, BufferView, GLTF2, Mesh, Node, Primitive, Scene
except Exception:  # pragma: no cover
    GLTF2 = None  # type: ignore
    Accessor = Asset = Buffer = BufferView = Mesh = Node = Primitive = Scene = object  # type: ignore


ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT = 5126
UNSIGNED_INT = 5125


def _require_pygltflib() -> None:
    if GLTF2 is None:
        raise RuntimeError("pygltflib is required for GLTF export")


def _pad4(data: bytes) -> bytes:
    pad = (4 - (len(data) % 4)) % 4
    if pad:
        return data + (b"\x00" * pad)
    return data


@dataclass
class GltfNodeData:
    buffer_data: bytes
    buffer_views: list[BufferView]
    accessors: list[Accessor]
    mesh: Mesh
    node: Node


def mesh_to_gltf_node(
    mesh: MeshBuffer,
    *,
    name: str = "",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extras: dict[str, object] | None = None,
) -> GltfNodeData:
    """Convert a MeshBuffer into GLTF node data."""
    _require_pygltflib()
    position_bytes, normal_bytes, uv_bytes, index_bytes = mesh.to_gltf_bytes()
    position_bytes = _pad4(position_bytes)
    normal_bytes = _pad4(normal_bytes)
    uv_bytes = _pad4(uv_bytes)
    index_bytes = _pad4(index_bytes)

    pos_offset = 0
    norm_offset = pos_offset + len(position_bytes)
    uv_offset = norm_offset + len(normal_bytes)
    idx_offset = uv_offset + len(uv_bytes)
    buffer_data = position_bytes + normal_bytes + uv_bytes + index_bytes

    bounds_min, bounds_max = mesh.position_bounds()
    vertex_count = len(mesh.vertices)
    tri_index_count = len(mesh.triangles) * 3
    translation_list = [float(value) for value in translation]
    node_extras = {"k3d": dict(extras or {})}

    buffer_views = [
        BufferView(buffer=0, byteOffset=pos_offset, byteLength=len(position_bytes), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=norm_offset, byteLength=len(normal_bytes), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=uv_offset, byteLength=len(uv_bytes), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=idx_offset, byteLength=len(index_bytes), target=ELEMENT_ARRAY_BUFFER),
    ]
    accessors = [
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=FLOAT,
            count=vertex_count,
            type="VEC3",
            min=[float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])],
            max=[float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])],
        ),
        Accessor(
            bufferView=1,
            byteOffset=0,
            componentType=FLOAT,
            count=vertex_count,
            type="VEC3",
        ),
        Accessor(
            bufferView=2,
            byteOffset=0,
            componentType=FLOAT,
            count=vertex_count,
            type="VEC2",
        ),
        Accessor(
            bufferView=3,
            byteOffset=0,
            componentType=UNSIGNED_INT,
            count=tri_index_count,
            type="SCALAR",
            min=[0] if tri_index_count else [0],
            max=[max(vertex_count - 1, 0)] if tri_index_count else [0],
        ),
    ]
    mesh_obj = Mesh(
        primitives=[
            Primitive(
                attributes={"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                indices=3,
                mode=4,
            )
        ]
    )
    node = Node(name=name, mesh=0, translation=translation_list, extras=node_extras)
    return GltfNodeData(
        buffer_data=buffer_data,
        buffer_views=buffer_views,
        accessors=accessors,
        mesh=mesh_obj,
        node=node,
    )


def compose_scene(nodes: list[GltfNodeData], *, asset_generator: str = "Knowledge3D") -> GLTF2:
    """Merge multiple GLTF node fragments into a single GLTF scene."""
    _require_pygltflib()
    gltf = GLTF2(asset=Asset(version="2.0", generator=asset_generator))
    gltf.buffers = []
    gltf.bufferViews = []
    gltf.accessors = []
    gltf.meshes = []
    gltf.nodes = []
    gltf.scenes = [Scene(nodes=[])]
    gltf.scene = 0

    blob = bytearray()
    for node_data in nodes:
        chunk = _pad4(node_data.buffer_data)
        chunk_offset = len(blob)
        blob.extend(chunk)

        buffer_view_base = len(gltf.bufferViews)
        accessor_base = len(gltf.accessors)
        mesh_base = len(gltf.meshes)

        for buffer_view in node_data.buffer_views:
            view = deepcopy(buffer_view)
            view.buffer = 0
            view.byteOffset = int(getattr(view, "byteOffset", 0) or 0) + chunk_offset
            gltf.bufferViews.append(view)

        for accessor in node_data.accessors:
            acc = deepcopy(accessor)
            if getattr(acc, "bufferView", None) is not None:
                acc.bufferView = int(acc.bufferView) + buffer_view_base
            gltf.accessors.append(acc)

        mesh_obj = deepcopy(node_data.mesh)
        for primitive in list(mesh_obj.primitives or []):
            if primitive.indices is not None:
                primitive.indices = int(primitive.indices) + accessor_base
            attrs = primitive.attributes
            if isinstance(attrs, dict):
                for key, value in list(attrs.items()):
                    attrs[key] = int(value) + accessor_base
            else:
                for key in ("POSITION", "NORMAL", "TEXCOORD_0"):
                    value = getattr(attrs, key, None)
                    if value is not None:
                        setattr(attrs, key, int(value) + accessor_base)
        gltf.meshes.append(mesh_obj)

        node_obj = deepcopy(node_data.node)
        node_obj.mesh = mesh_base
        gltf.nodes.append(node_obj)

    gltf.buffers = [Buffer(byteLength=len(blob))]
    gltf.set_binary_blob(bytes(blob))
    gltf.scenes[0].nodes = list(range(len(gltf.nodes)))
    return gltf


__all__ = ["GltfNodeData", "compose_scene", "mesh_to_gltf_node"]
