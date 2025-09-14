import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness

from .tree_loader import TreeLoader
from .registry import GARDEN_DIR, load_registry


def yaw_to_quat(yaw: float) -> List[float]:
    # Yaw around Y axis
    half = yaw * 0.5
    cy, sy = math.cos(half), math.sin(half)
    # quaternion [x,y,z,w]
    return [0.0, float(sy), 0.0, float(cy)]


def render_garden(out_path: str | Path = None) -> str:
    reg = load_registry()
    loader = TreeLoader()
    buf_chunks: List[bytes] = []
    views: List[BufferView] = []
    accs: List[Accessor] = []
    mats: List[Material] = [Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9))]
    meshes: List[Mesh] = []
    nodes: List[Node] = []

    def align4(n: int) -> int: return (n + 3) & ~3
    offset = 0

    for t in reg.get('trees', []):
        model, status = loader.load_tree(t['tree_id'])
        if model is None:
            continue
        prim = model.meshes[0].primitives[0]
        k3d = prim.extras.get('k3d', {}) if prim.extras else {}
        vview = int(k3d.get('vectorsView', 0))
        # fetch positions
        bv = model.bufferViews[vview]
        blob = model.binary_blob()
        pos = blob[(bv.byteOffset or 0): (bv.byteOffset or 0) + bv.byteLength]
        pcount = (bv.byteLength // 4) // 3
        # indices
        idx_bytes = b''
        icount = 0
        if prim.indices is not None:
            ia = model.accessors[prim.indices]
            ibv = model.bufferViews[ia.bufferView]
            iblob = blob[(ibv.byteOffset or 0): (ibv.byteOffset or 0) + ibv.byteLength]
            idx_bytes = iblob
            icount = ia.count

        # positions view
        p_off = offset; buf_chunks.append(pos); offset += len(pos)
        pad = align4(offset) - offset; buf_chunks.append(b"\x00" * pad); offset += pad
        views.append(BufferView(buffer=0, byteOffset=p_off, byteLength=len(pos), target=34962))
        # indices view
        i_off = offset; buf_chunks.append(idx_bytes); offset += len(idx_bytes)
        pad = align4(offset) - offset; buf_chunks.append(b"\x00" * pad); offset += pad
        views.append(BufferView(buffer=0, byteOffset=i_off, byteLength=len(idx_bytes), target=34963))

        acc_pos = Accessor(bufferView=len(views) - 2, componentType=5126, count=pcount, type='VEC3')
        acc_idx = Accessor(bufferView=len(views) - 1, componentType=5125, count=icount, type='SCALAR')
        accs.extend([acc_pos, acc_idx])

        prim_out = Primitive()
        prim_out.attributes = {'POSITION': len(accs) - 2}
        prim_out.indices = len(accs) - 1
        prim_out.mode = int(prim.mode or 1)
        prim_out.material = 0
        prim_out.extras = prim.extras
        mesh = Mesh(primitives=[prim_out])
        meshes.append(mesh)
        tx, ty, tz = t.get('position', [0.0, 0.0, 0.0])
        rot = float(t.get('rotation', 0.0))
        nodes.append(Node(mesh=len(meshes) - 1, name=t.get('tree_id','tree'), translation=[float(tx), float(ty), float(tz)], rotation=yaw_to_quat(rot)))

    blob = b''.join(buf_chunks)
    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = views
    glb.accessors = accs
    glb.materials = mats
    glb.meshes = meshes
    glb.nodes = nodes
    glb.scenes = [Scene(nodes=list(range(len(nodes))))]
    glb.scene = 0
    glb.set_binary_blob(blob)
    out = Path(out_path) if out_path else (GARDEN_DIR / 'knowledge_garden.glb')
    out.parent.mkdir(parents=True, exist_ok=True)
    glb.save_binary(str(out))
    return str(out)

