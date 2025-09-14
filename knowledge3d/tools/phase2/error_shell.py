import math
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def _uv_sphere(segments: int = 12, rings: int = 8, radius: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idx = []
    for i in range(rings + 1):
        v = i / rings
        phi = v * math.pi
        for j in range(segments + 1):
            u = j / segments
            theta = u * 2 * math.pi
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts.extend([x, y, z])
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            idx.extend([a, b, a + 1,  b, b + 1, a + 1])
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


def make_error_shell_glb(out_path: Path, *, error_code: float = 500.0, dims: int = 128) -> None:
    v, t = _uv_sphere(segments=16, rings=12, radius=0.6)
    emb = np.zeros((dims,), dtype=np.float32); emb[0] = -1.0; emb[1] = float(error_code); emb[2] = 0.0

    pos_bytes = v.tobytes(order='C')
    idx_bytes = t.tobytes(order='C')
    emb_bytes = emb.tobytes(order='C')

    def align4(n: int) -> int: return (n + 3) & ~3
    chunks = []
    off = 0
    pos_off = off; chunks.append(pos_bytes); off += len(pos_bytes)
    pad = align4(off) - off; chunks.append(b"\x00" * pad); off += pad
    idx_off = off; chunks.append(idx_bytes); off += len(idx_bytes)
    pad = align4(off) - off; chunks.append(b"\x00" * pad); off += pad
    emb_off = off; chunks.append(emb_bytes); off += len(emb_bytes)
    pad = align4(off) - off; chunks.append(b"\x00" * pad); off += pad
    blob = b"".join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=idx_off, byteLength=len(idx_bytes), target=34963),
        BufferView(buffer=0, byteOffset=emb_off, byteLength=len(emb_bytes), target=34962),
    ]
    glb.accessors = [
        Accessor(bufferView=0, componentType=5126, count=v.size // 3, type='VEC3'),
        Accessor(bufferView=1, componentType=5125, count=t.size, type='SCALAR'),
    ]
    mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.8), alphaMode='BLEND', name='ErrorShell', emissiveFactor=[0.0,0.0,0.0])
    glb.materials = [mat]
    prim = Primitive()
    prim.attributes = {'POSITION': 0}
    prim.indices = 1
    prim.mode = 4
    prim.material = 0
    prim.extras = {
        'k3d': {
            'version': '3.0',
            'memory_realm': 'garden',
            'client_views': { 'human': {'render_mode': 'pbr'}, 'ai': {'render_mode': 'embedding', 'direct_buffer_access': True} },
            'vectorsView': 0,
            'embeddingsView': 2,
            'embeddingDims': dims,
        },
        'k3d_garden': {
            'tree_id': 'error_shell',
            'domain': 'Unknown',
            'version': '1.0',
            'complexity': 0,
            'is_chiral': False,
            'checksum': None,
            'source_ref': None,
        }
    }
    glb.meshes = [Mesh(primitives=[prim])]
    glb.nodes = [Node(mesh=0, name='ErrorShell')]
    glb.scenes = [Scene(nodes=[0])]
    glb.scene = 0
    glb.set_binary_blob(blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb.save_binary(str(out_path))

