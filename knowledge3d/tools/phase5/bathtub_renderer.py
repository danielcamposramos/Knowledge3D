from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material,
    PbrMetallicRoughness
)


def _align4(n: int) -> int:
    return (n + 3) & ~3


def _gen_icosa(scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        -1,  t,  0,  1,  t,  0, -1, -t,  0,  1, -t,  0,
         0, -1,  t,  0,  1,  t,  0, -1, -t,  0,  1, -t,
         t,  0, -1,  t,  0,  1, -t,  0, -1, -t,  0,  1,
    ]
    faces = [
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    ]
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    v = (v / n * scale).astype(np.float32).reshape(-1)
    i = np.asarray(faces, dtype=np.uint32)
    return v, i


def _gen_cube(scale: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = scale
    v = np.array([
        -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
    ], dtype=np.float32)
    v = v.reshape(-1, 3)
    v[:, 0] *= sx * 0.5
    v[:, 1] *= sy * 0.5
    v[:, 2] *= sz * 0.5
    v = v.reshape(-1)
    i = np.array([
        0,1,2, 2,3,0,  4,7,6, 6,5,4,
        0,4,5, 5,1,0,  2,6,7, 7,3,2,
        0,3,7, 7,4,0,  1,5,6, 6,2,1
    ], dtype=np.uint32)
    return v, i


def _gen_cylinder(radius: float = 0.2, height: float = 1.0, seg: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[float] = []
    idx: List[int] = []
    # side wall
    for i in range(seg + 1):
        a = (i / seg) * math.tau
        x, z = radius * math.cos(a), radius * math.sin(a)
        verts += [x, -height * 0.5, z]
    base = len(verts) // 3
    for i in range(seg + 1):
        a = (i / seg) * math.tau
        x, z = radius * math.cos(a), radius * math.sin(a)
        verts += [x, height * 0.5, z]
    for i in range(seg):
        b0 = i
        b1 = (i + 1)
        t0 = base + i
        t1 = base + (i + 1)
        idx += [b0, t0, b1,  b1, t0, t1]
    return np.asarray(verts, dtype=np.float32), np.asarray(idx, dtype=np.uint32)


def _geometry_for_furniture(kind: str) -> Tuple[np.ndarray, np.ndarray]:
    k = (kind or '').lower().strip()
    if k == 'bookshelf':
        return _gen_cube((0.5, 1.2, 0.2))
    if k == 'display_case':
        return _gen_icosa(0.6)
    if k == 'workbench' or k == 'table':
        return _gen_cube((0.9, 0.4, 0.6))
    if k == 'lamp':
        return _gen_icosa(0.35)
    if k == 'pillar':
        v, i = _gen_cylinder(0.15, 1.4, seg=28)
        return v, i
    if k == 'gray_shell':
        return _gen_cube((0.6, 0.6, 0.6))
    # default artifact
    return _gen_icosa(0.5)


class BathtubRenderer:
    def __init__(self, registry_path: str, output_path: str) -> None:
        self.registry_path = Path(registry_path)
        self.output_path = Path(output_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {
            'bathtub_version': '1.0',
            'room_path': 'viewer/public/bathtub_room.glb',
            'furniture': [],
        }

    def _arrange_positions(self, n: int) -> List[Tuple[float, float, float]]:
        # Arrange on an ellipse within tub bounds (rx=1.5, rz=2.0) with slight inward offset
        rx, rz = 1.25, 1.7
        y = 0.05
        pos: List[Tuple[float, float, float]] = []
        for i in range(max(1, n)):
            a = (i / max(1, n)) * math.tau
            x = rx * math.cos(a)
            z = rz * math.sin(a)
            pos.append((float(x), float(y), float(z)))
        return pos

    def render(self) -> str:
        room_path = self.registry.get('room_path') or 'viewer/public/bathtub_room.glb'
        room = GLTF2().load_binary(room_path)
        blob = room.binary_blob() or b''
        offset = len(blob)
        chunks = [blob]

        room.buffers = room.buffers or [Buffer(byteLength=0)]
        if not room.scenes:
            room.scenes = [Scene(nodes=[])]
            room.scene = 0

        def append_bytes(bs: bytes) -> Tuple[int, int]:
            nonlocal offset
            start = offset
            chunks.append(bs)
            offset += len(bs)
            pad = _align4(offset) - offset
            if pad:
                chunks.append(b"\x00" * pad)
                offset += pad
            return start, len(bs)

        items = list(self.registry.get('furniture', []) or [])
        positions = self._arrange_positions(len(items))

        # Ensure material slots
        if room.materials is None:
            room.materials = []
        solid_mat_index = None
        gray_mat_index = None
        # Solid default
        room.materials.append(Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.6)))
        solid_mat_index = len(room.materials) - 1
        # Gray translucent
        m = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.1))
        # Use extras since pygltflib minimal; set alpha via extras if not exposed
        m.alphaMode = 'BLEND'
        try:
            m.pbrMetallicRoughness.baseColorFactor = [0.5, 0.5, 0.5, 0.5]
        except Exception:
            pass
        room.materials.append(m)
        gray_mat_index = len(room.materials) - 1

        for i, it in enumerate(items):
            kind = str(it.get('furniture_kind') or 'artifact')
            emb = np.asarray(it.get('embedding', []) or [], dtype=np.float32).reshape(-1)
            # If enhanced geometry present, use it; else generate proxy by kind
            geom = it.get('geometry') if isinstance(it.get('geometry'), dict) else None
            if geom and isinstance(geom.get('vertices'), list) and isinstance(geom.get('indices'), list):
                v = np.asarray(list(map(float, geom['vertices'])), dtype=np.float32).reshape(-1)
                idx = np.asarray(list(map(int, geom['indices'])), dtype=np.uint32).reshape(-1)
            else:
                v, idx = _geometry_for_furniture(kind)
            pos_bytes = v.tobytes(order='C')
            idx_bytes = idx.tobytes(order='C')
            emb_bytes = emb.tobytes(order='C') if emb.size > 0 else b''
            p_off, p_len = append_bytes(pos_bytes)
            i_off, i_len = append_bytes(idx_bytes)
            e_off, e_len = (append_bytes(emb_bytes) if emb_bytes else (offset, 0))

            # BufferViews
            if room.bufferViews is None:
                room.bufferViews = []
            bvi = len(room.bufferViews)
            room.bufferViews.append(BufferView(buffer=0, byteOffset=p_off, byteLength=p_len, target=34962))
            room.bufferViews.append(BufferView(buffer=0, byteOffset=i_off, byteLength=i_len, target=34963))
            if e_len:
                room.bufferViews.append(BufferView(buffer=0, byteOffset=e_off, byteLength=e_len, target=34962))

            # Accessors
            if room.accessors is None:
                room.accessors = []
            acc_pos = Accessor(bufferView=bvi + 0, componentType=5126, count=v.size // 3, type='VEC3')
            acc_idx = Accessor(bufferView=bvi + 1, componentType=5125, count=idx.size, type='SCALAR')
            ai = len(room.accessors)
            room.accessors.append(acc_pos)
            room.accessors.append(acc_idx)

            # Primitive extras
            k3d = {
                'version': '3.0',
                'memory_realm': 'bathtub',
                'client_views': {
                    'human': { 'render_mode': 'pbr' },
                    'ai': { 'render_mode': 'embedding', 'direct_buffer_access': True },
                },
                'vectorsView': bvi + 0,
                'embeddingDims': int(emb.size),
            }
            if e_len:
                k3d['embeddingsView'] = bvi + 2
            prim = Primitive()
            prim.attributes = { 'POSITION': ai }
            prim.indices = ai + 1
            prim.mode = 4
            prim.extras = {
                'k3d': k3d,
                'k3d_bathtub': {
                    'star_id': it.get('star_id'),
                    'id': it.get('id'),
                    'shape_type': it.get('shape_type'),
                    'furniture_kind': kind,
                    'is_crystallized': bool(it.get('is_crystallized', False)),
                    'honesty_score': float(it.get('honesty_score', 0.0)),
                    'compression_factor': float(it.get('compression_factor', 0.0)),
                    'status': it.get('status', 'unknown'),
                }
            }
            # Pick material
            dishonest = (str(it.get('status', 'crystallized')).lower() == 'dishonest') or (kind == 'gray_shell')
            enhanced = bool(it.get('is_enhanced', False))
            if dishonest:
                prim.material = gray_mat_index
            else:
                prim.material = solid_mat_index

            # Mesh / Node
            if room.meshes is None:
                room.meshes = []
            mesh_index = len(room.meshes)
            room.meshes.append(Mesh(primitives=[prim]))

            if room.nodes is None:
                room.nodes = []
            node = Node(mesh=mesh_index, name=str(it.get('id') or it.get('star_id') or f'furniture-{i}'))
            x, y, z = positions[i if i < len(positions) else 0]
            node.translation = [x, y, z]
            room.nodes.append(node)
            if room.scenes[0].nodes is None:
                room.scenes[0].nodes = []
            room.scenes[0].nodes.append(len(room.nodes) - 1)

        # Aggregate and save
        total_blob = b''.join(chunks)
        room.buffers[0].byteLength = len(total_blob)
        room.set_binary_blob(total_blob)
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        room.save_binary(str(out))
        return str(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='viewer/public/bathtub_registry.json')
    ap.add_argument('--out', default='viewer/public/bathtub_scene.glb')
    args = ap.parse_args()
    r = BathtubRenderer(args.registry, args.out)
    out = r.render()
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
