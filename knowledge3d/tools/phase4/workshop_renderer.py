import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def _align4(n: int) -> int:
    return (n + 3) & ~3


def _quat_from_euler(x: float, y: float, z: float) -> List[float]:
    # XYZ intrinsic rotations → quaternion [x,y,z,w]
    cx, cy, cz = math.cos(x * 0.5), math.cos(y * 0.5), math.cos(z * 0.5)
    sx, sy, sz = math.sin(x * 0.5), math.sin(y * 0.5), math.sin(z * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return [float(qx), float(qy), float(qz), float(qw)]


def _gen_geometry(shape: str, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    s = shape.lower() if isinstance(shape, str) else 'tetrahedron'
    if s == 'tetrahedron':
        v = np.array([
            0.0, 1.0, 0.0,
            -0.866, -0.5, 0.0,
            0.866, -0.5, 0.0,
            0.0, 0.0, 1.633,
        ], dtype=np.float32) * scale
        i = np.array([0, 1, 2,  0, 2, 3,  0, 3, 1,  1, 3, 2], dtype=np.uint32)
        return v, i
    if s == 'cube':
        v = np.array([
            -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
            -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
        ], dtype=np.float32) * (scale * 0.5)
        i = np.array([
            0,1,2, 2,3,0,  4,7,6, 6,5,4,
            0,4,5, 5,1,0,  2,6,7, 7,3,2,
            0,3,7, 7,4,0,  1,5,6, 6,2,1
        ], dtype=np.uint32)
        return v, i
    # default: icosahedron-like proxy (rough)
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
    v = np.asarray(verts, dtype=np.float32)
    # normalize to unit radius, then scale
    v = v.reshape(-1, 3)
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    v = (v / n * scale).astype(np.float32).reshape(-1)
    i = np.asarray(faces, dtype=np.uint32)
    return v, i


class WorkshopRenderer:
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
        # default structure
        return {
            'workshop_version': '1.0',
            'room_path': 'viewer/public/workshop_room.glb',
            'center_position': [0.0, 0.0, 0.0],
            'stars': [],
        }

    def render(self) -> str:
        room_path = self.registry.get('room_path') or 'viewer/public/workshop_room.glb'
        room = GLTF2().load_binary(room_path)
        blob = room.binary_blob() or b''
        offset = len(blob)
        chunks = [blob]

        # Ensure arrays are present
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

        for star in self.registry.get('stars', []):
            if str(star.get('status','active')).lower() != 'active':
                continue
            shape = str(star.get('shape_type', 'tetrahedron'))
            pos = [float(x) for x in star.get('position', [0.0, 0.0, 0.0])]
            rot_euler = [float(x) for x in star.get('rotation', [0.0, 0.0, 0.0])]
            q = _quat_from_euler(rot_euler[0], rot_euler[1], rot_euler[2])
            emb = np.asarray(star.get('embedding', []), dtype=np.float32).reshape(-1)
            v, idx = _gen_geometry(shape, scale=1.0)

            # Append positions, indices, embedding into the same buffer
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

            # Primitive
            prim = Primitive()
            prim.attributes = { 'POSITION': ai }
            prim.indices = ai + 1
            prim.mode = 4  # TRIANGLES
            # Extras with embedding linkage
            k3d = {
                'version': '3.0',
                'memory_realm': 'workshop',
                'client_views': {
                    'human': { 'render_mode': 'pbr' },
                    'ai': { 'render_mode': 'embedding', 'direct_buffer_access': True },
                },
                'vectorsView': bvi + 0,
                'embeddingDims': int(emb.size),
            }
            if e_len:
                k3d['embeddingsView'] = bvi + 2
            prim.extras = {
                'k3d': k3d,
                'k3d_workshop': {
                    'id': star.get('id'),
                    'shape_type': shape,
                    'media_types': star.get('media_types', []),
                    'is_fused': bool(star.get('is_fused', False)),
                    'source_galaxy_id': star.get('source_galaxy_id') or star.get('source_galaxy_ids'),
                }
            }
            # Material (simple neutral)
            if room.materials is None:
                room.materials = []
            if not room.materials:
                room.materials.append(Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.6)))
            prim.material = 0

            # Mesh and Node
            if room.meshes is None:
                room.meshes = []
            mesh_index = len(room.meshes)
            room.meshes.append(Mesh(primitives=[prim]))

            if room.nodes is None:
                room.nodes = []
            node = Node(mesh=mesh_index, name=str(star.get('id')))
            node.translation = [pos[0], pos[1], pos[2]]
            node.rotation = q
            room.nodes.append(node)
            # attach node to scene 0
            if room.scenes[0].nodes is None:
                room.scenes[0].nodes = []
            room.scenes[0].nodes.append(len(room.nodes) - 1)

        # Update primary buffer length and blob
        total_blob = b''.join(chunks)
        room.buffers[0].byteLength = len(total_blob)
        room.set_binary_blob(total_blob)

        # Save
        out = self.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        room.save_binary(str(out))
        return str(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', default='viewer/public/workshop/workshop_registry.json')
    ap.add_argument('--out', default='viewer/public/workshop/workshop_scene.glb')
    args = ap.parse_args()
    r = WorkshopRenderer(args.registry, args.out)
    path = r.render()
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()

