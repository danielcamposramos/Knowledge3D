from __future__ import annotations

import argparse
from pathlib import Path
import struct

from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def _align4(n: int) -> int: return (n + 3) & ~3


def build_house_master(out_path: str, scale: float = 1.0) -> None:
    # Floor plane (100x100)
    w = 100.0 * scale; d = 100.0 * scale
    x0, x1, z0, z1 = -w*0.5, w*0.5, -d*0.5, d*0.5
    V = [x0,0,z0,  x1,0,z0,  x1,0,z1,  x0,0,z1]
    I = [0,1,2, 2,3,0]
    # Zone markers: AI-only, encoded in extras
    zones = {
        'library':      (-30.0*scale, 0.0, 0.0),
        'garden':       ( 30.0*scale, 0.0, 0.0),
        'workshop':     (  0.0, 0.0, 30.0*scale),
        'bathtub':      (  0.0, 0.0,-30.0*scale),
        'living_room':  (  0.0, 0.0,  0.0),
    }

    pos_bytes = struct.pack('<' + 'f'*len(V), *V)
    idx_bytes = struct.pack('<' + 'I'*len(I), *I)
    off = 0
    chunks = []
    p_off = off; chunks.append(pos_bytes); off += len(pos_bytes); pad = _align4(off)-off; chunks.append(b"\x00"*pad); off += pad
    i_off = off; chunks.append(idx_bytes); off += len(idx_bytes); pad = _align4(off)-off; chunks.append(b"\x00"*pad); off += pad
    blob = b''.join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=p_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=i_off, byteLength=len(idx_bytes), target=34963),
    ]
    glb.accessors = [
        Accessor(bufferView=0, componentType=5126, count=len(V)//3, type='VEC3'),
        Accessor(bufferView=1, componentType=5125, count=len(I), type='SCALAR'),
    ]

    mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9))
    glb.materials = [mat]
    prim = Primitive(); prim.attributes={'POSITION':0}; prim.indices=1; prim.mode=4; prim.material=0
    # Encode zone registry in extras
    prim.extras = {
        'k3d': {
            'version': '3.0',
            'memory_realm': 'house',
            'client_views': { 'human': {'render_mode':'pbr'}, 'ai': {'render_mode':'embedding'} },
        },
        'k3d_house': {
            'zones': { name: {'position': list(map(float, pos))} for name, pos in zones.items() },
        },
    }
    glb.meshes = [Mesh(primitives=[prim])]
    glb.nodes = [Node(mesh=0, name='house_master')]
    glb.scenes = [Scene(nodes=[0])]; glb.scene = 0
    glb.set_binary_blob(blob)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    glb.save_binary(str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='viewer/public/house/house_master.glb')
    ap.add_argument('--scale', type=float, default=1.0)
    args = ap.parse_args()
    build_house_master(args.out, scale=args.scale)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

