from __future__ import annotations

import argparse
from typing import Tuple
import math
import struct

from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def _align4(n: int) -> int: return (n + 3) & ~3


def _add_plane(V: list[float], I: list[int], cx: float, cy: float, cz: float, w: float, d: float) -> None:
    x0, x1 = cx - w*0.5, cx + w*0.5
    z0, z1 = cz - d*0.5, cz + d*0.5
    base = len(V) // 3
    V += [x0,cy,z0,  x1,cy,z0,  x1,cy,z1,  x0,cy,z1]
    I += [base, base+1, base+2, base+2, base+3, base]


def _add_box(V: list[float], I: list[int], cx: float, cy: float, cz: float, w: float, h: float, d: float) -> None:
    x,y,z = w*0.5, h*0.5, d*0.5
    base = len(V) // 3
    V += [
        -x+cx,-y+cy,-z+cz,  x+cx,-y+cy,-z+cz,  x+cx, y+cy,-z+cz, -x+cx, y+cy,-z+cz,
        -x+cx,-y+cy, z+cz,  x+cx,-y+cy, z+cz,  x+cx, y+cy, z+cz, -x+cx, y+cy, z+cz,
    ]
    I += [
        base+0,base+1,base+2, base+2,base+3,base+0,
        base+4,base+7,base+6, base+6,base+5,base+4,
        base+0,base+4,base+5, base+5,base+1,base+0,
        base+2,base+6,base+7, base+7,base+3,base+2,
        base+0,base+3,base+7, base+7,base+4,base+0,
        base+1,base+5,base+6, base+6,base+2,base+1,
    ]


def build_living_room(width: float, height: float, depth: float, out_path: str) -> None:
    V: list[float] = []
    I: list[int] = []
    # Floor
    _add_plane(V, I, 0.0, 0.0, 0.0, width, depth)
    # Sofa
    _add_box(V, I, 0.0, 0.5, -depth*0.5 + 2.0, 4.0, 1.0, 2.0)
    # Screen
    _add_box(V, I, 0.0, height*0.7, depth*0.5 - 0.1, width*0.8, height*0.4, 0.05)
    # Tablet
    _add_box(V, I, -1.0, 1.2, 0.0, 0.3, 0.5, 0.02)

    pos_bytes = struct.pack('<' + 'f'*len(V), *V)
    idx_bytes = struct.pack('<' + 'I'*len(I), *I)
    off = 0
    chunks: list[bytes] = []
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
    # Materials
    mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.6))
    glb.materials = [mat]
    prim = Primitive(); prim.attributes={'POSITION':0}; prim.indices=1; prim.mode=4; prim.material=0
    prim.extras = { 'k3d': { 'version':'3.0', 'memory_realm':'living_room', 'client_views': { 'human': {'render_mode':'pbr'}, 'ai': {'render_mode':'embedding'} } } }
    glb.meshes = [Mesh(primitives=[prim])]
    glb.nodes = [Node(mesh=0, name='living_room')]
    glb.scenes = [Scene(nodes=[0])]; glb.scene = 0
    glb.set_binary_blob(blob)
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=20.0)
    ap.add_argument('--height', type=float, default=8.0)
    ap.add_argument('--depth', type=float, default=20.0)
    ap.add_argument('--out', default='viewer/public/living_room.glb')
    args = ap.parse_args()
    build_living_room(args.width, args.height, args.depth, args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

