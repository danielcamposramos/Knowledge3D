import argparse
import math
import struct
from typing import Dict

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


SECTOR_COLORS: Dict[str, list[float]] = {
    "Physics": [0.2, 0.4, 1.0],
    "Biology": [0.2, 1.0, 0.4],
    "Mathematics": [0.9, 0.9, 0.9],
    "Philosophy": [0.9, 0.8, 0.2],
    "Art": [1.0, 0.3, 0.7],
    "Engineering": [0.6, 0.8, 1.0],
}


def build_greenhouse(radius: float, height: float, out_path: str, sectors: Dict[str, tuple[float, float]] | None = None) -> None:
    # Floor disc (triangle fan)
    segments = 64
    v = [0.0, 0.0, 0.0]
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        v += [x, 0.0, z]
    idx_floor = []
    for i in range(1, segments + 1):
        idx_floor += [0, i, (i % segments) + 1]

    # Walls (cylinder)
    wall_segments = 32
    vb = len(v) // 3
    for i in range(wall_segments + 1):
        a = 2.0 * math.pi * i / wall_segments
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        v += [x, 0.0, z]
        v += [x, height, z]
    idx_wall = []
    for i in range(wall_segments):
        i0 = vb + i * 2
        i1 = vb + ((i + 1) % wall_segments) * 2
        idx_wall += [i0, i0 + 1, i1 + 1,  i0, i1 + 1, i1]

    pos_bytes = struct.pack('<' + 'f' * len(v), *v)
    idx_floor_bytes = struct.pack('<' + 'I' * len(idx_floor), *idx_floor)
    idx_wall_bytes = struct.pack('<' + 'I' * len(idx_wall), *idx_wall)

    def align4(n: int) -> int: return (n + 3) & ~3
    chunks: list[bytes] = []
    offset = 0
    pos_off = offset; chunks.append(pos_bytes); offset += len(pos_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad
    ifloor_off = offset; chunks.append(idx_floor_bytes); offset += len(idx_floor_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad
    iwall_off = offset; chunks.append(idx_wall_bytes); offset += len(idx_wall_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad
    blob = b"".join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=ifloor_off, byteLength=len(idx_floor_bytes), target=34963),
        BufferView(buffer=0, byteOffset=iwall_off, byteLength=len(idx_wall_bytes), target=34963),
    ]
    acc_pos = Accessor(bufferView=0, componentType=5126, count=len(v)//3, type='VEC3')
    acc_ifloor = Accessor(bufferView=1, componentType=5125, count=len(idx_floor), type='SCALAR')
    acc_iwall = Accessor(bufferView=2, componentType=5125, count=len(idx_wall), type='SCALAR')
    glb.accessors = [acc_pos, acc_ifloor, acc_iwall]

    # Materials
    mat_floor = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=1.0), emissiveFactor=[0.0,0.0,0.0])
    mat_walls = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.8), emissiveFactor=[0.0,0.0,0.0])
    glb.materials = [mat_floor, mat_walls]

    prim_floor = Primitive(); prim_floor.attributes={'POSITION':0}; prim_floor.indices=1; prim_floor.mode=4; prim_floor.material=0
    prim_walls = Primitive(); prim_walls.attributes={'POSITION':0}; prim_walls.indices=2; prim_walls.mode=4; prim_walls.material=1

    mesh_floor = Mesh(primitives=[prim_floor])
    mesh_walls = Mesh(primitives=[prim_walls])
    glb.meshes = [mesh_floor, mesh_walls]
    glb.nodes = [Node(mesh=0, name='GreenhouseFloor'), Node(mesh=1, name='GreenhouseWalls')]
    glb.scenes = [Scene(nodes=[0,1])]
    glb.scene = 0

    # Scene extras with greenhouse metadata
    if sectors is None:
        sectors = {}
    glb.scenes[0].extras = {
        'k3d_garden': {
            'memory_realm': 'garden',
            'room_type': 'circular_greenhouse',
            'center_position': [0.0, 0.0, 0.0],
            'radius': float(radius),
            'knowledge_sectors': {k: [float(a), float(b)] for k,(a,b) in sectors.items()},
        }
    }

    glb.set_binary_blob(blob)
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius', type=float, default=10.0)
    ap.add_argument('--height', type=float, default=5.0)
    ap.add_argument('--out', default='viewer/public/greenhouse_base.glb')
    args = ap.parse_args()
    build_greenhouse(args.radius, args.height, args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

