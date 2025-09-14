import argparse
import math
import random
import struct
from typing import Tuple

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def build_bathtub_room(width: float, height: float, depth: float, out_path: str, grid_size: int = 32, tub_segments: int = 32, particles: int = 24) -> None:
    # Undulating floor
    verts = []
    idx = []
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            x = -width*0.5 + (width * i) / grid_size
            z = -depth*0.5 + (depth * j) / grid_size
            y = 0.1 * math.sin(x * 0.5) * math.cos(z * 0.5)
            verts += [x, y, z]
    for i in range(grid_size):
        for j in range(grid_size):
            idx0 = i * (grid_size + 1) + j
            idx1 = idx0 + 1
            idx2 = idx0 + (grid_size + 1)
            idx3 = idx2 + 1
            idx += [idx0, idx2, idx1, idx1, idx2, idx3]

    # Oval tub rings (side walls)
    base = len(verts) // 3
    rx, rz, th = 1.5, 2.0, 0.8
    for i in range(tub_segments + 1):
        a = 2.0 * math.pi * i / tub_segments
        x = rx * math.cos(a); z = rz * math.sin(a)
        verts += [x, 0.0, z]
    for i in range(tub_segments + 1):
        a = 2.0 * math.pi * i / tub_segments
        x = rx * math.cos(a); z = rz * math.sin(a)
        verts += [x, th, z]
    for i in range(tub_segments):
        b0 = base + i
        b1 = base + (i + 1) % tub_segments
        t0 = base + (tub_segments + 1) + i
        t1 = base + (tub_segments + 1) + (i + 1) % tub_segments
        idx += [b0, t0, b1,  b1, t0, t1]

    # Binary blobs
    pos_bytes = struct.pack('<' + 'f'*len(verts), *verts)
    idx_bytes = struct.pack('<' + 'I'*len(idx), *idx)

    # Dream particles: POINTS mode positions
    random.seed(42)
    pts = []
    for _ in range(particles):
        x = (random.random()*2.0 - 1.0) * (width*0.4)
        y = 1.0 + random.random() * 1.0
        z = (random.random()*2.0 - 1.0) * (depth*0.4)
        pts += [x, y, z]
    pts_bytes = struct.pack('<' + 'f'*len(pts), *pts)

    def align4(n: int) -> int: return (n + 3) & ~3
    chunks = []; off = 0
    # Positions (floor+tub)
    p_off = off; chunks.append(pos_bytes); off += len(pos_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
    # Indices (floor+tub)
    i_off = off; chunks.append(idx_bytes); off += len(idx_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
    # Particles positions
    pt_off = off; chunks.append(pts_bytes); off += len(pts_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
    blob = b''.join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=p_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=i_off, byteLength=len(idx_bytes), target=34963),
        BufferView(buffer=0, byteOffset=pt_off, byteLength=len(pts_bytes), target=34962),
    ]
    acc_pos = Accessor(bufferView=0, componentType=5126, count=len(verts)//3, type='VEC3')
    acc_idx = Accessor(bufferView=1, componentType=5125, count=len(idx), type='SCALAR')
    acc_pts = Accessor(bufferView=2, componentType=5126, count=len(pts)//3, type='VEC3')
    glb.accessors = [acc_pos, acc_idx, acc_pts]

    # Materials
    mat_floor = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9))
    mat_particles = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.2))
    glb.materials = [mat_floor, mat_particles]

    prim_floor = Primitive(); prim_floor.attributes={'POSITION':0}; prim_floor.indices=1; prim_floor.mode=4; prim_floor.material=0
    prim_particles = Primitive(); prim_particles.attributes={'POSITION':2}; prim_particles.mode=0; prim_particles.material=1
    # Extras for realm tagging
    prim_floor.extras = { 'k3d': { 'version':'3.0', 'memory_realm':'bathtub', 'client_views': { 'human': {'render_mode':'pbr'}, 'ai': {'render_mode':'embedding'} } } }
    prim_particles.extras = prim_floor.extras

    mesh_floor = Mesh(primitives=[prim_floor])
    mesh_particles = Mesh(primitives=[prim_particles])
    glb.meshes = [mesh_floor, mesh_particles]
    glb.nodes = [Node(mesh=0, name='BathtubFloorAndTub'), Node(mesh=1, name='DreamParticles')]
    glb.scenes = [Scene(nodes=[0,1])]
    glb.scene = 0
    glb.set_binary_blob(blob)
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=20.0)
    ap.add_argument('--height', type=float, default=8.0)
    ap.add_argument('--depth', type=float, default=20.0)
    ap.add_argument('--out', default='viewer/public/bathtub_room.glb')
    args = ap.parse_args()
    build_bathtub_room(args.width, args.height, args.depth, args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

