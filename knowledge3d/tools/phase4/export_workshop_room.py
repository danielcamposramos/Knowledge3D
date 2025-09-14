import argparse
import struct
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def build_workshop_room(width: float, height: float, depth: float, out_path: str) -> None:
    # Build simple boxes for walls/screens/bench
    def add_box(verts: list[float], inds: list[int], cx: float, cy: float, cz: float, w: float, h: float, d: float):
        x0, x1 = cx - w/2, cx + w/2
        y0, y1 = cy - h/2, cy + h/2
        z0, z1 = cz - d/2, cz + d/2
        base = len(verts)//3
        verts += [x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0, x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1]
        inds += [base+0,base+1,base+2, base+2,base+3,base+0, base+4,base+7,base+6, base+6,base+5,base+4,
                 base+0,base+4,base+5, base+5,base+1,base+0, base+2,base+6,base+7, base+7,base+3,base+2,
                 base+0,base+3,base+7, base+7,base+4,base+0, base+1,base+5,base+6, base+6,base+2,base+1]

    v: list[float] = []
    idx: list[int] = []
    # Walls
    add_box(v, idx, 0.0, height*0.5, -depth*0.5 + 1.0, width*0.8, height*0.8, 0.1)
    add_box(v, idx, -width*0.5 + 1.0, height*0.5, 0.0, 0.1, height*0.8, depth*0.8)
    add_box(v, idx,  width*0.5 - 1.0, height*0.5, 0.0, 0.1, height*0.8, depth*0.8)
    # Screens
    add_box(v, idx, 0.0, height*0.7, depth*0.5 - 2.0, width*0.6, height*0.4, 0.05)
    add_box(v, idx, -width*0.5 + 3.0, height*0.5, 0.0, 2.0, 1.5, 0.05)
    add_box(v, idx,  width*0.5 - 3.0, height*0.5, 0.0, 2.0, 1.5, 0.05)
    # Workbench
    add_box(v, idx, 0.0, 0.5, 0.0, width*0.4, 1.0, depth*0.4)

    pos_bytes = struct.pack('<' + 'f'*len(v), *v)
    idx_bytes = struct.pack('<' + 'I'*len(idx), *idx)
    def align4(n): return (n+3)&~3
    chunks = []; off = 0
    pos_off = off; chunks.append(pos_bytes); off += len(pos_bytes); chunks.append(b"\x00"*((align4(off))-off)); off = align4(off)
    idx_off = off; chunks.append(idx_bytes); off += len(idx_bytes); chunks.append(b"\x00"*((align4(off))-off)); off = align4(off)
    blob = b''.join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=idx_off, byteLength=len(idx_bytes), target=34963),
    ]
    glb.accessors = [
        Accessor(bufferView=0, componentType=5126, count=len(v)//3, type='VEC3'),
        Accessor(bufferView=1, componentType=5125, count=len(idx), type='SCALAR'),
    ]
    # Materials
    mat = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9))
    glb.materials = [mat]
    prim = Primitive(); prim.attributes={'POSITION':0}; prim.indices=1; prim.mode=4; prim.material=0
    mesh = Mesh(primitives=[prim])
    glb.meshes = [mesh]
    glb.nodes = [Node(mesh=0, name='WorkshopRoom')]
    glb.scenes = [Scene(nodes=[0])]
    glb.scene = 0
    glb.set_binary_blob(blob)
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=20.0)
    ap.add_argument('--height', type=float, default=8.0)
    ap.add_argument('--depth', type=float, default=20.0)
    ap.add_argument('--out', default='viewer/public/workshop_room.glb')
    args = ap.parse_args()
    build_workshop_room(args.width, args.height, args.depth, args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()

