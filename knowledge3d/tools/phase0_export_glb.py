import argparse
import json
import struct
from typing import List

import numpy as np
from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Primitive,
    Buffer,
    BufferView,
    Accessor,
)

# Numeric constants (avoid version-specific imports)
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT = 5126
UNSIGNED_SHORT = 5123
VEC3 = "VEC3"
SCALAR = "SCALAR"


def build_tetrahedron_glb(vertices: np.ndarray, indices: List[int], embedding: np.ndarray, rays: list, out_path: str) -> None:
    # Prepare binary chunks: positions (float32), indices (uint16), embedding (float32)
    pos_bytes = struct.pack("<" + "f" * len(vertices), *vertices.tolist())
    idx_bytes = struct.pack("<" + "H" * len(indices), *indices)
    emb_bytes = struct.pack("<" + "f" * len(embedding), *embedding.tolist())

    # Concatenate into a single buffer with 4-byte alignment between chunks
    def align4(n):
        return (n + 3) & ~3

    offset = 0
    chunks = []
    # positions
    pos_off = offset
    chunks.append(pos_bytes)
    offset += len(pos_bytes)
    # indices (align to 4)
    pad = align4(offset) - offset
    if pad:
        chunks.append(b"\x00" * pad)
        offset += pad
    idx_off = offset
    chunks.append(idx_bytes)
    offset += len(idx_bytes)
    # embedding (align to 4)
    pad = align4(offset) - offset
    if pad:
        chunks.append(b"\x00" * pad)
        offset += pad
    emb_off = offset
    chunks.append(emb_bytes)
    offset += len(emb_bytes)
    # final pad to 4
    pad = align4(offset) - offset
    if pad:
        chunks.append(b"\x00" * pad)
        offset += pad

    buffer_binary = b"".join(chunks)

    glb = GLTF2()

    # Buffer
    buffer = Buffer(byteLength=len(buffer_binary))
    glb.buffers = [buffer]

    # BufferViews
    bv_pos = BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=ARRAY_BUFFER)
    bv_idx = BufferView(buffer=0, byteOffset=idx_off, byteLength=len(idx_bytes), target=ELEMENT_ARRAY_BUFFER)
    bv_emb = BufferView(buffer=0, byteOffset=emb_off, byteLength=len(emb_bytes), target=ARRAY_BUFFER)
    glb.bufferViews = [bv_pos, bv_idx, bv_emb]

    # Accessors
    acc_pos = Accessor(bufferView=0, componentType=FLOAT, count=len(vertices) // 3, type=VEC3)
    acc_idx = Accessor(bufferView=1, componentType=UNSIGNED_SHORT, count=len(indices), type=SCALAR)
    glb.accessors = [acc_pos, acc_idx]

    # Primitive
    prim = Primitive()
    prim.attributes = {"POSITION": 0}
    prim.indices = 1
    prim.mode = 4  # TRIANGLES

    # extras.k3d on the primitive
    k3d = {
        "version": "3.0",
        "memory_realm": "galaxy",
        "client_views": {
            "human": {"render_mode": "pbr"},
            "ai": {"render_mode": "embedding", "direct_buffer_access": True},
        },
        "vectorsView": 0,        # Float32 triples for vertices
        "embeddingsView": 2,     # Full embedding vector
        "embeddingDims": int(embedding.shape[0]),
        "embedding_mappings": {
            "vertex_offset": 0,
            "material_offset": 16,
            "ray_offset": 64,
            "total_dims": int(embedding.shape[0]),
        },
        "rays": rays,
    }
    prim.extras = {"k3d": k3d}

    mesh = Mesh(primitives=[prim])
    node = Node(mesh=0)
    scene = Scene(nodes=[0])
    glb.meshes = [mesh]
    glb.nodes = [node]
    glb.scenes = [scene]
    glb.scene = 0

    # Attach the binary buffer to GLB and save
    glb.set_binary_blob(buffer_binary)
    # Save as GLB
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertices", default="vertex_buffer.bin")
    ap.add_argument("--embedding", default="embedding.npy")
    ap.add_argument("--rays", default="rays.json")
    ap.add_argument("--out", default="hello_tetrahedron.glb")
    args = ap.parse_args()

    verts = np.fromfile(args.vertices, dtype=np.float32, count=12)
    emb = np.load(args.embedding).astype(np.float32)
    try:
        with open(args.rays, "r") as f:
            rays = json.load(f)
    except FileNotFoundError:
        rays = []

    # Tetra indices: 4 faces
    indices = [0, 1, 2,  0, 1, 3,  0, 2, 3,  1, 2, 3]

    build_tetrahedron_glb(verts, indices, emb, rays, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
