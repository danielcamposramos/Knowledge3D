import argparse
import struct
from typing import List, Tuple

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
    Material,
    PbrMetallicRoughness,
)

from knowledge3d.cranium.embedding_generator import DynamicEmbeddingGenerator


def book_cuboid(width: float, height: float, depth: float) -> Tuple[np.ndarray, List[int]]:
    hx, hy, hz = width * 0.5, height * 0.5, depth * 0.5
    v = np.array([
        -hx, -hy, -hz,
         hx, -hy, -hz,
         hx,  hy, -hz,
        -hx,  hy, -hz,
        -hx, -hy,  hz,
         hx, -hy,  hz,
         hx,  hy,  hz,
        -hx,  hy,  hz,
    ], dtype=np.float32)
    idx = [
        0, 1, 2, 2, 3, 0,
        4, 6, 5, 6, 4, 7,
        0, 4, 5, 5, 1, 0,
        3, 2, 6, 6, 7, 3,
        1, 5, 6, 6, 2, 1,
        0, 3, 7, 7, 4, 0,
    ]
    return v, idx


def low_discrepancy_positions(n: int, spacing: float, jitter: float) -> List[float]:
    phi = 1.61803398875
    xs = []
    for i in range(n):
        t = (i + 0.5) / n
        ld = (t * phi) % 1.0
        x = (i - (n - 1) * 0.5) * spacing + (ld - 0.5) * jitter * spacing
        xs.append(x)
    return xs


def build_library_glb(titles: List[str], out_path: str, positions: np.ndarray | None = None, books_config: List[dict] | None = None) -> None:
    gen = DynamicEmbeddingGenerator()
    # Geometry params
    width, height, depth = 0.05, 0.30, 0.20
    hy = height * 0.5
    spacing, jitter = 0.08, 0.4
    if positions is not None and positions.shape[0] >= len(titles):
        xs = [float(positions[i, 0]) for i in range(len(titles))]
        zs = [float(positions[i, 2]) for i in range(len(titles))]
    else:
        xs = low_discrepancy_positions(len(titles), spacing, jitter)
        zs = [0.0 for _ in titles]

    # Prepare one big buffer for all views
    chunks: List[bytes] = []
    offset = 0
    def align4(n: int) -> int:
        return (n + 3) & ~3

    buffer_views: List[BufferView] = []
    accessors: List[Accessor] = []
    meshes: List[Mesh] = []
    nodes: List[Node] = []
    materials: List[Material] = []

    for i, title in enumerate(titles):
        if books_config is not None:
            text = str(books_config[i].get('text', title))
            honesty = float(books_config[i].get('honesty', 1.0))
            emb = gen.generate(text, "text")
            if emb.shape[0] > 72:
                emb = emb.copy()
                emb[72] = honesty
        else:
            emb = gen.generate(title, "text")
        pos, idx = book_cuboid(width, height, depth)

        # Translate to shelf position (x, 0, z)
        pos = pos.reshape(-1, 3)
        pos[:, 0] += xs[i]
        pos[:, 2] += zs[i]
        pos = pos.reshape(-1)

        pos_bytes = struct.pack("<" + "f" * len(pos), *pos.tolist())
        # Split indices by region: pages (back/front/top/bottom), cover (+X), spine (-X)
        pages_idx = idx[0:24]
        right_idx = idx[24:30]
        left_idx = idx[30:36]
        idx_pages_bytes = struct.pack("<" + "H" * len(pages_idx), *pages_idx)
        idx_right_bytes = struct.pack("<" + "H" * len(right_idx), *right_idx)
        idx_left_bytes = struct.pack("<" + "H" * len(left_idx), *left_idx)
        emb_bytes = struct.pack("<" + "f" * len(emb), *emb.tolist())

        # positions
        pos_off = offset
        chunks.append(pos_bytes)
        offset += len(pos_bytes)
        pad = align4(offset) - offset
        if pad: chunks.append(b"\x00" * pad); offset += pad
        bv_pos = BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962)
        buffer_views.append(bv_pos)

        # indices: pages
        idx_pages_off = offset
        chunks.append(idx_pages_bytes)
        offset += len(idx_pages_bytes)
        pad = align4(offset) - offset
        if pad: chunks.append(b"\x00" * pad); offset += pad
        bv_idx_pages = BufferView(buffer=0, byteOffset=idx_pages_off, byteLength=len(idx_pages_bytes), target=34963)
        buffer_views.append(bv_idx_pages)

        # indices: right (+X) cover
        idx_right_off = offset
        chunks.append(idx_right_bytes)
        offset += len(idx_right_bytes)
        pad = align4(offset) - offset
        if pad: chunks.append(b"\x00" * pad); offset += pad
        bv_idx_right = BufferView(buffer=0, byteOffset=idx_right_off, byteLength=len(idx_right_bytes), target=34963)
        buffer_views.append(bv_idx_right)

        # indices: left (-X) spine
        idx_left_off = offset
        chunks.append(idx_left_bytes)
        offset += len(idx_left_bytes)
        pad = align4(offset) - offset
        if pad: chunks.append(b"\x00" * pad); offset += pad
        bv_idx_left = BufferView(buffer=0, byteOffset=idx_left_off, byteLength=len(idx_left_bytes), target=34963)
        buffer_views.append(bv_idx_left)

        # embedding
        emb_off = offset
        chunks.append(emb_bytes)
        offset += len(emb_bytes)
        pad = align4(offset) - offset
        if pad: chunks.append(b"\x00" * pad); offset += pad
        bv_emb = BufferView(buffer=0, byteOffset=emb_off, byteLength=len(emb_bytes), target=34962)
        buffer_views.append(bv_emb)

        # Accessors
        acc_pos = Accessor(bufferView=len(buffer_views) - 4, componentType=5126, count=len(pos) // 3, type="VEC3")
        acc_idx_pages = Accessor(bufferView=len(buffer_views) - 3, componentType=5123, count=len(pages_idx), type="SCALAR")
        acc_idx_right = Accessor(bufferView=len(buffer_views) - 2, componentType=5123, count=len(right_idx), type="SCALAR")
        acc_idx_left  = Accessor(bufferView=len(buffer_views) - 1, componentType=5123, count=len(left_idx), type="SCALAR")
        accessors.extend([acc_pos, acc_idx_pages, acc_idx_right, acc_idx_left])

        # Materials per region driven by embedding dims
        def _clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))
        rough = _clamp01(float(abs(emb[24])) if emb.shape[0] > 24 else 0.5)
        metal = _clamp01(float(abs(emb[25])) if emb.shape[0] > 25 else 0.0)
        em = _clamp01(float(emb[26] * 0.5 + 0.5) if emb.shape[0] > 26 else 0.0)
        honesty = float(emb[72]) if emb.shape[0] > 72 else 1.0
        # Honesty overlay tint
        if honesty < 0.0:
            ecol = (min(1.0, 1.0 + honesty), 0.0, 0.0)
        elif honesty < 0.5:
            t = (0.5 - honesty) * 2.0
            ecol = (min(1.0, t), min(1.0, t * 0.5), 0.0)
        else:
            t = (honesty - 0.5) * 2.0
            ecol = (0.0, min(1.0, t), 0.0)
        emissive = (max(em, ecol[0]), max(em, ecol[1]), max(em, ecol[2]))

        mat_pages = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=rough), emissiveFactor=list(emissive))
        mat_cover = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=metal, roughnessFactor=0.5), emissiveFactor=[0.0, 0.0, 0.0])
        mat_spine = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=rough), emissiveFactor=[0.0, 0.0, 0.0])
        materials.extend([mat_pages, mat_cover, mat_spine])
        i_pages = len(materials) - 3
        i_cover = len(materials) - 2
        i_spine = len(materials) - 1

        # Build three primitives sharing the same position accessor
        prim_pages = Primitive()
        prim_pages.attributes = {"POSITION": len(accessors) - 4}
        prim_pages.indices = len(accessors) - 3
        prim_pages.mode = 4
        prim_pages.material = i_pages
        # Simple title ray from top center using a few dims
        def _sig(x: float) -> float:
            import math
            return 1.0 / (1.0 + math.exp(-x))
        ray = []
        if emb.shape[0] >= 73:
            # Top center of the cuboid in local space
            top_y = hy
            start = [float(xs[i]), float(top_y), float(zs[i])]
            end = [start[0] + float(emb[32]) * 0.5, float(top_y + emb[33] * 0.5), start[2] + float(emb[34]) * 0.5]
            thickness = float(abs(emb[35]) * 0.05)
            color = [_sig(float(emb[68])), _sig(float(emb[69])), _sig(float(emb[70]))]
            rtype = 0 if emb[71] > 0.7 else (1 if emb[71] > 0.3 else 2)
            honesty = float(emb[72])
            ray = [{
                "start": start,
                "end": end,
                "thickness": thickness,
                "color": color,
                "type": rtype,
                "honesty": honesty,
            }]
        prim_pages.extras = {
            "k3d": {
                "version": "3.0",
                "memory_realm": "house",
                "client_views": {
                    "human": {"render_mode": "pbr"},
                    "ai": {"render_mode": "embedding", "direct_buffer_access": True},
                },
                "vectorsView": len(buffer_views) - 4,
                "embeddingsView": len(buffer_views) - 1,
                "embeddingDims": int(emb.shape[0]),
                "object": {"kind": "book", "title": title},
                "rays": ray,
            }
        }
        prim_cover = Primitive()
        prim_cover.attributes = {"POSITION": len(accessors) - 4}
        prim_cover.indices = len(accessors) - 2
        prim_cover.mode = 4
        prim_cover.material = i_cover
        prim_cover.extras = prim_pages.extras

        prim_spine = Primitive()
        prim_spine.attributes = {"POSITION": len(accessors) - 4}
        prim_spine.indices = len(accessors) - 1
        prim_spine.mode = 4
        prim_spine.material = i_spine
        prim_spine.extras = prim_pages.extras

        mesh = Mesh(primitives=[prim_pages, prim_cover, prim_spine])
        meshes.append(mesh)
        node = Node(mesh=len(meshes) - 1, name=str(title))
        nodes.append(node)

    blob = b"".join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = buffer_views
    glb.accessors = accessors
    glb.meshes = meshes
    glb.nodes = nodes
    glb.scenes = [Scene(nodes=list(range(len(nodes))))]
    glb.scene = 0
    glb.materials = materials
    glb.set_binary_blob(blob)
    glb.save_binary(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="library_room.glb")
    ap.add_argument("--books", default=None, help="JSON list: [{title,text,honesty}]")
    ap.add_argument("--positions", default=None, help="Float32 bin (x,y,z)*N")
    ap.add_argument("titles", nargs="*", default=[
        "The Geometry of Meaning",
        "Favela Protocol",
        "RLWHF — Honest Systems",
        "Galaxy & House",
    ])
    args = ap.parse_args()
    books_cfg = None
    titles = args.titles
    if args.books:
        import json
        with open(args.books, 'r') as f:
            books_cfg = json.load(f)
        titles = [str(b.get('title', 'Untitled')) for b in books_cfg]
    pos = None
    if args.positions and Path(args.positions).exists():
        pos = np.fromfile(args.positions, dtype=np.float32)
        if pos.size % 3 == 0:
            pos = pos.reshape(-1, 3)
        else:
            pos = None
    build_library_glb(titles, args.out, positions=pos, books_config=books_cfg)
    print(f"Wrote {args.out} with {len(titles)} books")


if __name__ == "__main__":
    main()
