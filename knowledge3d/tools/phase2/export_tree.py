import struct
from typing import List, Tuple

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness


def _flatten_tree(tree: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten TreeNode structure into arrays.
    Returns: positions (N,3), edges (E,2) u32, embeddings (N,D), similarities (N,)
    """
    positions: List[List[float]] = []
    emb_list: List[np.ndarray] = []
    honesty: List[float] = []
    edges: List[Tuple[int, int]] = []

    def walk(node: dict, parent_idx: int | None = None):
        idx = len(positions)
        positions.append(node.get('position', [0.0, 0.0, 0.0]))
        emb = np.asarray(node.get('embedding', []), dtype=np.float32).reshape(-1)
        emb_list.append(emb)
        honesty.append(float(node.get('honesty', 1.0)))
        if parent_idx is not None:
            edges.append((parent_idx, idx))
        for ch in node.get('children', []) or []:
            walk(ch, idx)

    walk(tree)
    E = np.array(edges, dtype=np.uint32) if edges else np.zeros((0, 2), dtype=np.uint32)
    P = np.array(positions, dtype=np.float32).reshape(-1, 3)
    D = max((len(e) for e in emb_list), default=0)
    if D == 0:
        EMB = np.zeros((len(emb_list), 0), dtype=np.float32)
    else:
        EMB = np.zeros((len(emb_list), D), dtype=np.float32)
        for i, e in enumerate(emb_list):
            EMB[i, : min(len(e), D)] = e[:D]
    # Similarity to root as simple dot of normalized vectors
    sim = np.zeros((len(emb_list),), dtype=np.float32)
    if D > 0:
        root = EMB[0]
        def _norm(x: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(x) + 1e-8)
            return x / n
        r = _norm(root)
        for i in range(len(emb_list)):
            sim[i] = float(np.clip(np.dot(r, _norm(EMB[i])), -1.0, 1.0))
    return P, E, EMB, sim


def exportTreeToGLB(tree: dict, out_path: str, *,
                    domain: str | None = None,
                    tree_id: str | None = None,
                    source_ref: str | None = None,
                    checksum: str | None = None,
                    version: str = '1.0') -> bool:
    P, E, EMB, SIM = _flatten_tree(tree)
    if P.shape[0] == 0:
        return False
    # Build binary chunks
    pos_bytes = P.astype(np.float32).tobytes(order='C')
    idx_bytes = (E.reshape(-1).astype(np.uint32)).tobytes(order='C')
    emb_bytes = EMB.astype(np.float32).tobytes(order='C')
    sim_bytes = SIM.astype(np.float32).tobytes(order='C')

    def align4(n: int) -> int: return (n + 3) & ~3

    chunks: list[bytes] = []
    offset = 0

    pos_off = offset; chunks.append(pos_bytes); offset += len(pos_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad

    idx_off = offset; chunks.append(idx_bytes); offset += len(idx_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad

    emb_off = offset; chunks.append(emb_bytes); offset += len(emb_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad

    sim_off = offset; chunks.append(sim_bytes); offset += len(sim_bytes)
    pad = align4(offset) - offset; chunks.append(b"\x00" * pad); offset += pad

    blob = b"".join(chunks)

    glb = GLTF2()
    glb.buffers = [Buffer(byteLength=len(blob))]
    glb.bufferViews = [
        BufferView(buffer=0, byteOffset=pos_off, byteLength=len(pos_bytes), target=34962),
        BufferView(buffer=0, byteOffset=idx_off, byteLength=len(idx_bytes), target=34963),
        BufferView(buffer=0, byteOffset=emb_off, byteLength=len(emb_bytes), target=34962),
        BufferView(buffer=0, byteOffset=sim_off, byteLength=len(sim_bytes), target=34962),
    ]
    # Accessors
    acc_pos = Accessor(bufferView=0, componentType=5126, count=P.shape[0], type="VEC3")
    acc_idx = Accessor(bufferView=1, componentType=5125, count=E.size, type="SCALAR")
    glb.accessors = [acc_pos, acc_idx]

    # Materials
    # Branch material neutral; leaves will be a Points mesh colored by similarity
    mat_branch = Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.9))
    glb.materials = [mat_branch]

    # Branch mesh as LINES
    prim = Primitive()
    prim.attributes = {"POSITION": 0}
    prim.indices = 1
    prim.mode = 1  # LINES
    prim.material = 0
    prim.extras = {
        "k3d": {
            "version": "3.0",
            "memory_realm": "garden",
            "client_views": {
                "human": {"render_mode": "lines"},
                "ai": {"render_mode": "embedding", "direct_buffer_access": True}
            },
            "vectorsView": 0,
            "embeddingsView": 2,
            "similarityView": 3,
            "embeddingDims": int(EMB.shape[1]),
            "object": {"kind": "tree", "nodes": int(P.shape[0]), "edges": int(E.shape[0])}
        },
        # Tree module metadata (no position here)
        "k3d_garden": {
            'tree_id': tree_id or f"tree_{(domain or 'domain').lower().replace(' ','_')}_001",
            'domain': domain or 'Unknown',
            'version': version,
            'embeddingDims': int(EMB.shape[1]),
            'complexity': int(P.shape[0]),
            'is_chiral': False,
            'checksum': checksum,
            'source_ref': source_ref,
        }
    }
    mesh = Mesh(primitives=[prim])
    glb.meshes = [mesh]
    glb.nodes = [Node(mesh=0, name="KnowledgeTree")]
    glb.scenes = [Scene(nodes=[0])]
    glb.scene = 0
    glb.set_binary_blob(blob)
    glb.save_binary(out_path)
    return True
