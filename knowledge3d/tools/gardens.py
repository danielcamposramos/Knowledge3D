"""
Build a Knowledge Garden GLB: ontology as trees with edges and greenhouse layout.

Design
- Hierarchical topics (roots -> branches -> leaves) with parent→child edges.
- 3D layout is deterministic: y by depth; x/z by sibling spread per tree plot.
- Embeddings: HashingVectorizer over full path strings, L2-normalized.
- Neighbors: small undirected subset for navigability (parent + first children).

Output
- viewer/public/knowledge_garden.glb by default (override with --gltf).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
from sklearn.preprocessing import normalize  # type: ignore

from k3dgen.__main__ import create_gltf_file
from pygltflib import GLTF2  # type: ignore
import struct
import math


@dataclass
class Node:
    id: str
    label: str
    path: str
    depth: int
    parent: str | None


def build_catalog() -> Dict[str, List[str]]:
    return {
        "AI": [
            "Machine Learning/Supervised Learning",
            "Machine Learning/Unsupervised Learning",
            "Machine Learning/Reinforcement Learning",
            "Neural Networks/CNN",
            "Neural Networks/RNN",
            "Neural Networks/Transformer",
            "NLP/Tokenization",
            "NLP/Embeddings",
            "NLP/RAG",
        ],
        "Mathematics": [
            "Algebra/Linear Algebra/Vectors",
            "Algebra/Linear Algebra/Matrices",
            "Algebra/Abstract Algebra/Groups",
            "Calculus/Differential",
            "Calculus/Integral",
            "Probability/Bayes",
            "Probability/Distributions",
        ],
        "Physics": [
            "Classical Mechanics",
            "Electromagnetism",
            "Quantum Mechanics/Qubits",
            "Quantum Mechanics/Entanglement",
            "Thermodynamics",
        ],
    }


def flatten(catalog: Dict[str, List[str]]) -> Tuple[List[Node], List[Tuple[str, str]]]:
    nodes: Dict[str, Node] = {}
    edges: List[Tuple[str, str]] = []
    for i, (root, branches) in enumerate(catalog.items()):
        rid = f"R:{root}"
        nodes[rid] = Node(id=rid, label=root, path=root, depth=0, parent=None)
        for spec in branches:
            parts = [p.strip() for p in spec.split("/") if p.strip()]
            parent = rid
            path = root
            for d, part in enumerate(parts, start=1):
                nid = f"{root}:{'/'.join(parts[:d])}"
                if nid not in nodes:
                    nodes[nid] = Node(id=nid, label=part, path=f"{path}/{part}", depth=d, parent=parent)
                    edges.append((parent, nid))
                parent = nid
                path = nodes[nid].path
    return list(nodes.values()), edges


def load_paths_file(path_file: str) -> Dict[str, List[str]]:
    """Load hierarchy from a simple paths file.

    Each line is a path like: Root/Branch/Leaf
    Returns a catalog mapping root -> list of full paths under that root.
    """
    p = Path(path_file)
    if not p.exists():
        raise FileNotFoundError(path_file)
    roots: Dict[str, List[str]] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split("/") if x.strip()]
        if not parts:
            continue
        root = parts[0]
        spec = "/".join(parts[1:]) if len(parts) > 1 else ""
        if spec:
            roots.setdefault(root, []).append(spec)
        else:
            roots.setdefault(root, [])
    return roots


def layout(nodes: List[Node]) -> Dict[str, Tuple[float, float, float]]:
    # Assign each root a plot along X; place children radially by sibling index; Y by depth
    roots = [n for n in nodes if n.depth == 0]
    root_x = {r.id: (i - (len(roots) - 1) / 2.0) * 6.0 for i, r in enumerate(roots)}
    # Group children per parent
    kids: Dict[str, List[Node]] = {}
    for n in nodes:
        if n.parent:
            kids.setdefault(n.parent, []).append(n)
    pos: Dict[str, Tuple[float, float, float]] = {}
    for r in roots:
        pos[r.id] = (root_x[r.id], 0.0, 0.0)
    # BFS
    queue: List[Node] = roots[:]
    while queue:
        cur = queue.pop(0)
        children = kids.get(cur.id, [])
        for idx, ch in enumerate(children):
            px, py, pz = pos[cur.id]
            spread = max(1.5, 4.0 / (ch.depth))
            angle = (idx - (len(children) - 1) / 2.0) * 0.6
            x = px + np.cos(angle) * spread
            y = py + 0.8  # upward growth per depth
            z = pz + np.sin(angle) * spread * 0.5
            pos[ch.id] = (float(x), float(y), float(z))
            queue.append(ch)
    return pos


def vectorize(nodes: List[Node], dims: int = 128) -> np.ndarray:
    vec = HashingVectorizer(n_features=dims, alternate_sign=False, norm="l2")
    texts = [f"{n.path} — {n.label}" for n in nodes]
    X = vec.transform(texts)
    dense = X.astype(np.float32).toarray()
    dense = normalize(dense, norm="l2")
    return dense


def build_neighbors(ids: List[str], edges: List[Tuple[str, str]], k: int = 3) -> np.ndarray:
    idx = {i: j for j, i in enumerate(ids)}
    adj: Dict[str, List[str]] = {i: [] for i in ids}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].append(b)
            adj[b].append(a)
    # include parent (first) and up to k-1 children
    arr = np.zeros((len(ids), k), dtype=int)
    for i, id_ in enumerate(ids):
        nbrs = adj.get(id_, [])[:k]
        if not nbrs:
            arr[i, :] = i
        else:
            for j in range(k):
                arr[i, j] = idx.get(nbrs[j], i) if j < len(nbrs) else i
    return arr


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build Knowledge Garden GLB (ontology trees or from Galaxy)")
    ap.add_argument("--gltf", default="viewer/public/knowledge_garden.glb")
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--paths", help="Optional text file with Root/Branch/Leaf lines to define the garden")
    ap.add_argument("--from-galaxy", dest="from_galaxy", help="Build garden from a Galaxy GLB (vectors+neighbors)")
    ap.add_argument("--seeds", type=int, default=3, help="Top hubs to seed trees from")
    ap.add_argument("--attract", type=int, default=1500, help="Max attraction points per seed")
    ap.add_argument("--radius", type=float, default=1.2, help="Colonization influence radius (multiples of local step)")
    ap.add_argument("--step", type=float, default=0.35, help="Colonization step scale (relative to local spacing)")
    args = ap.parse_args()

    if args.from_galaxy:
        ids, labels, positions, neighbors = _load_galaxy(args.from_galaxy)
        seed_ids = _top_hubs(ids, neighbors, args.seeds)
        gid_to_idx = {ids[i]: i for i in range(len(ids))}
        all_nodes: List[Tuple[str, Tuple[float,float,float], dict]] = []
        all_edges: List[Tuple[str,str]] = []
        for sid in seed_ids:
            root_pos = positions[gid_to_idx[sid]]
            lab = (labels[gid_to_idx[sid]] if gid_to_idx.get(sid) is not None else sid)
            # pick attraction points nearest to seed
            dists = [(_dist3(root_pos, positions[i]), i) for i in range(len(ids))]
            dists.sort(key=lambda t: t[0])
            # skip self; sample up to attract points
            near_idx = [i for (_,i) in dists[1:args.attract+1]]
            A = [positions[i] for i in near_idx]
            nodes, edges = _colonize(root_pos, A, step_scale=args.step, radius_scale=args.radius)
            # label and collect
            node_ids = []
            for k,(px,py,pz) in enumerate(nodes):
                nid = f"tree:{sid}:{k}"
                node_ids.append(nid)
                md = {"label": (lab if k==0 else f"branch-{k}"), "type": ("root" if k==0 else ("leaf" if k in _leaf_indices(edges) else "branch")), "seed": sid}
                all_nodes.append((nid, (px,py,pz), md))
            for (a,b) in edges:
                all_edges.append((node_ids[a], node_ids[b]))
        # assemble arrays
        out_ids = [nid for (nid,_,_) in all_nodes]
        out_points = np.array([pos for (_,pos,_) in all_nodes], dtype=np.float32)
        # deterministic tiny embeddings (zeros)
        out_emb = np.zeros((len(out_ids), 32), dtype=np.float32)
        # neighbors from edges
        nbr = _neighbors_from_edges(out_ids, all_edges)
        meta = [md for (_,_,md) in all_nodes]
        create_gltf_file(
            args.gltf,
            out_ids,
            out_points,
            out_emb,
            nbr,
            [m.get('label','') for m in meta],
            metadata_texts=None,
            metadata_override=meta,
            fmt="glb",
            emb_precision="f16",
            ai_protocol="spatial_reasoning",
            ai_flags={"is_traversable": True},
            ai_flags_mask=None,
            edges=all_edges,
        )
        print(f"[garden] from galaxy -> {args.gltf} nodes={len(out_ids)} edges={len(all_edges)} seeds={len(seed_ids)}")
        return

    # Legacy ontology mode (paths or default catalog)
    catalog = load_paths_file(args.paths) if args.paths else build_catalog()
    nodes, edges = flatten(catalog)
    pos = layout(nodes)
    ids = [n.id for n in nodes]
    labels = [n.label for n in nodes]
    meta = [
        {
            "label": n.label,
            "type": ("root" if n.depth == 0 else ("branch" if n.depth <= 2 else "leaf")),
            "path": n.path,
        }
        for n in nodes
    ]
    embeddings = vectorize(nodes, dims=args.dims)
    points = np.array([pos[i] for i in ids], dtype=np.float32)
    nbr_idx = build_neighbors(ids, edges, k=3)
    create_gltf_file(
        args.gltf,
        ids,
        points,
        embeddings,
        nbr_idx,
        labels,
        metadata_texts=None,
        metadata_override=meta,
        fmt="glb",
        emb_precision="f16",
        ai_protocol="spatial_reasoning",
        ai_flags={"is_traversable": True},
        ai_flags_mask=None,
        edges=edges,
    )
    print(f"Wrote {args.gltf} with {len(ids)} nodes and {len(edges)} edges")

def _dist3(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def _leaf_indices(edges: List[Tuple[int,int]]) -> set:
    deg = {}
    for a,b in edges:
        deg[a] = deg.get(a,0)+1
        deg[b] = deg.get(b,0)+1
    return {i for i,c in deg.items() if c == 1}

def _neighbors_from_edges(ids: List[str], edges: List[Tuple[str,str]]) -> np.ndarray:
    idx = {ids[i]: i for i in range(len(ids))}
    adj = [[ ] for _ in range(len(ids))]
    for a,b in edges:
        ia = idx.get(a); ib = idx.get(b)
        if ia is None or ib is None: continue
        adj[ia].append(ib); adj[ib].append(ia)
    # Each row: up to 3 neighbors; pad with self
    out = np.zeros((len(ids), 3), dtype=int)
    for i in range(len(ids)):
        row = (adj[i] + [i]*3)[:3]
        out[i,:] = row
    return out

def _top_hubs(ids: List[str], neighbors: List[List[str]], k: int) -> List[str]:
    deg = [(len(neighbors[i]) if i < len(neighbors) else 0, ids[i]) for i in range(len(ids))]
    deg.sort(reverse=True)
    return [ids for (_,ids) in deg[:max(1,k)]]

def _load_galaxy(glb_path: str) -> Tuple[List[str], List[str], List[Tuple[float,float,float]], List[List[str]]]:
    g = GLTF2().load(glb_path)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    labels: List[str] = []
    meta = k3d.get("metadata", []) or []
    for i in range(len(ids)):
        m = meta[i] if i < len(meta) else {}
        labels.append((m.get("label") if isinstance(m, dict) else None) or ids[i])
    neighbors: List[List[str]] = [list(r) for r in (k3d.get("neighbors", []) or [])]
    # vectorsView -> positions
    vview = int(k3d.get("vectorsView", 0))
    bv = g.bufferViews[vview]
    byte_offset = int(bv.byteOffset or 0)
    byte_length = int(bv.byteLength or 0)
    blob = g.binary_blob()
    chunk = blob[byte_offset: byte_offset+byte_length]
    # interpret as float32 triplets
    N = byte_length // 12
    positions: List[Tuple[float,float,float]] = []
    for i in range(N):
        x,y,z = struct.unpack_from('<fff', chunk, i*12)
        positions.append((float(x), float(y), float(z)))
    return ids, labels, positions, neighbors

def _colonize(root: Tuple[float,float,float], attract: List[Tuple[float,float,float]], step_scale: float = 0.35, radius_scale: float = 1.2, max_iter: int = 500) -> Tuple[List[Tuple[float,float,float]], List[Tuple[int,int]]]:
    # Estimate local spacing
    if not attract:
        return [root], []
    import random
    sample = attract[:256]
    dmed = sorted([_dist3(root, p) for p in sample])[min(len(sample)-1, len(sample)//2)] if sample else 1.0
    step = max(1e-3, dmed * step_scale)
    R = max(step*1.1, dmed * radius_scale)
    kill = max(step*0.9, R * 0.35)
    nodes = [root]
    tips = [0]
    edges: List[Tuple[int,int]] = []
    A = list(attract)
    for it in range(max_iter):
        if not A:
            break
        # Assign attraction points to nearest tip within R
        assigned: Dict[int, List[int]] = {}
        for j, ap in enumerate(A):
            best_i = -1; best_d = 1e18
            for ti in tips:
                d = _dist3(nodes[ti], ap)
                if d < best_d:
                    best_d = d; best_i = ti
            if best_i >= 0 and best_d <= R:
                assigned.setdefault(best_i, []).append(j)
        if not assigned:
            break
        new_tips: List[int] = []
        new_nodes: List[Tuple[int, Tuple[float,float,float]]] = []
        # Grow each assigned tip
        for ti, idxs in assigned.items():
            if not idxs:
                continue
            # average direction
            sx=0.0; sy=0.0; sz=0.0
            for j in idxs:
                ax,ay,az = A[j]
                bx,by,bz = nodes[ti]
                vx,vy,vz = (ax-bx, ay-by, az-bz)
                norm = math.sqrt(vx*vx+vy*vy+vz*vz)+1e-9
                sx += vx/norm; sy += vy/norm; sz += vz/norm
            dn = math.sqrt(sx*sx+sy*sy+sz*sz)+1e-9
            dx,dy,dz = (sx/dn, sy/dn, sz/dn)
            nx,ny,nz = (nodes[ti][0]+dx*step, nodes[ti][1]+dy*step, nodes[ti][2]+dz*step)
            new_index = len(nodes)
            nodes.append((nx,ny,nz))
            edges.append((ti, new_index))
            new_tips.append(new_index)
        tips = new_tips
        # Remove attraction points near any new node
        kept: List[Tuple[float,float,float]] = []
        for ap in A:
            too_close = False
            for idx in tips:
                if _dist3(ap, nodes[idx]) <= kill:
                    too_close = True; break
            if not too_close:
                kept.append(ap)
        A = kept
    return nodes, edges


if __name__ == "__main__":
    main()
