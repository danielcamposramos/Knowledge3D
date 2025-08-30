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
    ap = argparse.ArgumentParser(description="Build Knowledge Garden GLB (ontology trees)")
    ap.add_argument("--gltf", default="viewer/public/knowledge_garden.glb")
    ap.add_argument("--dims", type=int, default=128)
    args = ap.parse_args()
    catalog = build_catalog()
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


if __name__ == "__main__":
    main()
