from __future__ import annotations

"""
Add cross‑modal edges to a K3D GLB by linking each node to at least one
neighbor of a different modality (text|image|audio|video), if available.

Heuristic
- Use existing kNN neighbor list in `extras.k3d.neighbors` (as IDs).
- Derive a node's modality set from `metadata.type` and/or presence of
  `metadata.text|image|audio|video` keys.
- For each node, pick the first neighbor whose modality set differs.
- Output an undirected set of edges to `extras.k3d.edges`.

Usage
  python -m knowledge3d.tools.add_crossmodal_edges \
    --input viewer/public/galaxy.glb \
    --out   viewer/public/galaxy.cross.glb
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from pygltflib import GLTF2  # type: ignore


MODAL_KEYS = ("text", "image", "audio", "video")


def modality_set(md: dict) -> Set[str]:
    kinds: Set[str] = set()
    t = str(md.get("type") or "").lower()
    if t in MODAL_KEYS:
        kinds.add(t)
    for k in MODAL_KEYS:
        if md.get(k) is not None:
            kinds.add(k)
    return kinds


def load_k3d(path: str) -> Tuple[List[str], List[dict], List[List[str]]]:
    g = GLTF2().load(path)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    metadata: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    neighbors: List[List[str]] = [list(r) for r in (k3d.get("neighbors", []) or [])]
    return ids, metadata, neighbors


def write_edges(in_path: str, out_path: str, edges: List[Tuple[str, str]]) -> None:
    g = GLTF2().load(in_path)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    k3d["edges"] = edges
    prim.extras["k3d"] = k3d
    # Persist as proper GLB
    g.save_binary(out_path)


def build_cross_edges(ids: List[str], metadata: List[dict], neighbors: List[List[str]]) -> List[Tuple[str, str]]:
    id_to_idx: Dict[str, int] = {ids[i]: i for i in range(len(ids))}
    modal: List[Set[str]] = [modality_set(metadata[i] if i < len(metadata) else {}) for i in range(len(ids))]
    seen: Set[Tuple[str, str]] = set()
    edges: List[Tuple[str, str]] = []
    for i, id_i in enumerate(ids):
        my = modal[i]
        nbrs = neighbors[i] if i < len(neighbors) else []
        target: str | None = None
        for nid in nbrs:
            j = id_to_idx.get(nid)
            if j is None:
                continue
            if modal[j] and modal[j] != my:
                target = ids[j]
                break
        if target:
            a, b = (id_i, target) if id_i <= target else (target, id_i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    return edges


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Add cross‑modal edges to a K3D GLB")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids, meta, neighbors = load_k3d(args.input)
    if not ids:
        raise SystemExit("No K3D ids found in GLB")
    edges = build_cross_edges(ids, meta, neighbors)
    write_edges(args.input, args.out, edges)
    print(f"Wrote {args.out} with {len(edges)} cross‑modal edges")


if __name__ == "__main__":
    main()
