"""
Place doors based on hub centrality (approx via in-degree and node density) for a K3D GLB.

Strategy
- Compute in-degree for each node (how many neighbor lists reference it)
- Score nodes by in-degree; optionally boost nodes that are 'hubs' from a provided overlay (optional)
- Select top door_count nodes as doors; update metadata and has_new_information mask with simple trail

Usage:
  python3 -m knowledge3d.tools.place_doors_hubs \
    --input ../Knowledge3D.local/datasets/ai_compendium.180k.pca.aug.glb \
    --out   ../Knowledge3D.local/datasets/ai_compendium.180k.pca.aug.doors.glb \
    --doors 2880 --trail true
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Any

from pygltflib import GLTF2  # type: ignore


def place_doors(input_glb: str, output_glb: str, door_count: int, add_trail: bool) -> None:
    g = GLTF2().load(input_glb)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    n = len(ids)
    # Init metadata
    meta: List[Dict[str, Any]] = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [{} for _ in range(n)])]
    if len(meta) < n:
        meta += [{} for _ in range(n - len(meta))]
    # In-degree
    id_to_idx = {ids[i]: i for i in range(n)}
    indeg = [0] * n
    for i in range(n):
        for nb in neighbors[i]:
            j = id_to_idx.get(nb)
            if j is not None:
                indeg[j] += 1
    # Score = indegree
    ranked = sorted(range(n), key=lambda i: indeg[i], reverse=True)
    door_idxs = ranked[: min(door_count, n)]
    # Set doors
    for i in door_idxs:
        e = meta[i] if isinstance(meta[i], dict) else {}
        e["type"] = "door"
        meta[i] = e
    # Guidance mask
    mask = [False] * n
    for i in door_idxs:
        mask[i] = True
        if add_trail:
            nb = neighbors[i]
            if nb:
                j = id_to_idx.get(nb[0])
                if isinstance(j, int):
                    mask[j] = True
    flags_mask = k3d.get("ai_state_flags_mask") or {}
    flags_mask["has_new_information"] = mask
    k3d["ai_state_flags_mask"] = flags_mask
    k3d["metadata"] = meta
    prim.extras["k3d"] = k3d
    g.save(output_glb)
    print(f"Wrote {output_glb}")


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Place doors based on hub centrality (approx)")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--doors", type=int, default=256)
    p.add_argument("--trail", type=str, default="true")
    args = p.parse_args()
    add_trail = str(args.trail).lower() in {"1", "true", "yes", "y"}
    place_doors(args.input, args.out, args.doors, add_trail)


if __name__ == "__main__":
    main()

