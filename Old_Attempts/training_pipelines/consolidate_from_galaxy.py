from __future__ import annotations

"""
Sleep-time consolidation: pick top hubs from Galaxy and materialize them
as objects inside the House (memory_house.glb).

Usage
  python -m knowledge3d.tools.consolidate_from_galaxy \
    --gltf viewer/public/galaxy.glb \
    --top 24 \
    --out viewer/public/memory_house.glb
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from pygltflib import GLTF2  # type: ignore

from .house_memory import MemoryHouse  # type: ignore


def top_hubs(glb: Path, n: int) -> List[Tuple[str, int]]:
    g = GLTF2().load(str(glb))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    meta: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i,m in enumerate(meta) ]
    deg = [len(neighbors[i]) if i < len(neighbors) else 0 for i in range(len(ids))]
    order = sorted(range(len(ids)), key=lambda i: deg[i], reverse=True)[:n]
    return [(labels[i], deg[i]) for i in order]


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Consolidate top hubs from Galaxy into House objects")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    glb = Path(args.gltf)
    labels = [lab for lab, _ in top_hubs(glb, int(args.top))]
    house = MemoryHouse()
    house.bootstrap_defaults()
    room = "Study"
    for lab in labels:
        house.add_object(room, lab, text="consolidated from galaxy", kind="object")
    house.export_gltf(Path(args.out))
    print(f"Consolidated {len(labels)} hubs into {args.out}")


if __name__ == "__main__":
    main()

