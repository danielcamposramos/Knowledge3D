from __future__ import annotations

"""
Evaluate cross‑modal homophily on a Galaxy GLB.

Reports:
- average neighbor count
- mean/median homophily (fraction of neighbors with same metadata.type)
- cross‑type edge counts

Usage
  python -m knowledge3d.tools.eval_modal_homophily \
    --gltf viewer/public/galaxy.glb \
    --out docs/reports/status/galaxy_modal_homophily.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


def eval_homophily(glb: Path) -> Dict[str, object]:
    g = GLTF2().load(str(glb))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    meta: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    ids: List[str] = list(k3d.get("ids", []) or [])
    idx = {ids[i]: i for i in range(len(ids))}
    types = [m.get("type") for m in meta]
    same: List[float] = []
    counts: List[int] = []
    cross: Dict[Tuple[str, str], int] = {}
    for i, id_ in enumerate(ids):
        t = types[i]
        row = neighbors[i] if i < len(neighbors) else []
        tot = 0; s = 0
        for nid in row:
            j = idx.get(nid)
            if j is None:
                continue
            tot += 1
            tj = types[j]
            if t and tj == t:
                s += 1
            if t and tj and tj != t:
                a, b = sorted([t, tj])
                cross[(a, b)] = cross.get((a, b), 0) + 1
        if tot > 0:
            same.append(s / tot)
            counts.append(tot)
    out = {
        "nodes": len(ids),
        "avg_neighbors": float(np.mean(counts)) if counts else 0.0,
        "homophily_mean": float(np.mean(same)) if same else 0.0,
        "homophily_median": float(np.median(same)) if same else 0.0,
        "cross_edges": {f"{a}-{b}": int(v) for (a, b), v in cross.items()},
    }
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate modality homophily on Galaxy GLB")
    ap.add_argument("--gltf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = eval_homophily(Path(args.gltf))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

