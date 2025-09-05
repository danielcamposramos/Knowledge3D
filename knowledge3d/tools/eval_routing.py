"""
Evaluate routing strategies on a K3D GLB.

Metrics
- success rate, median hops
- avg route time (ms)

Compares
- BFS (baseline)
- A* (positions)
- A* LOD (positions + dynamic neighbor cap)

Usage
  python -m knowledge3d.tools.eval_routing --gltf viewer/public/k3d_foundation.6k.umap.glb \
    --pairs 256 --out docs/reports/status/routing-6k.json

  python -m knowledge3d.tools.eval_routing --gltf ../Knowledge3D.local/datasets/ai_compendium.80k.umap.ivf.glb \
    --pairs 256 --out docs/reports/status/routing-80k.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore
from knowledge3d.spatial.osi import Network3D  # type: ignore


def _load_graph_from_glb(p: Path) -> Tuple[List[str], List[List[str]], List[Tuple[float, float, float]]]:
    g = GLTF2().load(str(p))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids = list(k3d.get("ids", []))
    neighbors = [list(r) for r in k3d.get("neighbors", [])]
    # positions in bufferView 0 (float32)
    bv = g.bufferViews[0]
    blob = g.binary_blob()
    start = bv.byteOffset or 0
    end = start + bv.byteLength
    arr = np.frombuffer(blob[start:end], dtype=np.float32)
    pos = arr.reshape((-1, 3))
    positions = [(float(x), float(y), float(z)) for x, y, z in pos]
    return ids, neighbors, positions


def _summ(paths: List[Optional[List[str]]], times: List[float]) -> Dict[str, Any]:
    hops = [len(p) - 1 for p in paths if p]
    success = sum(1 for p in paths if p)
    med = (sorted(hops)[len(hops)//2] if hops else None)
    avg_ms = (sum(times) / len(times) * 1000.0) if times else 0.0
    return {"success": success, "count": len(paths), "success_rate": (success/len(paths) if paths else 0.0), "median_hops": med, "avg_ms": avg_ms}


def eval_routing(glb_path: Path, pairs: int) -> Dict[str, Any]:
    ids, neighbors, positions = _load_graph_from_glb(glb_path)
    n = len(ids)
    rnd = random.Random(42)
    pairs_idx = [(rnd.randrange(n), rnd.randrange(n)) for _ in range(max(1, pairs))]
    # Build undirected adjacency for fair comparison
    und: List[List[str]] = [[] for _ in range(n)]
    id_to_idx = {ids[i]: i for i in range(n)}
    for i, row in enumerate(neighbors):
        for b in row:
            und[i].append(b)
            j = id_to_idx.get(b)
            if j is not None and ids[i] not in und[j]:
                und[j].append(ids[i])

    # BFS
    bfs_paths: List[Optional[List[str]]] = []
    bfs_times: List[float] = []
    for a, b in pairs_idx:
        start_id = ids[a]; target_id = ids[b]
        t0 = time.perf_counter()
        p = Network3D.route_bfs(ids, und, start_id, target_id)
        bfs_times.append(time.perf_counter() - t0)
        bfs_paths.append(p)

    # A* (positions)
    astar_paths: List[Optional[List[str]]] = []
    astar_times: List[float] = []
    for a, b in pairs_idx:
        start_id = ids[a]; target_id = ids[b]
        t0 = time.perf_counter()
        p = Network3D.route_astar_ex(ids, und, positions, start_id, target_id)
        astar_times.append(time.perf_counter() - t0)
        astar_paths.append(p)

    # A* LOD
    lod_paths: List[Optional[List[str]]] = []
    lod_times: List[float] = []
    for a, b in pairs_idx:
        start_id = ids[a]; target_id = ids[b]
        t0 = time.perf_counter()
        p = Network3D.route_astar_lod(ids, und, positions, start_id, target_id)
        lod_times.append(time.perf_counter() - t0)
        lod_paths.append(p)

    return {
        "n": n,
        "pairs": pairs,
        "bfs": _summ(bfs_paths, bfs_times),
        "astar": _summ(astar_paths, astar_times),
        "astar_lod": _summ(lod_paths, lod_times),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate routing on a K3D GLB")
    p.add_argument("--gltf", required=True)
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = eval_routing(Path(args.gltf), args.pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
