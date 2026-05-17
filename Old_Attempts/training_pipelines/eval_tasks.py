"""
Generate simple evaluation tasks from a K3D GLB.

Outputs JSON with tasks:
- goto_tasks: random label pairs with BFS baseline hop count
- door_tasks: random source labels and target door labels (if doors exist)

Usage:
  python -m knowledge3d.tools.eval_tasks --gltf viewer/public/ai_books_basic.1k.umap.doors.glb \
    --out data/evals/ai_books_1k_tasks.json --pairs 32 --door 16
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pygltflib import GLTF2  # type: ignore


def bfs_route(ids: List[str], neighbors: List[List[str]], start_id: str, target_id: str) -> Optional[List[str]]:
    if start_id == target_id:
        return [start_id]
    # Build undirected adjacency for robustness
    idx = {ids[i]: i for i in range(len(ids))}
    adj: Dict[str, List[str]] = {i: [] for i in ids}  # type: ignore
    for i, row in enumerate(neighbors):
        a = ids[i]
        for b in row:
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    q: List[List[str]] = [[start_id]]
    seen = {start_id}
    while q:
        path = q.pop(0)
        last = path[-1]
        for nid in adj.get(last, []):
            if nid in seen:
                continue
            seen.add(nid)
            new = path + [nid]
            if nid == target_id:
                return new
            q.append(new)
    return None


def dijkstra_route(ids: List[str], neighbors: List[List[str]], start_id: str, target_id: str) -> Optional[List[str]]:
    """Unweighted Dijkstra equals BFS in hop count; kept for API completeness."""
    # Reuse BFS implementation for now
    return bfs_route(ids, neighbors, start_id, target_id)


def extract_positions(g) -> Optional[List[Tuple[float, float, float]]]:
    try:
        import numpy as np  # type: ignore
        bv = g.bufferViews[0]
        blob = g.binary_blob()
        start = bv.byteOffset or 0
        end = start + bv.byteLength
        arr = np.frombuffer(blob[start:end], dtype=np.float32)
        pos = arr.reshape((-1, 3))
        return [(float(x), float(y), float(z)) for x, y, z in pos]
    except Exception:
        return None


def astar_route(ids: List[str], neighbors: List[List[str]], positions: List[Tuple[float, float, float]], start_id: str, target_id: str) -> Optional[List[str]]:
    import heapq
    id_to_idx = {ids[i]: i for i in range(len(ids))}
    si = id_to_idx.get(start_id); ti = id_to_idx.get(target_id)
    if si is None or ti is None:
        return None
    # Compute min observed edge length on a small sample to keep heuristic admissible
    min_edge = float('inf')
    for i in range(0, min(10000, len(ids)), max(1, len(ids)//10000)):
        pi = positions[i]
        for nb in neighbors[i]:
            j = id_to_idx.get(nb)
            if j is None: continue
            pj = positions[j]
            d = ((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2) ** 0.5
            if d > 0 and d < min_edge:
                min_edge = d
    if not (min_edge > 0 and min_edge < float('inf')):
        min_edge = 1.0
    def h(i: int, j: int) -> float:
        pi = positions[i]; pj = positions[j]
        d = ((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2) ** 0.5
        return d / min_edge
    openq = []
    heapq.heappush(openq, (h(si, ti), 0.0, si, None))
    came: Dict[int, Optional[int]] = {}
    gscore: Dict[int, float] = {si: 0.0}
    visited = set()
    while openq:
        f, g, i, parent = heapq.heappop(openq)
        if i in visited:
            continue
        visited.add(i)
        came[i] = parent
        if i == ti:
            # reconstruct
            path_idx = []
            cur = i
            while cur is not None:
                path_idx.append(cur)
                cur = came.get(cur)
            path_idx.reverse()
            return [ids[k] for k in path_idx]
        for nb in neighbors[i]:
            j = id_to_idx.get(nb)
            if j is None or j in visited:
                continue
            ng = g + 1.0
            if ng < gscore.get(j, 1e18):
                gscore[j] = ng
                heapq.heappush(openq, (ng + h(j, ti), ng, j, i))
    return None


def load_extras_and_gltf(gltf_path: Path) -> Tuple[Dict[str, Any], Any]:
    g = GLTF2().load(str(gltf_path))
    prim = g.meshes[0].primitives[0]
    return prim.extras.get("k3d", {}), g


def generate_tasks(gltf_path: Path, pairs: int, door: int, router: str = "bfs") -> Dict[str, Any]:
    k3d, g = load_extras_and_gltf(gltf_path)
    ids: List[str] = list(k3d.get("ids", []))
    meta: List[Dict[str, Any]] = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i,m in enumerate(meta) ]
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    positions = extract_positions(g)
    n = len(ids)
    # random label pairs for goto
    rnd = random.Random(42)
    pairs_out = []
    for _ in range(max(0, pairs)):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a == b:
            b = (b + 1) % n
        start, target = ids[a], ids[b]
        if router == "astar" and positions is not None:
            p = astar_route(ids, neighbors, positions, start, target)
            if p is None:  # fallback
                p = bfs_route(ids, neighbors, start, target)
        elif router == "dijkstra":
            p = dijkstra_route(ids, neighbors, start, target)
        else:
            p = bfs_route(ids, neighbors, start, target)
        pairs_out.append({
            "from": labels[a],
            "to": labels[b],
            "path_len": (len(p) - 1) if p else None,
            "exists": p is not None,
        })
    # door tasks
    door_idxs = [i for i,m in enumerate(meta) if m.get("type") == "door"]
    door_out = []
    for _ in range(max(0, door)):
        if not door_idxs:
            break
        src = rnd.randrange(n)
        di = rnd.choice(door_idxs)
        if router == "astar" and positions is not None:
            p = astar_route(ids, neighbors, positions, ids[src], ids[di])
            if p is None:
                p = bfs_route(ids, neighbors, ids[src], ids[di])
        elif router == "dijkstra":
            p = dijkstra_route(ids, neighbors, ids[src], ids[di])
        else:
            p = bfs_route(ids, neighbors, ids[src], ids[di])
        door_out.append({
            "from": labels[src],
            "door": labels[di],
            "path_len": (len(p) - 1) if p else None,
            "exists": p is not None,
        })
    return {"goto_tasks": pairs_out, "door_tasks": door_out}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate evaluation tasks from a K3D GLB")
    p.add_argument("--gltf", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=32)
    p.add_argument("--door", type=int, default=16)
    p.add_argument("--router", choices=["bfs", "astar", "dijkstra"], default="bfs")
    args = p.parse_args()
    gltf_path = Path(args.gltf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(gltf_path, args.pairs, args.door, router=args.router)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
