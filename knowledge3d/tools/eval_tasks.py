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
from typing import Any, Dict, List, Optional

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


def load_extras(gltf_path: Path) -> Dict[str, Any]:
    g = GLTF2().load(str(gltf_path))
    prim = g.meshes[0].primitives[0]
    return prim.extras.get("k3d", {})


def generate_tasks(gltf_path: Path, pairs: int, door: int) -> Dict[str, Any]:
    k3d = load_extras(gltf_path)
    ids: List[str] = list(k3d.get("ids", []))
    meta: List[Dict[str, Any]] = [m if isinstance(m, dict) else {} for m in k3d.get("metadata", [])]
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) or ids[i] for i,m in enumerate(meta) ]
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    n = len(ids)
    # random label pairs for goto
    rnd = random.Random(42)
    pairs_out = []
    for _ in range(max(0, pairs)):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a == b:
            b = (b + 1) % n
        start, target = ids[a], ids[b]
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
    args = p.parse_args()
    gltf_path = Path(args.gltf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks(gltf_path, args.pairs, args.door)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
