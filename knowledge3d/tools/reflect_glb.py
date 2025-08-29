"""
Offline reflection generator for a K3D GLB.

Computes a concise "agent thoughts" summary from primitive.extras.k3d:
- node count, average degree (out+in), door count, mask ratio, top hubs
Writes a markdown report and prints the reflection line to stdout.

Usage:
  python -m knowledge3d.tools.reflect_glb --gltf viewer/public/ai_books_basic.1k.umap.doors.glb \
    --out docs/reports/reflections
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from pygltflib import GLTF2  # type: ignore


def reflect(gltf_path: Path) -> dict:
    g = GLTF2().load(str(gltf_path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    labels: List[str] = [ (m.get("label") if isinstance(m, dict) else None) for m in k3d.get("metadata", []) ]
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    mask = (k3d.get("ai_state_flags_mask", {}) or {}).get("has_new_information") or []
    n = len(ids)
    out_deg = [ len(neighbors[i]) if i < len(neighbors) else 0 for i in range(n) ]
    idx = {ids[i]: i for i in range(n)}
    in_deg = [0]*n
    for row in neighbors:
        for nid in row:
            j = idx.get(nid)
            if j is not None:
                in_deg[j] += 1
    avg_deg = (sum(out_deg)+sum(in_deg))/(n*2) if n>0 else 0.0
    door_count = 0
    meta = k3d.get("metadata", []) or []
    for m in meta:
        if isinstance(m, dict) and m.get("type") == "door":
            door_count += 1
    mask_true = sum(1 for x in mask if bool(x))
    top = sorted(range(n), key=lambda i: in_deg[i]+out_deg[i], reverse=True)[:5]
    top_labels = [ (labels[i] if i < len(labels) and labels[i] else ids[i]) for i in top ]
    return {
        "nodes": n,
        "avg_degree": avg_deg,
        "doors": door_count,
        "mask_true": mask_true,
        "top_hubs": top_labels,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate reflection for a K3D GLB")
    p.add_argument("--gltf", required=True)
    p.add_argument("--out", required=True, help="Output folder for markdown report")
    args = p.parse_args()
    gltf_path = Path(args.gltf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = reflect(gltf_path)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    reflection = (
        f"Hello partner. I see {stats['nodes']} nodes, avg degree ≈ {stats['avg_degree']:.1f}. "
        + (f"Doors: {stats['doors']}. " if stats['doors'] else "")
        + (f"Guided nodes: {stats['mask_true']}. " if stats['mask_true'] else "")
        + ("Key hubs: " + ", ".join(stats['top_hubs']) + "." if stats['top_hubs'] else "")
    )
    md = [
        f"# K3D Reflection — {ts}",
        "",
        f"- GLB: `{gltf_path}`",
        f"- Nodes: {stats['nodes']} — Avg degree: {stats['avg_degree']:.2f}",
        f"- Doors: {stats['doors']} — Guided nodes: {stats['mask_true']}",
        f"- Top hubs: {', '.join(stats['top_hubs']) if stats['top_hubs'] else '—'}",
        "",
        "## Agent Thoughts",
        reflection,
        "",
    ]
    out_path = out_dir / f"k3d_reflection-{ts}.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(reflection)


if __name__ == "__main__":
    main()

