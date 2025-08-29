"""
Mark a subset of nodes in a K3D GLB as doors and add a guidance mask.

Actions
- Set metadata[i]["type"] = "door" for selected indices.
- Add/merge ai_state_flags_mask.has_new_information boolean list.
- Optionally, also flag the first neighbor of each door as new information.

Usage
  python -m knowledge3d.tools.mark_doors \
    --input data/ai_books_basic.1k.umap.glb \
    --output data/ai_books_basic.1k.umap.doors.glb \
    --doors 24 --trail true

Notes
- Selection strategy: evenly spaced indices across the dataset.
- This script only edits primitive.extras["k3d"], not geometry/buffers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from pygltflib import GLTF2  # type: ignore


def spaced_indices(n: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= n:
        return list(range(n))
    if count == 1:
        return [0]
    step = (n - 1) / float(count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def mark_doors(input_glb: Path, output_glb: Path, door_count: int, add_trail: bool) -> None:
    g = GLTF2().load(str(input_glb))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids = list(k3d.get("ids", []))
    if not ids:
        raise SystemExit("No k3d.ids found in GLB")
    n = len(ids)
    meta = list(k3d.get("metadata", [{} for _ in range(n)]))
    if len(meta) < n:
        meta += [{} for _ in range(n - len(meta))]
    neighbors = k3d.get("neighbors") or [[] for _ in range(n)]
    id_to_idx = {i: j for j, i in enumerate(ids)}

    door_idxs = spaced_indices(n, door_count)
    # Set metadata.type = 'door'
    for i in door_idxs:
        try:
            entry = meta[i] if isinstance(meta[i], dict) else {}
            entry["type"] = "door"
            meta[i] = entry
        except Exception:
            continue

    # Build guidance mask (has_new_information)
    mask = [False] * n
    for i in door_idxs:
        mask[i] = True
        if add_trail:
            try:
                neigh = neighbors[i]
                if neigh:
                    j = id_to_idx.get(neigh[0])
                    if isinstance(j, int):
                        mask[j] = True
            except Exception:
                pass

    # Merge mask into extras
    flags_mask = k3d.get("ai_state_flags_mask") or {}
    flags_mask["has_new_information"] = mask
    k3d["ai_state_flags_mask"] = flags_mask
    # Announce intended usage to agents
    k3d.setdefault("ai_interaction_protocol", "spatial_reasoning")
    k3d["metadata"] = meta
    prim.extras["k3d"] = k3d

    g.save(str(output_glb))


def main() -> None:
    p = argparse.ArgumentParser(description="Mark doors and guidance mask in a K3D GLB")
    p.add_argument("--input", required=True, help="Input GLB path")
    p.add_argument("--output", required=True, help="Output GLB path")
    p.add_argument("--doors", type=int, default=24, help="Number of doors to mark (evenly spaced)")
    p.add_argument("--trail", type=str, default="true", help="Also flag one neighbor per door as new info (true/false)")
    args = p.parse_args()

    add_trail = str(args.trail).lower() in {"1", "true", "yes", "y"}
    mark_doors(Path(args.input), Path(args.output), args.doors, add_trail)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
