"""
Augment a K3D GLB neighbor graph with a voxel-based hub overlay to improve global connectivity.

Approach (fast, scalable):
- Extract 3D positions (bufferView 0) and ids + neighbors from primitive.extras.k3d
- Build a 3D grid (voxels) targeting ~H hubs; choose one representative (hub) per non-empty cell
- For each node, add a neighbor link to its cell's hub (if not already present)
- Connect neighboring hubs (6-neighborhood in grid) to form a backbone mesh
- Save to a new GLB with updated extras.k3d.neighbors

Usage:
  python3 -m knowledge3d.tools.augment_neighbors \
    --input ../Knowledge3D.local/datasets/ai_compendium.180k.pca.glb \
    --out   ../Knowledge3D.local/datasets/ai_compendium.180k.pca.aug.glb \
    --hubs 2000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore


@dataclass
class GLBData:
    ids: List[str]
    neighbors: List[List[str]]
    pos: np.ndarray  # (N,3) float32


def load_glb_positions_ids_neighbors(path: str) -> GLBData:
    g = GLTF2().load(path)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []))
    neighbors: List[List[str]] = k3d.get("neighbors", []) or []
    # positions in bufferView 0
    bv = g.bufferViews[0]
    blob = g.binary_blob()
    start = bv.byteOffset or 0
    end = start + bv.byteLength
    arr = np.frombuffer(blob[start:end], dtype=np.float32)
    pos = arr.reshape((-1, 3)).copy()
    return GLBData(ids=ids, neighbors=[list(row) for row in neighbors], pos=pos)


def write_neighbors_to_glb(in_path: str, out_path: str, new_neighbors: List[List[str]]) -> None:
    g = GLTF2().load(in_path)
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    k3d["neighbors"] = new_neighbors
    prim.extras["k3d"] = k3d
    g.save_binary(out_path)


def build_grid_hubs(pos: np.ndarray, target_hubs: int) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int], Dict[int, Tuple[int, int, int]]]:
    # Determine grid dimensions proportional to bbox size such that cells ≈ target_hubs
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    vol = spans[0] * spans[1] * spans[2]
    # Aim for ~cubic cells; let nx*ny*nz ≈ target_hubs
    base = int(round(target_hubs ** (1.0 / 3.0)))
    # Scale by axis spans
    ratios = spans / spans.max()
    nx = max(4, int(round(base * ratios[0])))
    ny = max(4, int(round(base * ratios[1])))
    nz = max(4, int(round(base * ratios[2])))
    # Prevent zero
    nx = max(nx, 1); ny = max(ny, 1); nz = max(nz, 1)
    # Map node -> cell
    gx = np.clip(((pos[:, 0] - mins[0]) / spans[0] * nx).astype(int), 0, nx - 1)
    gy = np.clip(((pos[:, 1] - mins[1]) / spans[1] * ny).astype(int), 0, ny - 1)
    gz = np.clip(((pos[:, 2] - mins[2]) / spans[2] * nz).astype(int), 0, nz - 1)
    # Choose representative hub per cell as first index encountered
    reps: Dict[Tuple[int, int, int], int] = {}
    for i in range(pos.shape[0]):
        cell = (int(gx[i]), int(gy[i]), int(gz[i]))
        if cell not in reps:
            reps[cell] = i
    # Invert map: hub_index -> cell
    hub_to_cell = {idx: cell for cell, idx in reps.items()}
    return (np.array([nx, ny, nz], dtype=int), reps, hub_to_cell)


def augment_neighbors(ids: List[str], neighbors: List[List[str]], pos: np.ndarray, target_hubs: int = 2000) -> List[List[str]]:
    grid_shape, cell_rep, hub_to_cell = build_grid_hubs(pos, target_hubs)
    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    # Assign each node to cell hub
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    gx = np.clip(((pos[:, 0] - mins[0]) / spans[0] * nx).astype(int), 0, nx - 1)
    gy = np.clip(((pos[:, 1] - mins[1]) / spans[1] * ny).astype(int), 0, ny - 1)
    gz = np.clip(((pos[:, 2] - mins[2]) / spans[2] * nz).astype(int), 0, nz - 1)
    id_by_index = ids
    id_to_index = {ids[i]: i for i in range(len(ids))}
    hubs_idx = set(cell_rep.values())
    hubs_ids = {i: id_by_index[i] for i in hubs_idx}
    # Add node -> hub link (directed; eval builds undirected later)
    new_neighbors: List[List[str]] = []
    for i in range(len(ids)):
        row = list(neighbors[i]) if i < len(neighbors) else []
        cell = (int(gx[i]), int(gy[i]), int(gz[i]))
        hub_i = cell_rep.get(cell)
        if hub_i is not None:
            hub_id = id_by_index[hub_i]
            if hub_id not in row and hub_id != id_by_index[i]:
                row.append(hub_id)
        new_neighbors.append(row)
    # Connect neighboring hubs (6-neighborhood)
    # Build a quick set for existing hub links to avoid dupes
    hub_links: Dict[int, set] = {i: set() for i in hubs_idx}
    for idx in hubs_idx:
        # get cell
        cell = hub_to_cell[idx]
        x, y, z = cell
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx_c, ny_c, nz_c = x+dx, y+dy, z+dz
            if 0 <= nx_c < nx and 0 <= ny_c < ny and 0 <= nz_c < nz:
                nbr = cell_rep.get((nx_c, ny_c, nz_c))
                if nbr is not None:
                    # add idx -> nbr
                    hub_links[idx].add(nbr)
    # Write hub edges into neighbor lists
    for a, nbrs in hub_links.items():
        a_id = id_by_index[a]
        row = new_neighbors[a]
        for b in nbrs:
            b_id = id_by_index[b]
            if b_id not in row:
                row.append(b_id)
    return new_neighbors


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Augment K3D neighbors with voxel-based hub overlay")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hubs", type=int, default=2000)
    args = ap.parse_args()
    data = load_glb_positions_ids_neighbors(args.input)
    new_nbrs = augment_neighbors(data.ids, data.neighbors, data.pos, target_hubs=args.hubs)
    write_neighbors_to_glb(args.input, args.out, new_nbrs)
    print(f"Wrote {args.out} (augmented neighbors; hubs~{args.hubs})")


if __name__ == "__main__":
    main()
