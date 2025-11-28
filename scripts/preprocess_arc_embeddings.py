#!/usr/bin/env python3
"""
Precompute ARC grid embeddings in parallel and store for sovereign training.

Outputs a pickled dict mapping grid_hash -> embedding (list of floats).
Use 12 workers by default (Ryzen 6C/12T target).
"""

from __future__ import annotations

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from knowledge3d.training.arc_agi.embedders import MultiModalGridEmbedder


def load_task(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hash_grid(grid) -> int:
    return hash(tuple(tuple(int(c) for c in row) for row in grid))


def preprocess_worker(task_data: Tuple[str, List[List[List[int]]]]) -> Dict[int, List[float]]:
    task_id, grids = task_data
    embedder = MultiModalGridEmbedder(matryoshka_dim=512)
    out: Dict[int, List[float]] = {}
    for idx, grid in enumerate(grids):
        emb = embedder.grid_to_multimodal_embedding(grid)
        out[_hash_grid(grid)] = [float(v) for v in emb]
    return out


def preprocess_all(task_files: List[Path], workers: int = 12) -> Dict[int, List[float]]:
    grids_per_task: List[Tuple[str, List[List[List[int]]]]] = []
    for path in task_files:
        task = load_task(path)
        tid = path.stem
        train = task.get("train", [])
        test = task.get("test", [])
        grids: List[List[List[int]]] = []
        for ex in train:
            grids.append(ex.get("input", []))
            grids.append(ex.get("output", []))
        for ex in test:
            grids.append(ex.get("input", []))
            grids.append(ex.get("output", []))
        grids_per_task.append((tid, grids))

    with Pool(processes=workers) as pool:
        results = pool.map(preprocess_worker, grids_per_task)

    merged: Dict[int, List[float]] = {}
    for res in results:
        merged.update(res)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", type=Path, required=True, help="Task json files.")
    ap.add_argument("--output", type=Path, required=True, help="Output pickle path.")
    ap.add_argument("--workers", type=int, default=12, help="Number of CPU workers.")
    args = ap.parse_args()

    emb = preprocess_all(args.tasks, workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(emb, f)
    print(f"[PREPROCESS] Stored {len(emb)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
