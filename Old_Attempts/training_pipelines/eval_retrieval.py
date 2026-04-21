"""
Evaluate retrieval (kNN) quality and latency on a K3D GLB.

Metrics
- recall@k against IndexFlatL2 (CPU) as ground truth
- average query time and build time for candidate index (flat/ivf)

Usage
  python -m knowledge3d.tools.eval_retrieval --gltf viewer/public/k3d_foundation.6k.umap.glb \
    --k 10 --queries 512 --ann ivf --out docs/reports/status/retrieval-6k.json

  python -m knowledge3d.tools.eval_retrieval --gltf ../Knowledge3D.local/datasets/ai_compendium.80k.umap.ivf.glb \
    --k 10 --queries 512 --ann ivf --out docs/reports/status/retrieval-80k.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pygltflib import GLTF2  # type: ignore


def _load_embeddings_from_glb(p: Path) -> np.ndarray:
    g = GLTF2().load(str(p))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ev = int(k3d.get("embeddingsView", 1))
    dims = int(k3d.get("embeddingDims", 384))
    prec = str(k3d.get("embeddingPrecision", "f32")).lower()
    bv = g.bufferViews[ev]
    blob = g.binary_blob()
    start = (bv.byteOffset or 0)
    end = start + bv.byteLength
    raw = blob[start:end]
    if prec == "f16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
    n = arr.size // dims
    emb = arr.reshape((n, dims)).copy()
    return emb


def _recall_at_k(truth: np.ndarray, cand: np.ndarray, k: int) -> float:
    # truth, cand: (Q, k) neighbor indices
    Q = truth.shape[0]
    acc = 0.0
    for i in range(Q):
        acc += len(set(truth[i]).intersection(set(cand[i]))) / float(k)
    return acc / float(Q)


def eval_retrieval(glb_path: Path, k: int, queries: int, ann: str = "ivf") -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors  # ground truth
    from knowledge3d.accel import knn_all  # type: ignore
    import os

    emb = _load_embeddings_from_glb(glb_path)
    n = emb.shape[0]
    qn = min(queries, n)
    rng = np.random.default_rng(42)
    q_idx = rng.choice(n, size=qn, replace=False)

    # Build ground truth (IndexFlat via sklearn on full set) and measure time
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(emb)
    _, truth_all = nn.kneighbors(emb[q_idx])
    truth = truth_all[:, 1:]
    t_truth = time.perf_counter() - t0

    # Candidate ANN (FAISS GPU; optional CPU fallback disabled when K3D_STRICT_GPU=1)
    t1 = time.perf_counter()
    try:
        idx = knn_all(emb, k, ann=ann)
    except Exception as e:
        if os.getenv("K3D_STRICT_GPU", "1").strip() not in {"", "0", "false", "False"}:
            raise
        # Optional fallback (not expected in strict mode)
        nn2 = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(emb)
        _, idx_full = nn2.kneighbors(emb)
        idx = idx_full[:, 1:]
    t_build = time.perf_counter() - t1
    # Extract candidate neighbors for q_idx rows
    cand = idx[q_idx]

    rec = _recall_at_k(truth, cand, k)
    return {
        "n": int(n),
        "dims": int(emb.shape[1]),
        "queries": int(qn),
        "k": int(k),
        "ann": ann,
        "recall@k": float(rec),
        "ground_truth_time_s": float(t_truth),
        "candidate_time_s": float(t_build),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate kNN retrieval on a K3D GLB")
    p.add_argument("--gltf", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=512)
    p.add_argument("--ann", choices=["flat", "ivf"], default="ivf")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = eval_retrieval(Path(args.gltf), args.k, args.queries, ann=args.ann)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
