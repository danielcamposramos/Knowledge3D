from __future__ import annotations

"""
Filter a GLB (embedded K3D) by metadata text/labels containing any of a set of
keywords, and write a new GLB with recomputed positions and neighbors.

Usage
  python -m knowledge3d.tools.filter_glb_by_keywords \
    --input viewer/public/coco_50k.glb \
    --out viewer/public/_world/image.glb \
    --keywords rain,street,car,city,child,speech \
    --max 800 --reducer pca --k 8
"""

import argparse
from typing import List, Tuple
from pathlib import Path

import numpy as np
from pygltflib import GLTF2  # type: ignore

from k3dgen.__main__ import create_gltf_file, reduce_dimensions, find_neighbors  # type: ignore


def _half_to_float(arr_u16: np.ndarray) -> np.ndarray:
    return arr_u16.view(np.float16).astype(np.float32, copy=False)


def _load_k3d(path: Path) -> Tuple[List[str], np.ndarray, List[dict]]:
    g = GLTF2().load(str(path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    meta: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    ev = k3d.get("embeddingsView")
    dims = int(k3d.get("embeddingDims") or 0)
    prec = str(k3d.get("embeddingPrecision") or "f32").lower()
    if ev is None or dims <= 0:
        emb_json = k3d.get("embeddings") or []
        X = np.asarray(emb_json, dtype=np.float32)
        return ids, X, meta
    bv = g.bufferViews[int(ev)]
    blob = g.binary_blob()
    start = int(bv.byteOffset or 0)
    end = start + int(bv.byteLength)
    buf = blob[start:end]
    if prec == "f16":
        arr = _half_to_float(np.frombuffer(buf, dtype=np.uint16))
    else:
        arr = np.frombuffer(buf, dtype=np.float32)
    X = np.asarray(arr, dtype=np.float32).reshape((-1, dims))
    return ids, X, meta


def _meta_text(m: dict) -> str:
    fields = ["caption", "text", "label", "title", "description"]
    parts: List[str] = []
    for k in fields:
        v = m.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    return " \n ".join(parts).lower()


def _select(ids: List[str], X: np.ndarray, meta: List[dict], keywords: List[str], max_items: int) -> Tuple[List[str], np.ndarray, List[dict]]:
    kws = [k.strip().lower() for k in keywords if k.strip()]
    keep: List[int] = []
    for i, m in enumerate(meta):
        if len(keep) >= max_items:
            break
        tx = _meta_text(m)
        if any(k in tx for k in kws):
            keep.append(i)
    if not keep:
        return ids[:0], X[:0], []
    sel_ids = [ids[i] for i in keep]
    sel_X = X[keep, :]
    sel_meta = [meta[i] for i in keep]
    return sel_ids, sel_X, sel_meta


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Filter GLB by metadata keywords and rebuild a compact GLB")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keywords", required=True, help="Comma-separated keywords")
    ap.add_argument("--max", type=int, default=800)
    ap.add_argument("--reducer", default="pca", choices=["pca", "umap", "tsne"])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    ids, X, meta = _load_k3d(Path(args.input))
    if X.size == 0 or not ids:
        raise SystemExit("No data in input GLB")
    sel_ids, sel_X, sel_meta = _select(ids, X, meta, args.keywords.split(","), int(args.max))
    if not sel_ids:
        raise SystemExit("No rows matched the keywords")

    # Reduce to 3D positions; neighbors via FAISS/CPU fallback from k3dgen
    pts = reduce_dimensions(sel_X.astype(np.float32), reducer=str(args.reducer))
    try:
        nbr = find_neighbors(sel_X.astype(np.float32), int(args.k))
    except Exception:
        # Fallback to sklearn brute force (k3dgen internally handles)
        nbr = find_neighbors(sel_X.astype(np.float32), int(args.k))

    labels = [str(m.get("label") or sid) for sid, m in zip(sel_ids, sel_meta)]
    text = [str(m.get("text") or m.get("caption") or "") for m in sel_meta]
    create_gltf_file(
        gltf_path=str(Path(args.out)),
        ids=sel_ids,
        points=pts,
        embeddings=sel_X.astype(np.float32),
        neighbor_indices=nbr,
        labels=labels,
        metadata_texts=text,
        metadata_override=sel_meta,
        fmt="glb",
        emb_precision="f16",
    )
    print(f"Wrote filtered GLB -> {args.out} (n={len(sel_ids)})")


if __name__ == "__main__":
    main()

