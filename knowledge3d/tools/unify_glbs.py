from __future__ import annotations

"""
Unify multiple per‑modality K3D GLBs into a single Galaxy GLB.

Loads embeddings + metadata from each input GLB, projects embeddings to a
common dimensionality, computes 3D positions and neighbors, and writes a
single output GLB with embedded K3D payload.

Usage
  python -m knowledge3d.tools.unify_glbs \
    --out viewer/public/galaxy.glb --dims 256 --k 10 --reducer pca \
    viewer/public/coco_50k.glb:image \
    viewer/public/clotho.glb:audio \
    viewer/public/vatex_2k.glb:video

Notes
- Modality overrides after a colon set `metadata.type` when missing.
- Preserves labels and other metadata fields when present.
- Handles f16/f32 embeddings.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pygltflib import GLTF2  # type: ignore

from k3dgen.__main__ import create_gltf_file, reduce_dimensions, find_neighbors  # type: ignore


@dataclass
class InputSpec:
    path: Path
    override_type: str | None


def parse_argspec(items: List[str]) -> List[InputSpec]:
    out: List[InputSpec] = []
    for tok in items:
        if ":" in tok:
            p, t = tok.split(":", 1)
            out.append(InputSpec(Path(p), t.strip().lower() or None))
        else:
            out.append(InputSpec(Path(tok), None))
    return out


def _half_to_float(arr_u16: np.ndarray) -> np.ndarray:
    # numpy supports float16; just view and astype
    return arr_u16.view(np.float16).astype(np.float32, copy=False)


def load_glb_embeddings_and_meta(path: Path) -> Tuple[List[str], np.ndarray, List[dict]]:
    g = GLTF2().load(str(path))
    prim = g.meshes[0].primitives[0]
    k3d = prim.extras.get("k3d", {})
    ids: List[str] = list(k3d.get("ids", []) or [])
    meta: List[dict] = [m if isinstance(m, dict) else {} for m in (k3d.get("metadata", []) or [])]
    emb_view = k3d.get("embeddingsView")
    dims = int(k3d.get("embeddingDims") or 0)
    prec = str(k3d.get("embeddingPrecision") or k3d.get("embedding_prec") or "f32").lower()
    if emb_view is None or dims <= 0:
        # Fall back to JSON embeddings if present (small sets only)
        emb_json = k3d.get("embeddings") or []
        X = np.asarray(emb_json, dtype=np.float32)
        return ids, X, meta
    bv = g.bufferViews[emb_view]
    blob = g.binary_blob()
    start = int(bv.byteOffset or 0)
    end = start + int(bv.byteLength)
    buf = blob[start:end]
    if prec == "f16":
        u16 = np.frombuffer(buf, dtype=np.uint16)
        arr = _half_to_float(u16)
    else:
        arr = np.frombuffer(buf, dtype=np.float32)
    X = np.asarray(arr, dtype=np.float32).reshape((-1, dims))
    return ids, X, meta


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Unify multiple per‑modality GLBs into one Galaxy GLB")
    ap.add_argument("inputs", nargs="+", help="Paths to GLBs, optionally with :type override (text|image|audio|video)")
    ap.add_argument("--out", required=True, help="Output GLB path (e.g., viewer/public/galaxy.glb)")
    ap.add_argument("--dims", type=int, default=256, help="Common embedding dimension before 3D reduction")
    ap.add_argument("--k", type=int, default=10, help="neighbors per node")
    ap.add_argument("--reducer", default="pca", choices=["pca", "umap", "tsne"], help="3D reducer for positions")
    args = ap.parse_args()

    specs = parse_argspec(args.inputs)
    ids_all: List[str] = []
    Xs: List[np.ndarray] = []
    metas_all: List[dict] = []
    kinds: List[str] = []
    for sp in specs:
        if not sp.path.exists():
            print(f"[warn] missing {sp.path}")
            continue
        ids, X, meta = load_glb_embeddings_and_meta(sp.path)
        if not len(ids) or X.size == 0:
            print(f"[warn] no data in {sp.path}")
            continue
        if sp.override_type:
            for m in meta:
                m.setdefault("type", sp.override_type)
        kinds.extend([m.get("type") or (sp.override_type or "other") for m in meta])
        ids_all.extend(ids)
        Xs.append(X)
        metas_all.extend(meta)

    if not ids_all:
        raise SystemExit("No inputs found; nothing to unify")

    # Project to common dimension via PCA over stacked embeddings
    d = int(args.dims)
    # Pad each to the max feature dimension so PCA can fit on a single stacked matrix
    max_d = max(X.shape[1] for X in Xs)
    def _pad(M: np.ndarray) -> np.ndarray:
        if M.shape[1] == max_d:
            return M
        Z = np.zeros((M.shape[0], max_d - M.shape[1]), dtype=np.float32)
        return np.hstack([M, Z])
    Xstack = np.vstack([_pad(X) for X in Xs])
    from sklearn.decomposition import PCA  # CPU safe
    d_eff = min(d, Xstack.shape[1], max(1, Xstack.shape[0] - 1))
    pca = PCA(n_components=d_eff)
    pca.fit(Xstack)
    Xproj: List[np.ndarray] = [pca.transform(_pad(X)) for X in Xs]
    emb = np.vstack(Xproj).astype(np.float32)

    # 3D positions and neighbors
    points = reduce_dimensions(emb, reducer=str(args.reducer))
    k = int(args.k)
    try:
        from cuml.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
        nn.fit(emb)
        dmat, ind = nn.kneighbors(emb)
        try:
            import cupy as cp  # type: ignore
            if isinstance(ind, cp.ndarray):
                ind = ind.get()
        except Exception:
            pass
        nbr = ind[:, 1:]
    except Exception:
        nbr = find_neighbors(emb, k)

    # Labels and metadata
    labels = []
    text = []
    metas_out = []
    for i, m in enumerate(metas_all):
        lab = str(m.get("label") or ids_all[i])
        labels.append(lab)
        text.append(str(m.get("text") or ""))
        mm = dict(m)
        if not mm.get("type"):
            mm["type"] = kinds[i] if i < len(kinds) else "other"
        metas_out.append(mm)

    create_gltf_file(
        gltf_path=str(Path(args.out)),
        ids=ids_all,
        points=points,
        embeddings=emb,
        neighbor_indices=nbr,
        labels=labels,
        metadata_texts=text,
        metadata_override=metas_out,
        fmt="glb",
        emb_precision="f16",
    )
    print(f"Unified GLB -> {args.out} (n={len(ids_all)}; dims={emb.shape[1]})")


if __name__ == "__main__":
    main()
