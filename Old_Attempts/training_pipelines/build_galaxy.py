from __future__ import annotations

"""
Build a single unified Galaxy GLB from multiple modality CSV+metadata pairs.

All knowledge lives in one galaxy (one space). This tool projects all
embeddings to a common dimensionality (PCA) and merges metadata with a
"type" field (image|audio|video|text|other). It then computes neighbors and
embeds everything in a single glTF/GLB using k3dgen's writer.

Usage
  python -m knowledge3d.tools.build_galaxy \
    --out viewer/public/galaxy.glb --dims 256 --k 10 --reducer pca \
    image:../Knowledge3D.local/datasets/coco.train.clip.csv:../Knowledge3D.local/datasets/coco.train.meta.json \
    audio:../Knowledge3D.local/datasets/clotho.clap.csv:../Knowledge3D.local/datasets/clotho.meta.json \
    video:../Knowledge3D.local/datasets/vatex.clip.csv:../Knowledge3D.local/datasets/vatex.meta.json

Notes
- Works even if some inputs are missing; skips gracefully and builds with the rest.
- Defaults favor CPU-safe operations (PCA for 3D positions and kNN fallback via sklearn).
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from k3dgen.__main__ import create_gltf_file, reduce_dimensions, find_neighbors  # type: ignore
from knowledge3d.accel import st_device_kwargs  # type: ignore


@dataclass
class ModalitySpec:
    kind: str  # image|audio|video|text|other
    csv_path: Path
    meta_path: Path | None


def parse_specs(args: List[str]) -> List[ModalitySpec]:
    specs: List[ModalitySpec] = []
    for token in args:
        # kind:csv:meta or kind:csv
        parts = token.split(":")
        if len(parts) < 2:
            continue
        kind = parts[0].strip()
        path1 = Path(parts[1].strip())
        meta_path = Path(parts[2].strip()) if len(parts) >= 3 else None
        specs.append(ModalitySpec(kind=kind, csv_path=path1, meta_path=meta_path))
    return specs


def load_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    vecs: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header:
            return ids, np.zeros((0, 1), dtype=float)
        cols = [i for i, h in enumerate(header) if i == 0 or h.startswith("v")]
        id_idx = cols[0]
        vec_idx = cols[1:]
        for row in r:
            try:
                ids.append(str(row[id_idx]))
                vecs.append([float(row[i]) for i in vec_idx])
            except Exception:
                continue
    if not ids:
        return ids, np.zeros((0, 1), dtype=float)
    return ids, np.asarray(vecs, dtype=float)


def load_textlines_embed(path: Path, strict_gpu: bool = True) -> Tuple[List[str], np.ndarray, List[str]]:
    lines = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    # Sentence-Transformers on GPU only
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise SystemExit("sentence-transformers not installed") from e
    dev = st_device_kwargs()
    if strict_gpu and (not dev or dev.get('device') != 'cuda'):
        raise SystemExit("GPU required for text embedding (K3D_STRICT_GPU)")
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', **dev)
    except TypeError:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = np.asarray(model.encode(lines, convert_to_numpy=True, **dev), dtype=float)
    ids = [f"text:{i}" for i in range(len(lines))]
    labels = [s if len(s) <= 48 else (s[:45] + '...') for s in lines]
    return ids, emb, labels


def load_meta(path: Path | None, n: int, defaults: dict) -> List[dict]:
    if not path or not path.exists():
        return [dict(defaults) for _ in range(n)]
    try:
        txt = path.read_text(encoding="utf-8")
        if txt.lstrip().startswith("["):
            arr = json.loads(txt)
            if isinstance(arr, list) and len(arr) == n:
                return [({**defaults, **(it if isinstance(it, dict) else {})}) for it in arr]
        # otherwise, try JSONL
        out: List[dict] = []
        for line in txt.splitlines():
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                j = {}
            out.append({**defaults, **(j if isinstance(j, dict) else {})})
        if len(out) == n:
            return out
    except Exception:
        pass
    return [dict(defaults) for _ in range(n)]


def project_common(Xs: List[np.ndarray], out_dims: int) -> List[np.ndarray]:
    """Project all matrices to a common dimensionality using PCA fitted on the stack."""
    if not Xs:
        return []
    keep = [X for X in Xs if X.size > 0]
    if not keep:
        return [np.zeros((0, out_dims), dtype=float) for _ in Xs]
    # Pad each to max feature dimension so we can stack for PCA
    max_d = max(M.shape[1] for M in keep)
    def _pad(M: np.ndarray) -> np.ndarray:
        if M.shape[1] == max_d:
            return M
        Z = np.zeros((M.shape[0], max_d - M.shape[1]), dtype=float)
        return np.hstack([M, Z])
    padded = [_pad(M) for M in keep]
    X = np.vstack(padded)
    d = min(out_dims, X.shape[1], max(1, X.shape[0] - 1))
    from sklearn.decomposition import PCA  # CPU-safe
    pca = PCA(n_components=d)
    pca.fit(X)
    out: List[np.ndarray] = []
    for M in Xs:
        if M.size == 0:
            out.append(np.zeros((0, d), dtype=float))
        else:
            out.append(pca.transform(_pad(M)))
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build unified Galaxy GLB from multiple modality CSV+meta pairs")
    ap.add_argument("specs", nargs="+", help="kind:csv[:meta] tokens, e.g., image:/a/b.csv:/a/b.json")
    ap.add_argument("--out", required=True, help="Output GLB path (e.g., viewer/public/galaxy.glb)")
    ap.add_argument("--dims", type=int, default=256, help="Common embedding dim (PCA)")
    ap.add_argument("--k", type=int, default=10, help="neighbors per node")
    ap.add_argument("--reducer", default="pca", choices=["pca", "umap", "tsne"], help="3D reducer for positions")
    args = ap.parse_args()

    specs = parse_specs(args.specs)
    ids_all: List[str] = []
    kind_all: List[str] = []
    Xs: List[np.ndarray] = []
    metas_all: List[dict] = []

    import os
    strict = os.getenv("K3D_STRICT_GPU", "1").strip() not in {"", "0", "false", "False"}
    for sp in specs:
        if not sp.csv_path.exists():
            continue
        if sp.kind == 'textlines':
            ids, X, labels = load_textlines_embed(sp.csv_path, strict_gpu=strict)
            meta_defaults = {"type": "text"}
            metas = load_meta(sp.meta_path, len(ids), meta_defaults)
            # Ensure labels populate metadata
            for i, m in enumerate(metas):
                m.setdefault('label', labels[i])
            ids_all.extend(ids)
            kind_all.extend(['text'] * len(ids))
            Xs.append(X)
            metas_all.extend(metas)
        else:
            ids, X = load_csv(sp.csv_path)
            if not ids:
                continue
            ids_pref = [f"{sp.kind}:{i}" for i in ids]
            meta_defaults = {"type": sp.kind}
            metas = load_meta(sp.meta_path, len(ids_pref), meta_defaults)
            ids_all.extend(ids_pref)
            kind_all.extend([sp.kind] * len(ids_pref))
            Xs.append(X)
            metas_all.extend(metas)

    if not ids_all:
        raise SystemExit("No inputs found; nothing to build")

    # Project to common dimension
    proj = project_common(Xs, int(args.dims))
    emb = np.vstack(proj).astype(np.float32)

    # 3D positions and neighbors (strict GPU only)
    points = reduce_dimensions(emb, reducer=str(args.reducer))
    k = int(args.k)
    try:
        # Prefer cuML NearestNeighbors (GPU)
        from cuml.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
        nn.fit(emb)
        d, ind = nn.kneighbors(emb)
        try:
            import cupy as cp  # type: ignore
            if isinstance(ind, cp.ndarray):
                ind = ind.get()
        except Exception:
            pass
        nbr = ind[:, 1:]
    except Exception:
        # Fallback to FAISS GPU path via accel (still strict GPU; will raise if not available)
        nbr = find_neighbors(emb, k)

    # Labels (prefer metadata.label if exists)
    labels = []
    text = []
    metas_out = []
    for i, m in enumerate(metas_all):
        lab = str(m.get("label") or ids_all[i])
        labels.append(lab)
        text.append(str(m.get("text") or ""))
        # Preserve important fields; append type
        mm = dict(m)
        mm["type"] = kind_all[i]
        metas_out.append(mm)

    # Write GLB
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
    print(f"Built unified galaxy -> {args.out} ({len(ids_all)} nodes; dims={emb.shape[1]})")


if __name__ == "__main__":
    main()
