from __future__ import annotations

"""
Ingest Open 3D asset folders into a K3D GLB index with embeddings.

Scans a directory for 3D assets (.glb, .gltf, .obj) and builds a K3D GLB where
each node represents an asset with metadata (label, path, type). Embeddings are
computed from textual captions (if present) or filename; positions derived via
reducer (UMAP default) and neighbors computed on GPU (FAISS when available via
k3dgen internals).

This is an index; meshes remain on disk. Viewers/tools can later resolve paths
to stream/load the actual assets.

Usage:
  scripts/k3d_env.sh run python -m knowledge3d.tools.ingest_open3d \
    --root /k3dlocal/datasets/shapes \
    --out viewer/public/shapes_index.glb \
    --pattern ".glb,.gltf,.obj" --reducer umap
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple


def discover_assets(root: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        for path in root.rglob(f"*{p}"):
            if path.is_file():
                out.append(path)
    # stable order
    out.sort(key=lambda p: str(p))
    return out


def load_captions(root: Path) -> dict:
    # Optional captions.json: [{"path": "rel/path/to/file", "caption": "..."}, ...]
    caps = {}
    for name in ("captions.json", "metadata.json"):
        f = root / name
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for it in data:
                        try:
                            rel = str(it.get("path") or "").strip()
                            cap = str(it.get("caption") or it.get("text") or "").strip()
                            if rel and cap:
                                caps[rel] = cap
                        except Exception:
                            continue
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str):
                            caps[str(k)] = v
            except Exception:
                pass
    return caps


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest 3D asset folder into a K3D GLB index")
    ap.add_argument("--root", required=True, help="Folder containing assets")
    ap.add_argument("--out", required=True, help="Output GLB index path")
    ap.add_argument("--pattern", default=".glb,.gltf,.obj", help="Comma-separated extensions to include")
    ap.add_argument("--reducer", default="umap", choices=["umap", "pca", "tsne"])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Text embedding model")
    args = ap.parse_args()
    root = Path(args.root)
    pats = [s.strip() for s in str(args.pattern).split(",") if s.strip()]
    assets = discover_assets(root, pats)
    if not assets:
        raise SystemExit(f"No assets found under {root} for patterns {pats}")
    # Build labels + texts
    caps = load_captions(root)
    ids: List[str] = []
    texts: List[str] = []
    meta: List[dict] = []
    for p in assets:
        rel = str(p.relative_to(root))
        lab = Path(rel).stem.replace("_", " ")
        txt = caps.get(rel, lab)
        ids.append(rel)
        texts.append(txt)
        meta.append({"label": lab, "path": rel, "type": "asset3d"})
    # Embeddings on GPU if available
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch  # type: ignore
        st = SentenceTransformer(args.model, device=("cuda" if torch.cuda.is_available() else "cpu"))
        embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        raise SystemExit(f"Failed to compute embeddings: {e}")
    # Use k3dgen internals to create GLB
    from k3dgen.__main__ import reduce_dimensions, find_neighbors, create_gltf_file
    reducer = str(args.reducer)
    pts = reduce_dimensions(embs, reducer=reducer)
    try:
        import numpy as _np  # type: ignore
        pts = _np.nan_to_num(pts, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        embs = _np.nan_to_num(embs, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception:
        pass
    # Compute neighbors with FAISS-GPU (strict GPU mode; will fail if GPU FAISS unavailable)
    nbr_idx = find_neighbors(embs, k=10)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [m.get("label", i) for m, i in zip(meta, ids)]
    fmt = "glb" if str(out_path).lower().endswith(".glb") else "gltf"
    create_gltf_file(
        str(out_path),
        ids,
        pts,
        embs,
        nbr_idx,
        labels,
        metadata_texts=None,
        metadata_override=meta,
        fmt=fmt,
        emb_precision="f16",
        ai_protocol="spatial_reasoning",
        ai_flags={"has_new_information": True},
        ai_flags_mask=None,
        lod_positions=[],
    )
    print(str(out_path))


if __name__ == "__main__":
    main()
