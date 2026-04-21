"""
HunyuanWorld Adapter (stub)

Purpose
- Bridge HunyuanWorld scene generation with K3D's permanent memory format
  (glTF/GLB with embedded extras.k3d), without tightly coupling repos.

Capabilities (now)
- Option A: Convert an embeddings CSV (+ optional metadata JSON) directly
  into a K3D GLB using the same code paths as k3dgen (GPU FAISS for k‑NN,
  PCA/UMAP via knowledge3d.accel).
- Option B (stub): Print instructions to run HunyuanWorld demo scripts and
  where to place outputs for later conversion.

Planned (future)
- Invoke ext/HunyuanWorld-1.0 demo_scenegen.py/panogen with prompts and
  capture generated scenes. Convert meshes/scene graphs into one or more
  glTF scenes; attach extras.k3d (ids, neighbors, embeddings, metadata)
  and optional thumbnails. Use CLIP to embed textures/object labels.

Usage
  # Convert embeddings CSV + metadata JSON into GLB
  python3 -m knowledge3d.tools.hunyuan_adapter to-k3d \
    --csv /k3dlocal/datasets/wit.sample.clip.csv \
    --out /k3dlocal/datasets/wit.sample.glb \
    --metadata /k3dlocal/datasets/wit.sample.meta.json \
    --k 10 --reducer pca --emb-precision f16

  # Show guidance to run HunyuanWorld demos (stub)
  python3 -m knowledge3d.tools.hunyuan_adapter gen \
    --mode scenegen --prompt "bookshelves in a study"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _convert_csv_to_k3d(
    csv_path: Path,
    out_glb: Path,
    k: int = 10,
    reducer: str = "pca",
    meta_path: Path | None = None,
    emb_precision: str = "f32",
) -> None:
    # Reuse k3dgen internals for a faithful embedded GLB
    from k3dgen.__main__ import load_vectors, reduce_dimensions, find_neighbors, create_gltf_file

    ids, embeddings = load_vectors(str(csv_path))
    if k <= 0 or k >= len(ids):
        raise ValueError("k must be >0 and < number of vectors")
    points = reduce_dimensions(embeddings, reducer=reducer)
    neighbor_indices = find_neighbors(embeddings, k)

    labels: List[str] | None = ids
    metadata_override: List[Dict[str, Any]] | None = None
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, list) and len(meta) == len(ids):
                metadata_override = meta
                # best-effort labels from metadata
                try:
                    labels = [ (m.get("label") or i) for m,i in zip(metadata_override, ids) ]  # type: ignore
                except Exception:
                    labels = ids
        except Exception:
            metadata_override = None

    # trivial edges for now; LOD positions omitted
    create_gltf_file(
        str(out_glb),
        ids,
        points,
        embeddings,
        neighbor_indices,
        labels,
        metadata_override=metadata_override,
        fmt="glb" if str(out_glb).lower().endswith(".glb") else "gltf",
        emb_precision=emb_precision,
    )


def _print_hunyuan_instructions(mode: str, prompt: str | None) -> None:
    root = Path(__file__).resolve().parents[2]
    ext = root / "ext" / "HunyuanWorld-1.0"
    print("HunyuanWorld adapter (stub)\n")
    if not ext.exists():
        print("- Missing ext/HunyuanWorld-1.0. Clone the repo:")
        print("    git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 ext/HunyuanWorld-1.0")
        return
    print(f"- Repo present: {ext}")
    print("- To generate scenes (examples):")
    if mode == "scenegen":
        print("    cd ext/HunyuanWorld-1.0 && python demo_scenegen.py --prompt \"<your prompt>\" --out /k3dlocal/datasets/hy_scenes/")
    else:
        print("    cd ext/HunyuanWorld-1.0 && python demo_panogen.py --prompt \"<your prompt>\" --out /k3dlocal/datasets/hy_pano/")
    if prompt:
        print(f"  prompt: {prompt}")
    print("\n- Then convert generated outputs to K3D via this tool's to-k3d mode (using CSV+metadata), or use a future mesh→glTF converter.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="HunyuanWorld → K3D adapter (stub)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("to-k3d", help="Convert embeddings CSV (+metadata) into a K3D GLB")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--metadata")
    p1.add_argument("--k", type=int, default=10)
    p1.add_argument("--reducer", choices=["pca", "umap", "tsne"], default="pca")
    p1.add_argument("--emb-precision", choices=["f32", "f16"], default="f32")

    p2 = sub.add_parser("gen", help="Show how to run HunyuanWorld demos (stub)")
    p2.add_argument("--mode", choices=["scenegen", "panogen"], default="scenegen")
    p2.add_argument("--prompt")

    args = ap.parse_args()
    if args.cmd == "to-k3d":
        _convert_csv_to_k3d(Path(args.csv), Path(args.out), k=int(args.k), reducer=str(args.reducer), meta_path=Path(args.metadata) if args.metadata else None, emb_precision=str(args.emb_precision))
    else:
        _print_hunyuan_instructions(str(args.mode), str(args.prompt) if args.prompt else None)


if __name__ == "__main__":
    main()

