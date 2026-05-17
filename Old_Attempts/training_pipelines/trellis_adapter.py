"""
TRELLIS Adapter (stub + mesh→K3D helper)

Purpose
- Bridge Microsoft TRELLIS asset generation with K3D's permanent memory
  format (glTF/GLB + extras.k3d), without hard-coupling repos.

Features
- to-k3d: Convert embeddings CSV (+metadata JSON) into a K3D GLB using
  k3dgen internals (GPU FAISS neighbors; PCA/UMAP reducers via accel).
- from-mesh: Inject extras.k3d into an existing glTF/GLB that already has
  POSITION data, mapping vertices to K3D nodes (ids=v0..v{n-1}) so it can
  be navigated in the viewer immediately.
- gen (stub): Print instructions to run TRELLIS generation and where to
  place outputs for later conversion.

Usage
  # CSV (+metadata) → GLB
  python3 -m knowledge3d.tools.trellis_adapter to-k3d \
    --csv /k3dlocal/datasets/trellis.sample.clip.csv \
    --out /k3dlocal/datasets/trellis.sample.glb \
    --metadata /k3dlocal/datasets/trellis.sample.meta.json \
    --k 10 --reducer pca --emb-precision f16

  # Inject K3D into a mesh glTF so the viewer can load it
  python3 -m knowledge3d.tools.trellis_adapter from-mesh \
    --gltf /k3dlocal/datasets/asset.gltf --out /k3dlocal/datasets/asset.k3d.gltf

  # Print TRELLIS run guidance (stub)
  python3 -m knowledge3d.tools.trellis_adapter gen --prompt "a study room with shelves"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _csv_to_k3d(csv: Path, out: Path, k: int, reducer: str, meta: Path | None, emb_precision: str) -> None:
    from k3dgen.__main__ import load_vectors, reduce_dimensions, find_neighbors, create_gltf_file
    ids, embeddings = load_vectors(str(csv))
    n = len(ids)
    if k <= 0 or k >= n:
        # allow tiny sets by capping k
        k = max(1, min(10, n - 1))
    points = reduce_dimensions(embeddings, reducer=reducer)
    try:
        nbrs = find_neighbors(embeddings, k)
    except Exception:
        # Fallback: trivial ring neighbors; unify_glbs will recompute global neighbors anyway
        import numpy as _np
        nbrs = _np.zeros((n, k), dtype=int)
        for i in range(n):
            for j in range(k):
                nbrs[i, j] = (i + j + 1) % n
    labels: List[str] | None = ids
    meta_override: List[Dict[str, Any]] | None = None
    if meta and meta.exists():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if isinstance(m, list) and len(m) == len(ids):
                meta_override = m
                try:
                    labels = [(mi.get("label") or ii) for mi, ii in zip(meta_override, ids)]  # type: ignore
                except Exception:
                    labels = ids
        except Exception:
            meta_override = None
    create_gltf_file(
        str(out), ids, points, embeddings, nbrs, labels,
        metadata_override=meta_override,
        fmt="glb" if str(out).lower().endswith(".glb") else "gltf",
        emb_precision=emb_precision,
    )


def _from_mesh_add_k3d(gltf_in: Path, gltf_out: Path) -> None:
    """Inject minimal extras.k3d into a glTF/GLB that has POSITION.

    - Uses the first primitive's POSITION accessor to infer node count and
      bufferView index. Sets vectorsView to reuse that bufferView.
    - Creates ids ["v0"..] and attaches k3dIds and k3d payload with empty
      neighbors and metadata (label defaults to id).
    """
    from pygltflib import GLTF2
    g = GLTF2().load(str(gltf_in))
    if not g.meshes or not g.meshes[0].primitives:
        raise RuntimeError("No meshes/primitives found in glTF")
    prim = g.meshes[0].primitives[0]
    if not prim.attributes or "POSITION" not in prim.attributes:
        raise RuntimeError("Primitive lacks POSITION attribute")
    accessor_index = prim.attributes["POSITION"]
    if accessor_index is None:
        raise RuntimeError("POSITION accessor missing")
    accessor = g.accessors[accessor_index]
    if accessor.bufferView is None:
        raise RuntimeError("POSITION accessor missing bufferView")
    count = int(accessor.count or 0)
    view_index = int(accessor.bufferView)
    ids = [f"v{i}" for i in range(count)]
    meta = [{"label": vid} for vid in ids]
    neighbors: List[List[str]] = [[] for _ in range(count)]
    k3d_payload: Dict[str, Any] = {
        "ids": ids,
        "vectorsView": view_index,
        "embeddingDims": 0,
        "metadata": meta,
        "neighbors": neighbors,
    }
    # Attach to primitive extras
    if prim.extras is None:
        prim.extras = {}
    prim.extras["k3dIds"] = ids
    prim.extras["k3d"] = k3d_payload
    g.save(str(gltf_out))


def _print_trellis_instructions(prompt: str | None) -> None:
    root = Path(__file__).resolve().parents[2]
    ext = root / "ext" / "TRELLIS"
    print("TRELLIS adapter (stub)\n")
    print("- Clone (outside the repo or under ext/):")
    print("    git clone --depth 1 https://github.com/microsoft/TRELLIS ext/TRELLIS")
    print("- Follow their README to set up weights and run generation.")
    print("- Place outputs (meshes) under /k3dlocal/datasets/trellis/")
    print("- Then inject K3D extras via from-mesh:")
    print("    python -m knowledge3d.tools.trellis_adapter from-mesh --gltf /k3dlocal/datasets/trellis/asset.gltf --out /k3dlocal/datasets/trellis/asset.k3d.gltf")
    if prompt:
        print(f"  prompt: {prompt}")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="TRELLIS → K3D adapter (stub)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("to-k3d", help="Convert embeddings CSV (+metadata) into a K3D GLB")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--metadata")
    p1.add_argument("--k", type=int, default=10)
    p1.add_argument("--reducer", choices=["pca", "umap", "tsne"], default="pca")
    p1.add_argument("--emb-precision", choices=["f32", "f16"], default="f32")

    p2 = sub.add_parser("from-mesh", help="Inject extras.k3d into an existing glTF/GLB (uses POSITION as vectorsView)")
    p2.add_argument("--gltf", required=True)
    p2.add_argument("--out", required=True)

    p3 = sub.add_parser("gen", help="Show how to run TRELLIS generation (stub)")
    p3.add_argument("--prompt")

    args = ap.parse_args()
    if args.cmd == "to-k3d":
        _csv_to_k3d(Path(args.csv), Path(args.out), int(args.k), str(args.reducer), Path(args.metadata) if args.metadata else None, str(args.emb_precision))
    elif args.cmd == "from-mesh":
        _from_mesh_add_k3d(Path(args.gltf), Path(args.out))
    else:
        _print_trellis_instructions(str(args.prompt) if args.prompt else None)


if __name__ == "__main__":
    main()
