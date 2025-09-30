from __future__ import annotations

"""Ingest open 3D shape datasets and convert to K3D GLBs (minimal pipeline).

Supports: ModelNet10 (OFF format). Downloads a small subset, converts meshes
to GLB with extras.k3d metadata, and updates the House manifest for Tablet use.

Usage:
  PYTHONPATH=. python -m knowledge3d.tools.phase25.ingest_open_shapes \
    --dataset modelnet10 --limit 50
"""

import argparse
import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT.parent / "Knowledge3D.local" / "datasets" / "shapes"
HOUSE_PUBLIC = ROOT / "viewer" / "public" / "house"
MATERIAL_DIR = HOUSE_PUBLIC / "materialized_objects"
MANIFEST = MATERIAL_DIR / "manifest.json"


def _ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MATERIAL_DIR.mkdir(parents=True, exist_ok=True)


def _download_modelnet10(dst_zip: Path) -> None:
    import urllib.request
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    if dst_zip.exists() and dst_zip.stat().st_size > 1024:
        return
    print("Downloading ModelNet10 (small subset)...")
    urllib.request.urlretrieve(url, dst_zip.as_posix())


def _iter_off_files(root: Path, limit: Optional[int]) -> Iterable[Path]:
    num = 0
    for cls in sorted((root / "ModelNet10").glob("*")):
        if not cls.is_dir():
            continue
        for split in ["train", "test"]:
            for f in (cls / split).glob("*.off"):
                yield f
                num += 1
                if limit and num >= int(limit):
                    return


def _load_off(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Minimal OFF parser: vertices and faces
    text = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not text or not text[0].strip().lower().endswith("off"):
        raise ValueError("not an OFF file")
    idx = 1
    while idx < len(text) and (not text[idx].strip() or text[idx].strip().startswith("#")):
        idx += 1
    counts = list(map(int, text[idx].strip().split()))
    idx += 1
    vcount, fcount = counts[0], counts[1]
    verts = []
    for _ in range(vcount):
        parts = text[idx].strip().split()
        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        idx += 1
    faces = []
    for _ in range(fcount):
        parts = text[idx].strip().split()
        n = int(parts[0])
        idx += 1
        if n == 3:
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
        elif n == 4:
            # triangulate quad
            a, b, c, d = map(int, parts[1:5])
            faces.append([a, b, c]); faces.append([a, c, d])
        else:
            # skip polygons >4
            continue
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _write_glb(out_path: Path, vertices: np.ndarray, faces: np.ndarray, extras: dict) -> None:
    from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Asset
    vbytes = vertices.astype(np.float32).tobytes()
    idx = faces.reshape(-1).astype(np.uint32)
    ibytes = idx.tobytes()
    blob = vbytes + ibytes
    g = GLTF2(asset=Asset(generator="k3d-ingest-shapes"), scenes=[Scene(nodes=[0])], scene=0)
    g.buffers.append(Buffer(byteLength=len(blob)))
    g.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=len(vbytes), target=34962))
    g.bufferViews.append(BufferView(buffer=0, byteOffset=len(vbytes), byteLength=len(ibytes), target=34963))
    g.accessors.append(
        Accessor(bufferView=0, byteOffset=0, componentType=5126, count=vertices.shape[0], type="VEC3", min=[], max=[])
    )
    g.accessors.append(
        Accessor(bufferView=1, byteOffset=0, componentType=5125, count=idx.size, type="SCALAR", min=[], max=[])
    )
    g.meshes.append(Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1, mode=4)]))
    g.nodes.append(Node(mesh=0, extras={"k3d": extras}))
    g.set_binary_blob(blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(out_path))


def _update_manifest(entries: List[dict]) -> None:
    import json
    if MANIFEST.exists():
        try:
            obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            obj = {"shapes": [], "rays": []}
    else:
        obj = {"shapes": [], "rays": []}
    shapes = obj.get("shapes") if isinstance(obj, dict) else []
    if not isinstance(shapes, list):
        shapes = []
    # append new entries (avoid duplicates by path)
    existing = {s.get("path") for s in shapes if isinstance(s, dict)}
    for e in entries:
        if e.get("path") not in existing:
            shapes.append(e)
    obj["shapes"] = shapes
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_modelnet10(limit: int) -> None:
    _ensure_dirs()
    zpath = DATA_ROOT / "ModelNet10.zip"
    _download_modelnet10(zpath)
    with zipfile.ZipFile(zpath, 'r') as zf:
        if not (DATA_ROOT / "ModelNet10").exists():
            zf.extractall(DATA_ROOT)
    # Convert a subset to GLB
    out_entries: List[dict] = []
    count = 0
    for off in _iter_off_files(DATA_ROOT, limit):
        try:
            V, F = _load_off(off)
        except Exception:
            continue
        rel_name = f"modelnet10_{off.parent.parent.name}_{off.stem}.glb"
        out = MATERIAL_DIR / rel_name
        extras = {
            "type": "external_shape",
            "name": f"{off.parent.parent.name}:{off.stem}",
            "source": "ModelNet10",
            "prompt": f"shape {off.parent.parent.name}",
            "vertex_count": int(V.shape[0]),
            "face_count": int(F.shape[0]),
        }
        try:
            _write_glb(out, V, F, extras)
            rel = "/" + str(out.relative_to(ROOT / "viewer" / "public")).replace(os.sep, "/")
            out_entries.append({
                "path": rel,
                "name": extras["name"],
                "shape_type": off.parent.parent.name,
                "prompt": extras["prompt"],
            })
            count += 1
        except Exception:
            continue
        if count >= limit:
            break
    if out_entries:
        _update_manifest(out_entries)
        print(f"Ingested {len(out_entries)} shapes into manifest.")
    else:
        print("No shapes ingested.")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Ingest open-source shapes to House GLB")
    ap.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10"])
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    if args.dataset == "modelnet10":
        ingest_modelnet10(int(args.limit))


if __name__ == "__main__":
    main()

