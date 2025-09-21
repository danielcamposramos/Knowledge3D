from __future__ import annotations

"""Rebuild the default House asset as a PTX-friendly binary GLB."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np  # type: ignore
from pygltflib import (  # type: ignore
    ARRAY_BUFFER,
    Accessor,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Mesh,
    Node,
    Primitive,
    Scene,
)


def _load_house_gltf(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"House glTF not found: {path}")
    if path.suffix.lower() != ".gltf":
        raise ValueError("Source must be a .gltf file")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def _extract_vectors_embeddings(extras: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(extras.get("vectors"), dtype=np.float32)
    embeddings = np.asarray(extras.get("embeddings"), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("House vectors must be Nx3 array")
    if embeddings.ndim != 2:
        raise ValueError("House embeddings must be 2D array")
    if len(vectors) != len(embeddings):
        raise ValueError("Vectors and embeddings length mismatch")
    return vectors, embeddings


def _build_glb(extras: Dict[str, object]) -> GLTF2:
    ids: List[str] = list(extras.get("ids", []))
    if not ids:
        raise ValueError("House extras missing ids")
    vectors, embeddings = _extract_vectors_embeddings(extras)
    metadata = extras.get("metadata", [])
    neighbors = extras.get("neighbors", [])

    pos_bytes = vectors.astype(np.float32).tobytes()
    emb_bytes = embeddings.astype(np.float32).tobytes()
    blob = pos_bytes + emb_bytes

    buffer = Buffer(byteLength=len(blob))
    view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=ARRAY_BUFFER)
    view_emb = BufferView(buffer=0, byteOffset=len(pos_bytes), byteLength=len(emb_bytes))

    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,
        count=vectors.shape[0],
        type="VEC3",
        max=vectors.max(axis=0).tolist(),
        min=vectors.min(axis=0).tolist(),
    )

    k3d_extras: Dict[str, object] = {
        "ids": ids,
        "metadata": metadata,
        "neighbors": neighbors,
        "vectorsView": 0,
        "embeddingsView": 1,
        "embeddingDims": int(embeddings.shape[1]),
    }
    for key in ("embeddingPrecision", "ai_interaction_protocol", "ai_state_flags"):
        if key in extras:
            k3d_extras[key] = extras[key]

    primitive = Primitive(attributes={"POSITION": 0}, mode=0, extras={"k3d": k3d_extras})
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0, name="k3d-house-memory")
    scene = Scene(nodes=[0])

    gltf = GLTF2(
        asset=Asset(generator="knowledge3d.rebuild_house_glb"),
        buffers=[buffer],
        bufferViews=[view_pos, view_emb],
        accessors=[accessor],
        meshes=[mesh],
        nodes=[node],
        scenes=[scene],
        scene=0,
    )
    gltf.set_binary_blob(blob)
    return gltf


def rebuild_house(src: Path, dst: Path) -> None:
    data = _load_house_gltf(src)
    try:
        extras = data["meshes"][0]["primitives"][0]["extras"]["k3d"]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unexpected house glTF structure; unable to locate k3d extras") from exc

    glb = _build_glb(extras)
    dst.parent.mkdir(parents=True, exist_ok=True)
    glb.save(dst.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild legacy house glTF as PTX-ready GLB")
    parser.add_argument("src", type=Path, help="Path to legacy memory_house.glb")
    parser.add_argument("dst", type=Path, help="Output memory_house.glb path")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    rebuild_house(args.src.resolve(), args.dst.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()

