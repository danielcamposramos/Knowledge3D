"""Build a PTX-ready house memory index from consolidated artifacts."""
from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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


DEFAULT_EMBED_DIM = 512


@dataclass
class HouseRecord:
    record_id: str
    label: str
    embedding: np.ndarray
    payload: Dict[str, object]
    tags: List[str]


def _hash_embedding(parts: Iterable[str], dim: int) -> np.ndarray:
    joined = "\u241f".join(p.strip() for p in parts if p)
    if not joined:
        joined = "house_memory"
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    values: List[float] = []
    seed = digest
    while len(values) < dim:
        for byte in seed:
            values.append(((byte / 255.0) * 2.0) - 1.0)
            if len(values) >= dim:
                break
        seed = hashlib.sha256(seed).digest()
    return np.asarray(values[:dim], dtype=np.float32)


def _gather_records(root: Path, limit: Optional[int], embed_dim: int) -> List[HouseRecord]:
    records: List[HouseRecord] = []
    count = 0
    for path in sorted(root.glob("*.json")):
        if limit is not None and count >= limit:
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        record_id = str(data.get("id") or path.stem or hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16])
        artifact_type = str(data.get("type") or "artifact")
        label = str(data.get("title") or data.get("name") or artifact_type).strip() or record_id
        zone = str(data.get("zone_placement") or data.get("zone") or "unknown")
        created = str(data.get("created_at") or data.get("timestamp") or "")

        key_candidate = (
            data.get("prompt")
            or data.get("summary")
            or data.get("description")
            or label
        )
        key_norm = str(key_candidate).strip().lower()
        if not key_norm:
            continue

        embedding = _hash_embedding([key_norm], embed_dim)

        payload = {
            "path": str(path),
            "type": artifact_type,
            "zone": zone,
            "created_at": created,
            "updated_at": data.get("updated_at"),
            "title": label,
        }
        if data.get("summary"):
            payload["summary"] = data.get("summary")
        if data.get("description") and "summary" not in payload:
            payload["summary"] = data.get("description")
        if data.get("prompt") and "summary" not in payload:
            payload["summary"] = data.get("prompt")
        if data.get("star_id"):
            payload["star_id"] = data["star_id"]
        if data.get("tags"):
            payload["tags"] = data.get("tags")

        tags: List[str] = []
        tags.append(artifact_type)
        if zone:
            tags.append(zone)
        extra_tags = data.get("tags")
        if isinstance(extra_tags, list):
            for tag in extra_tags:
                if tag and tag not in tags:
                    tags.append(str(tag))

        records.append(
            HouseRecord(
                record_id=record_id,
                label=label,
                embedding=embedding,
                payload=payload,
                tags=tags,
            )
        )
        count += 1
    return records


def _compute_positions(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings to position")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if centered.shape[1] < 3:
        pad = np.zeros((centered.shape[0], 3 - centered.shape[1]), dtype=np.float32)
        return np.concatenate([centered, pad], axis=1)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3, :].T
    coords = centered @ components
    return coords.astype(np.float32)


def _build_glb(records: Sequence[HouseRecord], positions: np.ndarray, embeddings: np.ndarray, label: str) -> GLTF2:
    ids = [record.record_id for record in records]
    modality_counts: Dict[str, int] = {}
    metadata: List[Dict[str, object]] = []
    for record in records:
        artifact_type = record.payload.get("type", "artifact")
        modality_counts.setdefault(str(artifact_type), 0)
        modality_counts[str(artifact_type)] += 1
        metadata.append(
            {
                "name": record.label,
                "tags": list(record.tags),
                "payload": record.payload,
                "modality": str(artifact_type),
            }
        )

    pos_bytes = positions.astype(np.float32).tobytes()
    emb_bytes = embeddings.astype(np.float32).tobytes()
    blob = pos_bytes + emb_bytes

    buffer = Buffer(byteLength=len(blob))
    view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(pos_bytes), target=ARRAY_BUFFER)
    view_emb = BufferView(buffer=0, byteOffset=len(pos_bytes), byteLength=len(emb_bytes))

    accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,
        count=positions.shape[0],
        type="VEC3",
        max=positions.max(axis=0).tolist(),
        min=positions.min(axis=0).tolist(),
    )

    extras = {
        "label": label,
        "language": "house",
        "modalities": modality_counts,
        "k3d": {
            "ids": ids,
            "vectorsView": 0,
            "embeddingsView": 1,
            "embeddingDims": int(embeddings.shape[1]),
            "metadata": metadata,
            "neighbors": [[] for _ in records],
        },
    }

    primitive = Primitive(attributes={"POSITION": 0}, mode=0, extras=extras)
    mesh = Mesh(primitives=[primitive])
    node = Node(mesh=0, name="house_memory")
    scene = Scene(nodes=[0])

    gltf = GLTF2(
        asset=Asset(generator="knowledge3d.tools.house_memory_builder"),
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


def _write_manifest(path: Path, records: Sequence[HouseRecord], positions: np.ndarray, embeddings: np.ndarray) -> None:
    counts: Dict[str, int] = {}
    for record in records:
        art_type = str(record.payload.get("type", "artifact"))
        counts[art_type] = counts.get(art_type, 0) + 1
    manifest = {
        "label": "House Memory",
        "count": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "bounding_box": {
            "min": positions.min(axis=0).tolist(),
            "max": positions.max(axis=0).tolist(),
        },
        "artifact_counts": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_house_memory(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"House materialized directory not found: {root}")
    records = _gather_records(root, args.limit, args.embedding_dim)
    if not records:
        raise ValueError("No artifacts discovered for house memory")

    embeddings = np.vstack([record.embedding for record in records]).astype(np.float32)
    positions = _compute_positions(embeddings)
    glb = _build_glb(records, positions, embeddings, args.label)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb.save(out_path.as_posix())

    if args.manifest:
        _write_manifest(Path(args.manifest).resolve(), records, positions, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build house memory GLB from consolidated artifacts")
    parser.add_argument("--root", default="viewer/public/house/materialized_objects", help="Directory with house artifacts (JSON)")
    parser.add_argument("--out", default="viewer/public/house/house_memory.glb", help="Output GLB path")
    parser.add_argument("--manifest", default="viewer/public/house/house_memory.json", help="Manifest JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on artifacts")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBED_DIM, help="Embedding dimension")
    parser.add_argument("--label", default="House Memory Index", help="Label stored in GLB extras")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry
    args = parse_args()
    build_house_memory(args)


if __name__ == "__main__":  # pragma: no cover
    main()
