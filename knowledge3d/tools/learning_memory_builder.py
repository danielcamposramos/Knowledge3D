from __future__ import annotations

"""Build a PTX-ready learning memory galaxy from teacher feedback JSONL records."""

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


@dataclass
class LearningRecord:
    record_id: str
    label: str
    embedding: np.ndarray
    payload: Dict[str, object]
    tags: Sequence[str]


def _hash_embedding(text_parts: Iterable[str], dim: int = 512) -> np.ndarray:
    joined = "\u241f".join(part.strip() for part in text_parts if part)
    if not joined:
        joined = "learning_memory"
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


def _load_records(paths: Sequence[Path], limit: Optional[int], embedding_dim: int) -> List[LearningRecord]:
    records: List[LearningRecord] = []
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and count >= limit:
                    return records
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(data.get("prompt", ""))
                true_answer = str(data.get("true_answer", ""))
                predicted = str(data.get("predicted", ""))
                quick = json.dumps(data.get("quick_feedback", {}), ensure_ascii=False)
                deep = json.dumps(data.get("deep_feedback", {}), ensure_ascii=False)
                prompt_key = prompt.strip().lower()
                if not prompt_key:
                    continue
                embedding = _hash_embedding([prompt_key], dim=embedding_dim)
                record_id = str(data.get("id") or f"learning_{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]}")
                label = prompt[:80] or record_id
                payload = {
                    "prompt": prompt,
                    "true_answer": true_answer,
                    "predicted": predicted,
                    "score": data.get("score"),
                    "quick_feedback": data.get("quick_feedback"),
                    "deep_feedback": data.get("deep_feedback"),
                    "timestamp": data.get("timestamp"),
                    "language": data.get("language"),
                }
                tags = data.get("concepts") or data.get("tags") or []
                records.append(
                    LearningRecord(
                        record_id=record_id,
                        label=label,
                        embedding=embedding,
                        payload=payload,
                        tags=list(tags) if isinstance(tags, list) else [str(tags)],
                    )
                )
                count += 1
    return records


def _compute_positions(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings to position")
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    if centred.shape[1] < 3:
        pad = np.zeros((centred.shape[0], 3 - centred.shape[1]), dtype=np.float32)
        return np.concatenate([centred, pad], axis=1)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:3, :].T
    coords = centred @ components
    return coords.astype(np.float32)


def _build_glb(records: Sequence[LearningRecord], positions: np.ndarray, embeddings: np.ndarray, label: str) -> GLTF2:
    ids = [record.record_id for record in records]
    metadata = []
    modality_counts: Dict[str, int] = {}
    for record in records:
        metadata.append(
            {
                "name": record.label,
                "tags": list(record.tags),
                "payload": record.payload,
                "modality": "learning_memory",
            }
        )
        modality_counts.setdefault("text", 0)
        modality_counts["text"] += 1

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
        "language": "learning",
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
    node = Node(mesh=0, name="learning_memory")
    scene = Scene(nodes=[0])

    gltf = GLTF2(
        asset=Asset(generator="knowledge3d.tools.learning_memory_builder"),
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


def _write_manifest(path: Path, records: Sequence[LearningRecord], positions: np.ndarray, embeddings: np.ndarray) -> None:
    bbox = {
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
    }
    manifest = {
        "label": "Learning Memory",
        "language": "learning",
        "count": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "bounding_box": bbox,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_learning_memory(args: argparse.Namespace) -> None:
    input_paths = [Path(p).resolve() for p in args.input]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {path}")

    records = _load_records(input_paths, args.limit, args.embedding_dim)
    if not records:
        raise ValueError("No learning records parsed from inputs")

    embeddings = np.vstack([record.embedding for record in records]).astype(np.float32)
    positions = _compute_positions(embeddings)

    glb = _build_glb(records, positions, embeddings, args.label)
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.save(output_path.as_posix())

    if args.manifest:
        _write_manifest(Path(args.manifest).resolve(), records, positions, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build learning memory galaxy from JSONL logs")
    parser.add_argument("--input", action="append", required=True, help="Learning memory JSONL input (repeatable)")
    parser.add_argument("--out", required=True, help="Output GLB path")
    parser.add_argument("--manifest", help="Optional manifest JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit")
    parser.add_argument("--label", default="Learning Memory Galaxy", help="Label stored in GLB extras")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    build_learning_memory(args)


if __name__ == "__main__":  # pragma: no cover
    main()
