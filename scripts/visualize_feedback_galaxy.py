#!/usr/bin/env python3
"""
Render Feedback Galaxy JSONL into a lightweight glTF.
"""

from __future__ import annotations

from array import array
import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List
import random


def _embed_to_vec(embedding: List[float]) -> List[float]:
    if not embedding:
        return [0.0, 0.0, 0.0]
    dim = len(embedding)
    third = max(1, dim // 3)
    x = sum(embedding[:third]) / float(third)
    y = sum(embedding[third : 2 * third]) / float(third)
    z = sum(embedding[2 * third :]) / float(max(1, dim - 2 * third))
    return [x, y, z]


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def build_gltf(entries: List[Dict[str, Any]], *, radius: float = 1.5) -> Dict[str, Any]:
    ids: List[str] = []
    vectors: List[List[float]] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []

    node_color = [1.0, 0.6, 0.0]

    for idx, entry in enumerate(entries):
        embedding = entry.get("embedding") or []
        if embedding:
            base = _embed_to_vec(embedding)
            base = [v * radius for v in base]
        else:
            base = [random.uniform(-1, 1) for _ in range(3)]
        node_id = entry.get("trace_id") or f"feedback_{idx}"
        ids.append(node_id)
        vectors.append(base)
        embeddings.append(embedding)
        metadata.append(
            {
                "trace_id": entry.get("trace_id"),
                "problem_text": entry.get("problem_text"),
                "teacher_score": entry.get("teacher_score"),
                "suggested_rule": entry.get("suggested_rule"),
                "feedback_text": entry.get("feedback_text"),
                "color": node_color,
            }
        )

    positions = array("f")
    for vec in vectors:
        positions.extend(float(v) for v in vec)
    pos_bytes = positions.tobytes()
    pos_b64 = base64.b64encode(pos_bytes).decode("ascii")

    mins = [min(v[i] for v in vectors) for i in range(3)] if vectors else [0.0, 0.0, 0.0]
    maxs = [max(v[i] for v in vectors) for i in range(3)] if vectors else [0.0, 0.0, 0.0]

    k3d_payload = {
        "ids": ids,
        "vectors": vectors,
        "embeddings": embeddings,
        "metadata": metadata,
        "embeddingDims": len(embeddings[0]) if embeddings else 0,
        "embeddingPrecision": "f32",
    }

    return {
        "asset": {"version": "2.0", "generator": "k3d-feedback-galaxy"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "feedback-galaxy"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "mode": 0,
                        "extras": {"k3d": k3d_payload},
                    }
                ]
            }
        ],
        "buffers": [
            {
                "byteLength": len(pos_bytes),
                "uri": f"data:application/octet-stream;base64,{pos_b64}",
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(pos_bytes),
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vectors),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Feedback Galaxy JSONL as glTF.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/feedback_galaxy_v1.jsonl",
        help="Input Feedback Galaxy JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viewer/public/feedback_galaxy.gltf",
        help="Output glTF path.",
    )
    parser.add_argument("--radius", type=float, default=1.5, help="Scale for embedding vectors.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Feedback Galaxy JSONL not found: {input_path}")

    entries = _load_entries(input_path)
    if not entries:
        raise SystemExit("Feedback Galaxy JSONL is empty.")

    gltf = build_gltf(entries, radius=float(args.radius))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gltf, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[FeedbackGalaxy] Wrote {output}")


if __name__ == "__main__":
    main()
