#!/usr/bin/env python3
"""
Visualize Sleep Galaxy JSONL as glTF.

Keep nodes: tetrahedrons (text modality) with high alpha.
Discard nodes: octahedrons (audio modality) with low alpha ("dust").
Uncertain nodes: boxes (image modality) with mid alpha.
"""

from __future__ import annotations

from array import array
import argparse
import base64
import json
import random
from pathlib import Path
from typing import Any, Dict, List


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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _decision_style(entry: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(entry.get("decision", "uncertain")).lower()
    meta = entry.get("metadata") or {}
    negative_wisdom = bool(meta.get("negative_wisdom", False))
    if negative_wisdom and decision == "keep":
        return {"type": "audio", "alpha": 0.9, "color": [0.8, 0.1, 0.1], "offset": 1.0}
    if decision == "keep":
        return {"type": "text", "alpha": 1.0, "color": [0.6, 0.0, 0.8], "offset": 2.0}
    if decision == "discard":
        return {"type": "audio", "alpha": 0.3, "color": [0.5, 0.5, 0.5], "offset": -2.0}
    return {"type": "image", "alpha": 0.6, "color": [0.9, 0.7, 0.2], "offset": 0.0}


def build_gltf(entries: List[Dict[str, Any]], *, radius: float = 1.5) -> Dict[str, Any]:
    ids: List[str] = []
    vectors: List[List[float]] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    alpha_mask: List[float] = []

    for idx, entry in enumerate(entries):
        style = _decision_style(entry)
        embedding = entry.get("embedding") or []
        if embedding:
            base = _embed_to_vec(embedding)
            base = [v * radius for v in base]
        else:
            base = [random.uniform(-1, 1) for _ in range(3)]
        base[2] += float(style["offset"])

        trace_id = entry.get("trace_id") or f"trace_{idx}"
        ids.append(trace_id)
        vectors.append(base)
        embeddings.append(embedding)
        alpha_mask.append(float(style["alpha"]))
        meta = dict(entry.get("metadata") or {})
        meta.update(
            {
                "trace_id": trace_id,
                "decision": str(entry.get("decision", "uncertain")).lower(),
                "decision_score": entry.get("decision_score"),
                "type": style["type"],
                "color": style["color"],
            }
        )
        metadata.append(meta)

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
        "temporal": {"alphaMask": alpha_mask},
    }

    return {
        "asset": {"version": "2.0", "generator": "k3d-sleep-galaxy"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "sleep-galaxy"}],
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
    parser = argparse.ArgumentParser(description="Visualize Sleep Galaxy JSONL as glTF.")
    parser.add_argument("--input", required=True, help="Sleep Galaxy JSONL path.")
    parser.add_argument("--output", required=True, help="Output glTF path.")
    parser.add_argument("--radius", type=float, default=1.5, help="Scale for embedding vectors.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Sleep Galaxy JSONL not found: {input_path}")

    entries = _load_entries(input_path)
    if not entries:
        raise SystemExit("Sleep Galaxy JSONL is empty.")

    gltf = build_gltf(entries, radius=float(args.radius))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gltf, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[SleepGalaxy] Wrote {output}")


if __name__ == "__main__":
    main()
