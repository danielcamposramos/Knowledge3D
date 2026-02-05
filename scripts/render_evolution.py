#!/usr/bin/env python3
"""
Render an evolution glTF showing V1 -> V4 trace crystallization.
"""

from __future__ import annotations

from array import array
import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
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


def _load_traces(path: Path) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))
    return traces


def _collect_nodes(
    traces: List[Dict[str, Any]],
    *,
    phase: str,
    z_offset: float,
    step_scale: float,
    radius: float,
    color: List[float],
) -> Tuple[List[str], List[List[float]], List[List[float]], List[Dict[str, Any]], List[Tuple[str, str]]]:
    ids: List[str] = []
    vectors: List[List[float]] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []

    for trace_idx, entry in enumerate(traces):
        embedding = entry.get("problem_embedding") or []
        if embedding:
            base = _embed_to_vec(embedding)
            base = [v * radius for v in base]
        else:
            base = [random.uniform(-1, 1) for _ in range(3)]
        base[2] += z_offset
        steps = entry.get("step_sequence") or []
        prev_id = None
        for step_idx, step in enumerate(steps):
            node_id = f"{phase}:{entry.get('trace_id', trace_idx)}:{step_idx}"
            ids.append(node_id)
            vectors.append([base[0], base[1], base[2] + step_idx * step_scale])
            embeddings.append(embedding)
            metadata.append(
                {
                    "phase": phase,
                    "trace_id": entry.get("trace_id"),
                    "step_index": step_idx,
                    "rule": step.get("rule") or step.get("label"),
                    "color": color,
                }
            )
            if prev_id is not None:
                edges.append((prev_id, node_id))
            prev_id = node_id
    return ids, vectors, embeddings, metadata, edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evolution of log traces V1->V4.")
    parser.add_argument("--v1", default="data/log_galaxy_neural_v1.jsonl", help="V1 log JSONL.")
    parser.add_argument("--v4", default="data/log_galaxy_neural_v4.jsonl", help="V4 log JSONL.")
    parser.add_argument("--output", required=True, help="Output GLTF path.")
    parser.add_argument("--step-scale", type=float, default=0.25, help="Step spacing in Z.")
    parser.add_argument("--radius", type=float, default=1.5, help="Scale for embedding vectors.")
    parser.add_argument("--phase-gap", type=float, default=3.0, help="Z gap between phases.")
    args = parser.parse_args()

    v1_traces = _load_traces(Path(args.v1))
    v4_traces = _load_traces(Path(args.v4))

    ids: List[str] = []
    vectors: List[List[float]] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []

    for phase, traces, offset, color in [
        ("v1", v1_traces, 0.0, [0.5, 0.5, 0.5]),
        ("v4", v4_traces, float(args.phase_gap), [1.0, 0.0, 1.0]),
    ]:
        part_ids, part_vecs, part_embeds, part_meta, part_edges = _collect_nodes(
            traces,
            phase=phase,
            z_offset=offset,
            step_scale=float(args.step_scale),
            radius=float(args.radius),
            color=color,
        )
        ids.extend(part_ids)
        vectors.extend(part_vecs)
        embeddings.extend(part_embeds)
        metadata.extend(part_meta)
        edges.extend(part_edges)

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
        "edges": edges,
        "edgesColor": [0.7, 0.7, 0.7],
        "embeddingDims": len(embeddings[0]) if embeddings else 0,
        "embeddingPrecision": "f32",
        "temporal": {
            "alpha": 1.0,
        },
    }

    gltf = {
        "asset": {"version": "2.0", "generator": "k3d-evolution"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "evolution"}],
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gltf, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[Evolution] Wrote {output}")


if __name__ == "__main__":
    main()
