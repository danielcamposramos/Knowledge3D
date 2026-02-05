#!/usr/bin/env python3
"""
Render Log Galaxy JSONL into a lightweight glTF with magenta edges.
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


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _load_feedback_map(path: str) -> Dict[str, Dict[str, Any]]:
    fb_map = {}
    if not path:
        return fb_map
    for entry in _load_entries(Path(path)):
        fb_map[entry.get("trace_id")] = entry
    return fb_map


def build_gltf(entries: List[Dict[str, Any]], *, step_scale: float = 0.25, radius: float = 1.5) -> Dict[str, Any]:
    ids: List[str] = []
    vectors: List[List[float]] = []
    embeddings: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []

    for trace_idx, entry in enumerate(entries):
        # Check for feedback
        feedback = entry.get("_feedback")
        teacher_score = int(feedback.get("teacher_score", 0)) if feedback else 0

        embedding = entry.get("problem_embedding") or []
        if embedding:
            base = _embed_to_vec(embedding)
            base = [v * radius for v in base]
        else:
            # Fallback: random position on sphere if no embedding
            base = [random.uniform(-1, 1) for _ in range(3)]
        policy_mode = (entry.get("metadata") or {}).get("policy_mode") or "heuristic"

        # Determine color based on policy mode or teacher score
        if teacher_score > 0:
            node_color = [0.0, 1.0, 0.0] # Green (Good)
        elif teacher_score < 0:
            node_color = [1.0, 0.0, 0.0] # Red (Bad)
        elif policy_mode == "neural":
            node_color = [1.0, 0.0, 1.0] # Magenta
        elif policy_mode == "mixed":
            node_color = [1.0, 1.0, 0.0] # Yellow
        else:
            node_color = [0.5, 0.5, 0.5] # Grey

        steps = entry.get("step_sequence") or []
        if not steps:
            # Fallback: create a single node for the problem
            node_id = entry.get("trace_id") or f"trace_{trace_idx}"
            ids.append(node_id)
            vectors.append(base)
            embeddings.append(embedding)
            metadata.append(
                {
                    "trace_id": entry.get("trace_id"),
                    "layer": policy_mode,
                    "step_index": 0,
                    "policy_mode": policy_mode,
                    "color": node_color,
                    "teacher_feedback": feedback.get("feedback_text") if feedback else None,
                    "suggested_rule": feedback.get("suggested_rule") if feedback else None,
                }
            )
            continue

        prev_id = None
        for step_idx, step in enumerate(steps):
            node_id = f"{entry.get('trace_id') or 'trace'}:{step_idx}"
            ids.append(node_id)
            vectors.append([base[0], base[1], base[2] + step_idx * step_scale])
            embeddings.append(embedding)
            metadata.append(
                {
                    "trace_id": entry.get("trace_id"),
                    "layer": policy_mode,
                    "step_index": step_idx,
                    "policy_mode": policy_mode,
                    "rule": step.get("rule") or step.get("label"),
                    "color": node_color,
                    "teacher_feedback": feedback.get("feedback_text") if feedback else None,
                }
            )
            if prev_id is not None:
                edges.append((prev_id, node_id))
            prev_id = node_id
        
        # Teacher Gaze: Add a floating teacher node if feedback exists
        if feedback:
            teacher_node_id = f"teacher_{entry.get('trace_id')}"
            teacher_pos = [base[0] * 1.2, base[1] * 1.2, base[2] + (len(steps) * step_scale) + 0.5]
            ids.append(teacher_node_id)
            vectors.append(teacher_pos)
            embeddings.append(embedding)
            metadata.append({
                "type": "teacher_node",
                "trace_id": entry.get("trace_id"),
                "score": teacher_score,
                "color": [0.0, 1.0, 1.0] # Cyan
            })
            # Link from last step to teacher
            edges.append((prev_id, teacher_node_id))

    positions = array("f")
    for vec in vectors:
        positions.extend(float(v) for v in vec)
    pos_bytes = positions.tobytes()
    pos_b64 = base64.b64encode(pos_bytes).decode("ascii")

    mins = [min(v[i] for v in vectors) for i in range(3)]
    maxs = [max(v[i] for v in vectors) for i in range(3)]

    k3d_payload = {
        "ids": ids,
        "vectors": vectors,
        "embeddings": embeddings,
        "metadata": metadata,
        "edges": edges,
        "edgesColor": [1.0, 0.0, 1.0],
        "embeddingDims": len(embeddings[0]) if embeddings else 0,
        "embeddingPrecision": "f32",
    }

    return {
        "asset": {"version": "2.0", "generator": "k3d-log-galaxy"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "log-galaxy"}],
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
    parser = argparse.ArgumentParser(description="Visualize Log Galaxy JSONL as glTF.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/log_galaxy_neural_v1.jsonl",
        help="Input Log Galaxy JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viewer/public/log_galaxy_neural_v1.gltf",
        help="Output glTF path.",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default=None,
        help="Optional Feedback Galaxy JSONL to overlay.",
    )
    parser.add_argument("--step-scale", type=float, default=0.25, help="Step spacing in Z.")
    parser.add_argument("--radius", type=float, default=1.5, help="Scale for embedding vectors.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Log Galaxy JSONL not found: {input_path}")

    entries = _load_entries(input_path)
    if not entries:
        raise SystemExit("Log Galaxy JSONL is empty.")

    if args.feedback:
        fb_map = _load_feedback_map(args.feedback)
        for entry in entries:
            tid = entry.get("trace_id")
            if tid in fb_map:
                entry["_feedback"] = fb_map[tid]

    gltf = build_gltf(entries, step_scale=float(args.step_scale), radius=float(args.radius))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gltf, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[LogGalaxy] Wrote {output}")


if __name__ == "__main__":
    main()
