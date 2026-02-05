#!/usr/bin/env python3
"""
Visualize Skill Galaxy JSONL files as a 3D constellation of crystals.
Supports multiple input files with version-coded colors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List
import array
import base64

def _embed_to_vec(embedding: List[float]) -> List[float]:
    if not embedding:
        return [0.0, 0.0, 0.0]
    dim = len(embedding)
    # Simple projection to 3D for visualization
    third = max(1, dim // 3)
    x = sum(embedding[:third]) / float(third)
    y = sum(embedding[third : 2 * third]) / float(third)
    z = sum(embedding[2 * third :]) / float(max(1, dim - 2 * third))
    
    # Normalize and scale
    mag = math.sqrt(x*x + y*y + z*z) + 1e-9
    scale = 5.0 
    return [(x/mag) * scale, (y/mag) * scale, (z/mag) * scale]

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Skill Galaxy.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input skill galaxy JSONL files.")
    parser.add_argument(
        "--output",
        type=str,
        default="viewer/public/skill_galaxy.gltf",
        help="Output glTF path.",
    )
    args = parser.parse_args()

    colors = {
        "v1": [0.0, 0.5, 1.0],  # Blue
        "v2": [0.0, 1.0, 1.0],  # Cyan
        "v3": [0.0, 1.0, 0.0],  # Green
        "v4": [1.0, 0.0, 1.0],  # Magenta
        "router": [1.0, 0.6, 0.0],  # Orange (router skills)
        "default": [1.0, 1.0, 1.0],
    }

    ids = []
    vectors = []
    embeddings = []
    metadata = []
    versions_seen = set()
    
    for input_file in args.inputs:
        path = Path(input_file)
        if not path.exists():
            continue
            
        version = "default"
        if "v1" in path.name: version = "v1"
        elif "v2" in path.name: version = "v2"
        elif "v3" in path.name: version = "v3"
        elif "v4" in path.name: version = "v4"
        
        color = colors.get(version, colors["default"])

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    skill_id = entry.get("skill_id", "unknown")
                    embedding = entry.get("embedding", [])
                    meta = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
                    is_router = False
                    if "router" in str(skill_id).lower():
                        is_router = True
                    if str(meta.get("role", "")).lower() == "router_gatekeeper":
                        is_router = True
                    entry_version = "router" if is_router else version
                    entry_color = colors.get(entry_version, color)
                    
                    if not embedding:
                        pos = [random.uniform(-2, 2) for _ in range(3)]
                    else:
                        pos = _embed_to_vec(embedding)
                    
                    ids.append(skill_id)
                    vectors.append(pos)
                    embeddings.append(embedding)
                    metadata.append({
                        "skill_id": skill_id,
                        "version": entry_version,
                        "description": meta.get("description", ""),
                        "color": entry_color
                    })
                    versions_seen.add(entry_version)
                    
                except json.JSONDecodeError:
                    continue

    # Add Legend Nodes
    legend_x = -5.0
    legend_y = 5.0
    for ver, col in colors.items():
        if ver == "default":
            continue
        if versions_seen and ver not in versions_seen:
            continue
        ids.append(f"Legend_{ver.upper()}")
        vectors.append([legend_x, legend_y, 0.0])
        embeddings.append([0.0] * (len(embeddings[0]) if embeddings else 8))
        metadata.append({
            "skill_id": f"Legend_{ver.upper()}",
            "version": ver,
            "description": f"Color Key for {ver}",
            "color": col
        })
        legend_y -= 1.0

    flat_pos = array.array('f')
    for v in vectors:
        flat_pos.extend(v)
    
    pos_bytes = flat_pos.tobytes()
    pos_b64 = base64.b64encode(pos_bytes).decode("ascii")
    
    if vectors:
        mins = [min(v[i] for v in vectors) for i in range(3)]
        maxs = [max(v[i] for v in vectors) for i in range(3)]
    else:
        mins = [0.0, 0.0, 0.0]
        maxs = [0.0, 0.0, 0.0]

    k3d_payload = {
        "ids": ids,
        "vectors": vectors,
        "embeddings": embeddings,
        "metadata": metadata,
        "embeddingDims": len(embeddings[0]) if embeddings else 0,
        "embeddingPrecision": "f32"
    }

    gltf = {
        "asset": {"version": "2.0", "generator": "k3d-skill-galaxy"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "skill-galaxy"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "mode": 0, 
                        "extras": {"k3d": k3d_payload}
                    }
                ]
            }
        ],
        "buffers": [
            {
                "byteLength": len(pos_bytes),
                "uri": f"data:application/octet-stream;base64,{pos_b64}"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(pos_bytes)
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126, 
                "count": len(vectors),
                "type": "VEC3",
                "min": mins,
                "max": maxs
            }
        ]
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)
    
    print(f"[SkillGalaxy] Wrote {args.output}")


if __name__ == "__main__":
    main()
