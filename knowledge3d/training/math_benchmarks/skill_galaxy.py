"""
Skill Galaxy: store learned behavior weights as Galaxy nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from array import array
import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from knowledge3d.training.math_benchmarks.router_embedder import embed_text


@dataclass
class SkillGalaxyEntry:
    skill_id: str
    embedding: List[float]
    geometry: str
    payload_b64: str
    payload_format: str = "torch_state_dict"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "embedding": self.embedding,
            "geometry": self.geometry,
            "payload_b64": self.payload_b64,
            "payload_format": self.payload_format,
            "metadata": self.metadata,
        }


class SkillGalaxy:
    """
    In-memory Skill Galaxy container with JSONL export.
    """

    def __init__(self, *, embedding_dim: int = 256):
        self.embedding_dim = int(embedding_dim)
        self.entries: List[SkillGalaxyEntry] = []

    def add_skill(
        self,
        *,
        skill_id: str,
        description: str,
        payload: bytes,
        geometry: str = "crystal",
        payload_format: str = "torch_state_dict",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkillGalaxyEntry:
        embedding = embed_text(description, dim=self.embedding_dim)
        payload_b64 = base64.b64encode(payload).decode("ascii")
        meta = dict(metadata or {})
        meta.setdefault("description", description)
        meta.setdefault("label", skill_id)
        meta.setdefault("geometry", geometry)
        meta.setdefault("color", [0.0, 1.0, 1.0])
        entry = SkillGalaxyEntry(
            skill_id=skill_id,
            embedding=embedding,
            geometry=geometry,
            payload_b64=payload_b64,
            payload_format=payload_format,
            metadata=meta,
        )
        self.entries.append(entry)
        return entry

    def load_skill(self, skill_id: str) -> Optional[bytes]:
        for entry in self.entries:
            if entry.skill_id == skill_id:
                return base64.b64decode(entry.payload_b64.encode("ascii"))
        return None

    def to_jsonl(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")

    def to_gltf(self, path: str) -> None:
        """
        Export a minimal glTF with embedded extras.k3d for visualization.
        """
        if not self.entries:
            raise ValueError("SkillGalaxy has no entries to export.")

        ids: List[str] = []
        vectors: List[List[float]] = []
        embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        radius = 1.5
        for idx, entry in enumerate(self.entries):
            angle = (2.0 * math.pi * idx) / max(1, len(self.entries))
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            z = 0.0
            ids.append(entry.skill_id)
            vectors.append([x, y, z])
            embeddings.append(entry.embedding)

            meta = dict(entry.metadata)
            meta.setdefault("skill_id", entry.skill_id)
            meta.setdefault("description", entry.metadata.get("description"))
            meta.setdefault("type", "video")  # map to icosa geometry in viewer
            meta.setdefault("payload_b64", entry.payload_b64)
            meta.setdefault("payload_format", entry.payload_format)
            metadata.append(meta)

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
            "embeddingDims": len(embeddings[0]) if embeddings else 0,
            "embeddingPrecision": "f32",
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

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(gltf, ensure_ascii=True, indent=2), encoding="utf-8")


__all__ = ["SkillGalaxy", "SkillGalaxyEntry"]
