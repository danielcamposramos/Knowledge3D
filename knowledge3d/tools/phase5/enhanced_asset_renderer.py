from __future__ import annotations

"""
Enhanced Asset Renderer (Phase 5.2)

Prepares material properties per sub-component derived from embeddings.
For MVP, returns computed material property arrays; GLB emission remains
single-material unless the caller splits primitives.
"""

from dataclasses import dataclass
from typing import List
import math


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def sigmoid(x: float) -> float:
    try:
        return float(1.0 / (1.0 + math.exp(-float(x))))
    except Exception:
        return 0.5


@dataclass
class MaterialProperties:
    roughness: float
    metallic: float
    albedo: List[float]  # [r,g,b]
    emission: float


@dataclass
class MeshData:
    vertices: List[float]
    indices: List[int]
    materials: List[MaterialProperties]


class EnhancedAssetRenderer:
    def render_enhanced_asset(self, furniture: dict) -> MeshData:
        geom = furniture.get('geometry') or {}
        vertices = list(geom.get('vertices') or [])
        indices = list(geom.get('indices') or [])
        materials: List[MaterialProperties] = []
        if not vertices:
            return MeshData(vertices, indices, materials)
        # Bind sub-component materials every 30 vertices (10 triangles)
        for i in range(0, len(vertices) // 3):
            sub_idx = self.get_sub_component_idx(i, furniture)
            sub_embed = self.get_sub_component_embedding(furniture, sub_idx)
            materials.append(self.generate_material_from_embedding(sub_embed))
        return MeshData(vertices, indices, materials)

    def get_sub_component_idx(self, vertex_idx: int, furniture: dict) -> int:
        return int(vertex_idx // 30)

    def get_sub_component_embedding(self, furniture: dict, sub_component_idx: int) -> List[float]:
        base = list(furniture.get('embedding') or [])
        if not base:
            return [0.0] * 8
        out: List[float] = []
        offset = int(sub_component_idx) * 8
        for i in range(8):
            out.append(float(base[(offset + i) % len(base)]))
        return out

    def generate_material_from_embedding(self, embedding: List[float]) -> MaterialProperties:
        e = embedding + [0.0] * (6 - len(embedding))
        return MaterialProperties(
            roughness=clamp(abs(e[0]), 0.0, 1.0),
            metallic=clamp(abs(e[1]), 0.0, 1.0),
            albedo=[sigmoid(e[2]), sigmoid(e[3]), sigmoid(e[4])],
            emission=clamp(float(e[5]) * 2.5, 0.0, 10.0),
        )

