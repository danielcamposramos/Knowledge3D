from __future__ import annotations

"""
Asset Upgrade Engine (Phase 5.2)

Enables self-enhancing assets: when understanding (embedding dims) and honesty
are high enough, replace simple proxy furniture with semantically richer
geometry data, preserving embeddings and metadata.

Registry furniture schema (bathtub_registry.json):
  {
    id, star_id, shape_type, furniture_kind,
    is_crystallized, honesty_score, compression_factor,
    embedding: [...],
    status: 'crystallized'|'dishonest',
    ts,
    # Optional fields added by upgrade:
    is_enhanced: true,
    upgrade_timestamp: ISO8601,
    geometry: { type, vertices, indices, material: {...} }
  }
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import math


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def sigmoid(x: float) -> float:
    try:
        return float(1.0 / (1.0 + math.exp(-float(x))))
    except Exception:
        return 0.5


@dataclass
class Geometry:
    type: str
    vertices: List[float]
    indices: List[int]
    material: Dict[str, float]


class AssetUpgradeEngine:
    def __init__(self) -> None:
        self.upgrade_rules = self.load_upgrade_rules()
        self.asset_library = self.load_asset_library()

    def load_upgrade_rules(self) -> Dict[str, Dict]:
        return {
            'bookshelf': {
                'simple': 'cube',
                'enhanced': 'detailed_bookshelf',
                'trigger_embedding_dim': 128,
                'min_honesty': 0.9,
                'upgrade_prompt': "Generate a detailed bookshelf with carved wood, glass doors, and individual shelf embeddings.",
            },
            'display_case': {
                'simple': 'icosahedron',
                'enhanced': 'glass_display_case',
                'trigger_embedding_dim': 256,
                'min_honesty': 0.85,
                'upgrade_prompt': "Generate a glass-and-metal display case with embedded lighting and per-item embeddings.",
            },
        }

    def load_asset_library(self) -> Dict[str, Callable[[List[float]], Geometry]]:
        return {
            'detailed_bookshelf': self.generate_detailed_bookshelf,
            'glass_display_case': self.generate_glass_display_case,
        }

    def should_upgrade_asset(self, furniture: Dict) -> bool:
        # Determine furniture 'type'
        ftype = str(furniture.get('furniture_kind') or furniture.get('type') or '').strip().lower()
        if not ftype:
            return False
        rule = self.upgrade_rules.get(ftype)
        if not rule:
            return False
        # Embedding dimensionality
        emb = furniture.get('embedding') or []
        if not isinstance(emb, list) or len(emb) < int(rule.get('trigger_embedding_dim', 9999)):
            return False
        # Honesty threshold
        if float(furniture.get('honesty_score') or 0.0) < float(rule.get('min_honesty', 1.0)):
            return False
        # Crystallized only
        if not bool(furniture.get('is_crystallized', False)):
            return False
        # Avoid re-upgrading
        if bool(furniture.get('is_enhanced', False)):
            return False
        return True

    def upgrade_asset(self, furniture: Dict) -> Dict:
        if not self.should_upgrade_asset(furniture):
            return furniture
        ftype = str(furniture.get('furniture_kind') or '').strip().lower()
        rule = self.upgrade_rules.get(ftype)
        if not rule:
            return furniture
        enhanced_key = str(rule.get('enhanced'))
        gen = self.asset_library.get(enhanced_key)
        if not gen:
            return furniture
        emb: List[float] = [float(x) for x in (furniture.get('embedding') or [])]
        geo = gen(emb)
        out = dict(furniture)
        out['geometry'] = {
            'type': geo.type,
            'vertices': list(map(float, geo.vertices)),
            'indices': list(map(int, geo.indices)),
            'material': dict(geo.material),
        }
        out['is_enhanced'] = True
        out['upgrade_timestamp'] = datetime.utcnow().isoformat() + 'Z'
        return out

    # --- Geometry generators ---
    def generate_detailed_bookshelf(self, embedding: List[float]) -> Geometry:
        # Use embedding to control details
        num_shelves = max(3, min(9, int(abs(embedding[0]) * 7))) if embedding else 5
        wood_grain = clamp(abs(embedding[1]) if len(embedding) > 1 else 0.4, 0.0, 1.0)
        glass_doors = sigmoid(embedding[2] if len(embedding) > 2 else 0.0) > 0.5

        # Base frame: width x height x depth
        W, H, D = 1.0, 1.6, 0.28
        # Vertices for a box centered at origin
        def _box(w, h, d) -> Tuple[List[float], List[int]]:
            x, y, z = w * 0.5, h * 0.5, d * 0.5
            v = [
                -x, -y, -z,  x, -y, -z,  x,  y, -z, -x,  y, -z,
                -x, -y,  z,  x, -y,  z,  x,  y,  z, -x,  y,  z,
            ]
            i = [
                0,1,2, 2,3,0,  4,7,6, 6,5,4,
                0,4,5, 5,1,0,  2,6,7, 7,3,2,
                0,3,7, 7,4,0,  1,5,6, 6,2,1,
            ]
            return v, i

        vertices: List[float] = []
        indices: List[int] = []
        # Outer frame
        v0, i0 = _box(W, H, D)
        vertices += v0
        indices += i0
        base_vert = len(v0) // 3
        # Shelves as thin boxes
        shelf_th = 0.04
        for s in range(num_shelves):
            y = -H * 0.45 + s * (H * 0.9 / max(1, (num_shelves - 1)))
            vS, iS = _box(W * 0.9, shelf_th, D * 0.9)
            # translate shelf to y
            for j in range(0, len(vS), 3):
                vS[j + 1] += y
            # reindex
            off = len(vertices) // 3
            indices += [off + ii for ii in iS]
            vertices += vS

        material = {
            'wood_grain': wood_grain,
            'glass_doors': 1.0 if glass_doors else 0.0,
            'color_r': sigmoid(embedding[3] if len(embedding) > 3 else 0.2),
            'color_g': sigmoid(embedding[4] if len(embedding) > 4 else 0.2),
            'color_b': sigmoid(embedding[5] if len(embedding) > 5 else 0.2),
        }
        return Geometry('detailed_bookshelf', vertices, indices, material)

    def generate_glass_display_case(self, embedding: List[float]) -> Geometry:
        # Parameters from embedding
        num_panels = max(4, min(12, int(abs(embedding[0]) * 10))) if embedding else 6
        metal_finish = clamp(abs(embedding[1]) if len(embedding) > 1 else 0.5, 0.0, 1.0)
        lighting = clamp(sigmoid(embedding[2] if len(embedding) > 2 else 0.0), 0.0, 1.0)

        # Cylinder-like frame made of panels
        R, H = 0.45, 1.4
        verts: List[float] = []
        idx: List[int] = []
        # Vertical struts
        strut_w = 0.04
        for p in range(num_panels):
            a = (p / num_panels) * 2.0 * math.pi
            x, z = R * math.cos(a), R * math.sin(a)
            # Each strut is a thin box
            w, h, d = strut_w, H, strut_w
            sx, sy, sz = w * 0.5, h * 0.5, d * 0.5
            v = [
                -sx, -sy, -sz,  sx, -sy, -sz,  sx,  sy, -sz, -sx,  sy, -sz,
                -sx, -sy,  sz,  sx, -sy,  sz,  sx,  sy,  sz, -sx,  sy,  sz,
            ]
            # rotate/translate into place
            for j in range(0, len(v), 3):
                px, py, pz = v[j], v[j + 1], v[j + 2]
                # rotate around Y by angle a
                rx = px * math.cos(a) + pz * math.sin(a)
                rz = -px * math.sin(a) + pz * math.cos(a)
                v[j], v[j + 1], v[j + 2] = rx + x, py, rz + z
            off = len(verts) // 3
            verts += v
            idx += [off + ii for ii in [
                0,1,2, 2,3,0,  4,7,6, 6,5,4,
                0,4,5, 5,1,0,  2,6,7, 7,3,2,
                0,3,7, 7,4,0,  1,5,6, 6,2,1,
            ]]

        material = {
            'metal_finish': metal_finish,
            'lighting_intensity': lighting,
            'color_r': sigmoid(embedding[3] if len(embedding) > 3 else 0.7),
            'color_g': sigmoid(embedding[4] if len(embedding) > 4 else 0.8),
            'color_b': sigmoid(embedding[5] if len(embedding) > 5 else 0.9),
        }
        return Geometry('glass_display_case', verts, idx, material)

