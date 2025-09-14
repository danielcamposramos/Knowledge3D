from __future__ import annotations

"""
Consolidation Engine (Phase 5.1)

Input: Star dict { id, embedding, shape_type }
Process:
  1) Honesty check: embedding[72] > 0.7 ⇒ crystallize, else gray shell
  2) Compression: keep top-K% dims by absolute magnitude (simple PCA-like proxy)
  3) Furniture mapping: shape_type → furniture kind (tetrahedron→bookshelf, icosahedron→display_case, etc.)
  4) Dishonest → semi‑transparent gray shell with error embedding

Output: dict suitable for bathtub_registry.json entries with:
  - id, star_id, shape_type, furniture_kind
  - is_crystallized, honesty_score, compression_factor
  - embedding (compressed or error marker)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import time


@dataclass
class ConsolidationResult:
    id: str
    star_id: str
    shape_type: str
    furniture_kind: str
    is_crystallized: bool
    honesty_score: float
    compression_factor: float
    embedding: List[float]
    status: str  # 'crystallized' | 'dishonest'
    ts: float


_FURNITURE_MAP = {
    'tetrahedron': 'bookshelf',
    'icosahedron': 'display_case',
    'cube': 'workbench',
    'sphere': 'lamp',
    'cylinder': 'pillar',
}


def _honesty_score(embedding: List[float]) -> float:
    if not embedding:
        return 0.0
    i = 72
    if i < len(embedding):
        try:
            return float(embedding[i])
        except Exception:
            return 0.0
    return 0.0


def _compress_embedding(embedding: List[float], keep_ratio: float = 0.5) -> Tuple[List[float], float]:
    """Keep top-K% dims by |value|. Returns (compressed, compression_factor)."""
    if not embedding:
        return [], 0.0
    v = list(map(float, embedding))
    n = len(v)
    k = max(1, min(n, int(math.ceil(n * max(0.0, min(1.0, keep_ratio))))))
    # Rank by absolute magnitude; keep top-k, zero the rest
    order = sorted(range(n), key=lambda i: abs(v[i]), reverse=True)
    keep_idx = set(order[:k])
    out = [ (v[i] if i in keep_idx else 0.0) for i in range(n) ]
    return out, (k / float(n))


def _map_furniture(shape_type: str, honest: bool) -> str:
    s = (shape_type or '').lower().strip()
    if not honest:
        return 'gray_shell'
    return _FURNITURE_MAP.get(s, 'artifact')


class ConsolidationEngine:
    def __init__(self, keep_ratio: float = 0.5):
        self.keep_ratio = keep_ratio

    def consolidate_star(self, star: Dict) -> ConsolidationResult:
        star_id = str(star.get('id') or star.get('star_id') or 'star:unknown')
        shape_type = str(star.get('shape_type') or 'tetrahedron')
        emb = star.get('embedding') or []
        if not isinstance(emb, list):
            try:
                emb = list(emb)
            except Exception:
                emb = []
        honesty = _honesty_score(emb)
        honest_ok = (honesty > 0.7)
        if honest_ok:
            comp, cf = _compress_embedding(emb, keep_ratio=self.keep_ratio)
        else:
            # Dishonest: produce a small error embedding signature
            comp = [0.0 for _ in range(len(emb))] or [0.0]
            cf = 0.0
        kind = _map_furniture(shape_type, honest_ok)
        rid = f"furn:{abs(hash(star_id + '|' + kind)) % (10**10):010d}"
        return ConsolidationResult(
            id=rid,
            star_id=star_id,
            shape_type=shape_type,
            furniture_kind=kind,
            is_crystallized=honest_ok,
            honesty_score=float(honesty),
            compression_factor=float(cf),
            embedding=[float(x) for x in comp],
            status=('crystallized' if honest_ok else 'dishonest'),
            ts=time.time(),
        )

    def to_registry_item(self, res: ConsolidationResult) -> Dict:
        """Convert result to a registry JSON dict for bathtub_renderer.

        Position/rotation are left for the renderer to arrange; we include optional hints.
        """
        return {
            'id': res.id,
            'star_id': res.star_id,
            'shape_type': res.shape_type,
            'furniture_kind': res.furniture_kind,
            'is_crystallized': res.is_crystallized,
            'honesty_score': res.honesty_score,
            'compression_factor': res.compression_factor,
            'embedding': res.embedding,
            'status': res.status,
            'ts': res.ts,
        }

