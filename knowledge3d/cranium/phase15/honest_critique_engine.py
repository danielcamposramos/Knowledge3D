from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

from pygltflib import GLTF2  # type: ignore


class HonestCritiqueEngine:
    """Simple pre‑consolidation critique engine.

    Loads GLB, inspects extras.k3d.honesty_score and returns critique suggestions
    for items under the threshold.
    """

    def __init__(self, material_dir: str, galaxy_path: str):
        self.material_dir = Path(material_dir)
        self.galaxy_path = Path(galaxy_path)
        self.threshold = 0.85

    def _read_honesty(self, glb_path: str) -> float:
        try:
            gltf = GLTF2().load(glb_path)
            for n in (gltf.nodes or []):
                ex = getattr(n, 'extras', None)
                if hasattr(ex, 'to_dict'):
                    try:
                        ex = ex.to_dict()
                    except Exception:
                        ex = dict(ex)
                if isinstance(ex, dict) and isinstance(ex.get('k3d'), dict):
                    h = ex['k3d'].get('honesty_score')
                    if isinstance(h, (int, float)):
                        return float(h)
        except Exception:
            pass
        return 0.5

    def critique_shapes(self, shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for s in shapes:
            p = str(s.get('path') or '')
            if not p:
                continue
            h = self._read_honesty(p)
            if h < self.threshold:
                results.append({
                    'path': p,
                    'issue': 'low_honesty',
                    'honesty': h,
                    'suggestion': 'increase_honesty',
                })
        return results

