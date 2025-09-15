from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from pygltflib import GLTF2  # type: ignore


class CritiqueApplier:
    """Applies simple critique suggestions to GLBs by adjusting extras.k3d.honesty_score
    and writing a versioned copy (…_vN.glb) into the working directory.
    """

    def __init__(self, material_dir: str, galaxy_working_dir: str):
        self.material_dir = Path(material_dir)
        self.galaxy_working_dir = Path(galaxy_working_dir)
        self.galaxy_working_dir.mkdir(parents=True, exist_ok=True)

    def apply_shape_critique(self, glb_path: str, revision: int, delta: float = 0.15) -> str:
        p = Path(glb_path)
        try:
            gltf = GLTF2().load(str(p))
        except Exception:
            raise RuntimeError(f"Failed to load GLB: {glb_path}")
        # bump honesty on first node with extras.k3d
        changed = False
        for n in (gltf.nodes or []):
            ex = getattr(n, 'extras', None)
            if hasattr(ex, 'to_dict'):
                try:
                    ex = ex.to_dict()
                except Exception:
                    ex = dict(ex)
            if isinstance(ex, dict):
                k3d = ex.get('k3d') if isinstance(ex.get('k3d'), dict) else None
                if isinstance(k3d, dict):
                    h = k3d.get('honesty_score', 0.5)
                    try:
                        nh = float(h) + float(delta)
                    except Exception:
                        nh = 0.5 + delta
                    k3d['honesty_score'] = float(min(1.0, nh))
                    # restore back into node extras
                    n.extras = {'k3d': k3d}
                    changed = True
                    break
        if not changed:
            # add a minimal extras if none exists
            n0 = gltf.nodes[0] if gltf.nodes else None
            if n0 is not None:
                n0.extras = {'k3d': {'honesty_score': float(min(1.0, delta)), 'type': 'revised'}}
        # write versioned copy into working dir
        out = self.galaxy_working_dir / f"{p.stem}_v{int(revision)}.glb"
        try:
            blob = gltf.binary_blob()
            gltf.set_binary_blob(blob)
            gltf.save_binary(str(out))
        except Exception:
            gltf.save(str(out.with_suffix('.gltf')))
            out = out.with_suffix('.gltf')
        return str(out)

