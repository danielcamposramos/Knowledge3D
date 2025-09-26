from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pygltflib import GLTF2  # type: ignore


class GalaxyStateSerializer:
    def __init__(self, galaxy_path: str, state_path: str = "viewer/public/galaxy/galaxy_state.json"):
        self.galaxy_path = Path(galaxy_path)
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def serialize_galaxy_state(self) -> Dict[str, Any]:
        """Serialize entire Galaxy state (stars + rays) from GLB extras.k3d into JSON."""
        try:
            gltf = GLTF2().load(str(self.galaxy_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load Galaxy GLB: {self.galaxy_path} ({e})")

        state: Dict[str, Any] = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'stars': [],
            'rays': [],
            'metadata': {
                'node_count': len(gltf.nodes or []),
                'scene_count': len(gltf.scenes or []),
            },
        }
        for node in (gltf.nodes or []):
            extras = getattr(node, 'extras', None)
            if hasattr(extras, 'to_dict'):
                try:
                    extras = extras.to_dict()
                except Exception:
                    extras = dict(extras)
            if not isinstance(extras, dict):
                continue
            k3d = extras.get('k3d') if isinstance(extras.get('k3d'), dict) else None
            if not isinstance(k3d, dict):
                continue
            rec = {
                'id': k3d.get('id'),
                'type': k3d.get('type'),
                'position': k3d.get('position', [0.0, 0.0, 0.0]),
                'embedding': k3d.get('embedding', []),
                'honesty_score': k3d.get('honesty_score', 0.5),
                'connected_stars': k3d.get('connected_stars', []),
                'chat_history': k3d.get('chat_history', []),
                'self_reflections': k3d.get('self_reflections', []),
                'generated_shapes': k3d.get('generated_shapes', []),
                'ray_thickness': k3d.get('ray_thickness', 0.02),
                'modality': k3d.get('modality', 'text'),
            }
            t = str(k3d.get('type') or '').lower()
            if t == 'star':
                state['stars'].append(rec)
            elif t == 'ray':
                state['rays'].append(rec)

        with self.state_path.open('w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"💾 Galaxy State Serialized: {len(state['stars'])} stars, {len(state['rays'])} rays → {self.state_path}")
        return state

    def deserialize_galaxy_state(self) -> Optional[Dict[str, Any]]:
        """Load Galaxy state if present (no GLB reconstruction here)."""
        if not self.state_path.exists():
            print("🆕 No saved Galaxy state found — starting fresh.")
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding='utf-8'))
            print(f"📂 Galaxy State Loaded: {len(data.get('stars', []))} stars, {len(data.get('rays', []))} rays from {self.state_path}")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load Galaxy state: {e}")
            return None
