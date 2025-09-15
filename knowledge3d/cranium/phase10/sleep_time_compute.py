from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class _Model:
    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes


class _GLBLoaderFallback:
    def load(self, path: str) -> _Model:  # pragma: no cover
        # Fallback: return empty model if parsing not available or file missing
        try:
            p = Path(path)
            if not p.exists():
                return _Model([])
        except Exception:
            pass
        return _Model([])


def _load_glb(path: str) -> _Model:
    try:
        from pygltflib import GLTF2  # type: ignore
        p = Path(path)
        if not p.exists():
            return _Model([])
        gltf = GLTF2().load(str(p))
        nodes: List[Dict[str, Any]] = []
        for node in (gltf.nodes or []):
            # Normalize to dict with extras for our traversal
            extras = getattr(node, "extras", None)
            # Ensure dict
            if hasattr(extras, "to_dict"):
                try:
                    extras = extras.to_dict()
                except Exception:
                    extras = dict(extras)
            if extras is None:
                extras = {}
            nodes.append({
                "name": getattr(node, "name", None),
                "extras": extras,
            })
        return _Model(nodes)
    except Exception:
        return _GLBLoaderFallback().load(path)


class SleepTimeCompute:
    def __init__(self, house_path: str, galaxy_path: str, output_path: str | None = None, material_dir: str | None = None):
        self.house_path = Path(house_path)
        self.galaxy_path = Path(galaxy_path)
        self.output_path = Path(output_path) if output_path else self.house_path.parent / "house_post_sleep.glb"
        self.material_dir = Path(material_dir) if material_dir else self.house_path.parent / "materialized_objects"
        self.material_dir.mkdir(parents=True, exist_ok=True)
        self.house: Optional[Dict[str, Any]] = None
        self.galaxy: Optional[List[Dict[str, Any]]] = None

    def load_house(self) -> Dict[str, Any]:
        """Load House GLB — extract zones, rays, embeddings (best-effort)."""
        model = _load_glb(str(self.house_path))
        house_data: Dict[str, Any] = {
            'zones': [],
            'rays': [],
            'nodes': model.nodes,
        }
        for node in model.nodes:
            extras = node.get('extras') or {}
            k3d = extras.get('k3d') if isinstance(extras, dict) else None
            if not isinstance(k3d, dict):
                continue
            if k3d.get('type') == 'zone':
                house_data['zones'].append({
                    'id': k3d.get('id'),
                    'name': k3d.get('name') or node.get('name'),
                    'position': list(k3d.get('position', [0.0, 0.0, 0.0])),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                })
            elif k3d.get('type') == 'ray':
                house_data['rays'].append({
                    'id': k3d.get('id'),
                    'origin_zone': k3d.get('origin_zone'),
                    'target_zone': k3d.get('target_zone'),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                })
        return house_data

    def load_galaxy(self) -> List[Dict[str, Any]]:
        """Load Galaxy GLB — extract stars, honesty scores, chat logs, reflections (best-effort)."""
        model = _load_glb(str(self.galaxy_path))
        stars: List[Dict[str, Any]] = []
        for node in model.nodes:
            extras = node.get('extras') or {}
            k3d = extras.get('k3d') if isinstance(extras, dict) else None
            if not isinstance(k3d, dict):
                continue
            if k3d.get('type') == 'star':
                stars.append({
                    'id': k3d.get('id'),
                    'position': list(k3d.get('position', [0.0, 0.0, 0.0])),
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                    'embedding_entropy': float(k3d.get('embedding_entropy', 0.0)),
                    'chat_history': list(k3d.get('chat_history', [])),
                    'self_reflections': list(k3d.get('self_reflections', [])),
                    'generated_shapes': list(k3d.get('generated_shapes', [])),
                    'connected_stars': list(k3d.get('connected_stars', [])),
                    'embedding': list(k3d.get('embedding', [])),
                })
        return stars

    def _write_json(self, path: Path, data: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

    def materialize_chat_history(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize chat history into a book object (JSON for now)."""
        if not star.get('chat_history'):
            return None
        book_id = f"book_chat_{star.get('id')}_{int(time.time())}"
        book_path = self.material_dir / f"{book_id}.json"
        book_data = {
            'type': 'chat_history_book',
            'title': f"Chat Log — Star {star.get('id')}",
            'author': 'AI Self',
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'content': star.get('chat_history', []),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 3 (Library)',
        }
        return self._write_json(book_path, book_data)

    def materialize_diary_entry(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize self-reflections into diary entries (JSON)."""
        if not star.get('self_reflections'):
            return None
        diary_id = f"diary_{star.get('id')}_{int(time.time())}"
        diary_path = self.material_dir / f"{diary_id}.json"
        diary_data = {
            'type': 'diary_entry',
            'title': f"Self-Reflection — {datetime.now().strftime('%Y-%m-%d')}",
            'author': 'AI Self',
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'content': star.get('self_reflections', []),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 7 (Mirror Room)',
        }
        return self._write_json(diary_path, diary_data)

    def materialize_fractal_tree(self, star: Dict[str, Any]) -> Optional[str]:
        """Materialize knowledge as a fractal tree metadata JSON (GLB stub)."""
        if float(star.get('honesty_score', 0.0)) < 0.6:
            return None
        tree_id = f"tree_{star.get('id')}_{int(time.time())}"
        tree_meta_path = self.material_dir / f"{tree_id}.json"
        tree_data = {
            'type': 'fractal_tree',
            'name': f"Knowledge Tree — Star {star.get('id')}",
            'created_at': datetime.now().isoformat(),
            'honesty_score': star.get('honesty_score', 0.0),
            'branch_density': int(1.618 * float(star.get('honesty_score', 0.0)) * 10),
            'embedding': star.get('embedding', []),
            'zone_placement': 'Zone 5 (Knowledge Garden)',
        }
        return self._write_json(tree_meta_path, tree_data)

    def compute_nightly_adjustments(self) -> Dict[str, Any]:
        """Adjust House zones, prune rays, and materialize knowledge into permanent objects."""
        self.house = self.load_house()
        self.galaxy = self.load_galaxy()

        adjustments: Dict[str, Any] = {
            'zone_shifts': [],
            'ray_adjustments': [],
            'pruned_rays': [],
            'materialized_objects': [],
        }

        zones = self.house.get('zones', []) if self.house else []
        rays = self.house.get('rays', []) if self.house else []
        stars = self.galaxy or []
        star_by_id = {s.get('id'): s for s in stars}

        # Adjust zone positions based on honesty-weighted alignment to star positions
        for z in zones:
            sid = z.get('id')
            star = star_by_id.get(sid)
            if not star:
                continue
            h = float(star.get('honesty_score', 0.0))
            if h <= 0.5:
                continue
            old_pos = list(z.get('position', [0.0, 0.0, 0.0]))
            st_pos = list(star.get('position', [0.0, 0.0, 0.0]))
            shift = [(st_pos[i] - old_pos[i]) * h for i in range(3)]
            new_pos = [old_pos[i] + shift[i] for i in range(3)]
            z['position'] = new_pos
            adjustments['zone_shifts'].append({
                'zone_id': sid,
                'old_position': old_pos,
                'new_position': new_pos,
                'shift_vector': shift,
                'honesty_score': h,
            })

        # Adjust ray origins to updated zone positions; prune low-honesty rays
        kept_rays = []
        for r in rays:
            h = float(r.get('honesty_score', 0.0))
            if h < 0.3:
                adjustments['pruned_rays'].append(r.get('id'))
                continue
            origin_id = r.get('origin_zone')
            origin_zone = next((zz for zz in zones if zz.get('id') == origin_id), None)
            if origin_zone:
                adjustments['ray_adjustments'].append({
                    'ray_id': r.get('id'),
                    'origin_zone': origin_id,
                    'new_origin_position': origin_zone.get('position', [0.0, 0.0, 0.0]),
                })
            kept_rays.append(r)
        if self.house is not None:
            self.house['rays'] = kept_rays

        # Materialize knowledge objects per high-honesty stars
        for s in stars:
            h = float(s.get('honesty_score', 0.0))
            if h >= 0.5:
                # Chat history → book
                p = self.materialize_chat_history(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'chat_history_book', 'path': p, 'zone': 'Zone 3 (Library)', 'star_id': s.get('id')})
                # Self-reflections → diary entry
                p = self.materialize_diary_entry(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'diary_entry', 'path': p, 'zone': 'Zone 7 (Mirror Room)', 'star_id': s.get('id')})
                # Knowledge tree → fractal
                p = self.materialize_fractal_tree(s)
                if p:
                    adjustments['materialized_objects'].append({'type': 'fractal_tree', 'path': p, 'zone': 'Zone 5 (Knowledge Garden)', 'star_id': s.get('id')})

        return adjustments

    def save_house(self) -> str:
        """Save modified House GLB (stub). Logs adjustments and materializations."""
        adjustments = self.compute_nightly_adjustments()
        # Always log to central logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / 'sleep_time_adjustments.json'
        with log_path.open('w', encoding='utf-8') as f:
            json.dump(adjustments, f, ensure_ascii=False, indent=2)
        print(f"🌙 Sleep-Time Compute: Adjustments logged to {log_path}")
        print(f"📚 Materialized {len(adjustments['materialized_objects'])} permanent objects.")
        print(f"🌙 Modified House GLB would be saved to {self.output_path} (GLB writing stubbed for Phase 10.7)")
        return str(self.output_path)

    def run(self) -> None:
        """Execute sleep-time compute pipeline."""
        print("🌙 Initiating Sleep-Time Compute & Permanent Materialization...")
        out = self.save_house()
        print(f"🌙 Sleep-Time Compute Complete. House ready for reload at {out}")
        print(f"🏛️  All materialized objects stored in: {self.material_dir}")
