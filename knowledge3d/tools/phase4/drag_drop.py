import json
import os
from typing import List, Tuple, Optional


class DragDropHandler:
    def __init__(self, galaxy_registry_path: str, workshop_registry_path: str):
        self.galaxy_registry_path = galaxy_registry_path
        self.workshop_registry_path = workshop_registry_path
        self.galaxy_registry = self._load_registry(galaxy_registry_path)
        self.workshop_registry = self._load_registry(workshop_registry_path)

    def _load_registry(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"objects": [], "center_position": [0.0, 0.0, 0.0], "current_radius": 5.0}

    def _save_registry(self, registry: dict, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2)

    def drag_star_to_workshop(self, star_id: str, position: Tuple[float, float, float]) -> bool:
        star = next((s for s in self.galaxy_registry.get('objects', []) if s.get('id') == star_id), None)
        if not star:
            return False
        ws_star = dict(star)
        ws_star['position'] = list(position)
        ws_star['rotation'] = [0.0, 0.0, 0.0]
        ws_star['id'] = f"workshop_{star_id}"
        self.workshop_registry['objects'].append(ws_star)
        self._save_registry(self.workshop_registry, self.workshop_registry_path)
        return True

    def fuse_stars(self, star_ids: List[str]) -> Optional[dict]:
        stars = []
        for sid in star_ids:
            st = next((s for s in self.workshop_registry.get('objects', []) if s.get('id') == sid), None)
            if not st:
                return None
            stars.append(st)
        if len(stars) < 2:
            return None
        hybrid_embedding = self._average_embeddings([s.get('embedding', []) for s in stars])
        if not hybrid_embedding:
            return None
        shape_types = [s.get('shape_type', 'tetrahedron') for s in stars]
        hybrid_shape = self._get_most_complex_shape(shape_types)
        fused_star = {
            'id': f"fused_{'_'.join([s['id'] for s in stars])}",
            'shape_type': hybrid_shape,
            'embedding': hybrid_embedding,
            'position': self._average_positions([s.get('position', [0,0,0]) for s in stars]),
            'rotation': [0.0, 0.0, 0.0],
            'media_types': self._merge_media_types([s.get('media_types', []) for s in stars]),
            'is_fused': True,
            'source_stars': star_ids,
        }
        self.workshop_registry['objects'].append(fused_star)
        self._save_registry(self.workshop_registry, self.workshop_registry_path)
        return fused_star

    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        if not embeddings or not embeddings[0]:
            return []
        dim = len(embeddings[0])
        avg = [0.0] * dim
        count = 0
        for emb in embeddings:
            if len(emb) == dim:
                for i, v in enumerate(emb):
                    avg[i] += float(v)
                count += 1
        if count == 0:
            return []
        return [v / count for v in avg]

    def _get_most_complex_shape(self, shape_types: List[str]) -> str:
        complexity_order = [
            'tetrahedron','cube','octahedron','icosahedron','dodecahedron',
            'triangular_prism','pentagonal_prism','rhombic_dodecahedron',
            'truncated_icosahedron','snub_dodecahedron','great_rhombicuboctahedron',
            'omnitruncated_icosahedron','hypersphere_projection'
        ]
        best = 'tetrahedron'; best_idx = -1
        for s in shape_types:
            if s in complexity_order and complexity_order.index(s) > best_idx:
                best = s; best_idx = complexity_order.index(s)
        return best

    def _average_positions(self, positions: List[List[float]]) -> List[float]:
        if not positions:
            return [0.0, 0.0, 0.0]
        sx=sy=sz=0.0
        for p in positions:
            sx += float(p[0]); sy += float(p[1]); sz += float(p[2])
        n = float(len(positions))
        return [sx/n, sy/n, sz/n]

    def _merge_media_types(self, lists: List[List[str]]) -> List[str]:
        out = set()
        for ls in lists:
            for m in ls: out.add(str(m))
        return sorted(out)

