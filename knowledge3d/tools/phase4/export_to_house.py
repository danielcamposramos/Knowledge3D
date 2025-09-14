import json
import os
from datetime import datetime


class HouseExporter:
    def __init__(self):
        self.library_registry_path = "viewer/public/library/library_registry.json"
        self.garden_registry_path = "viewer/public/knowledge_garden/garden_registry.json"
        self.bathtub_registry_path = "viewer/public/bathtub/bathtub_registry.json"

    def export_to_library(self, star: dict, title: str) -> bool:
        if not star or 'embedding' not in star:
            return False
        book = {
            'id': f"book_{star.get('id')}",
            'title': title,
            'embedding': star['embedding'],
            'media_types': star.get('media_types', []),
            'shape_type': star.get('shape_type', 'cube'),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'source_star_id': star.get('id'),
        }
        reg = self._load_registry(self.library_registry_path, kind='library')
        reg['books'].append(book)
        self._save_registry(reg, self.library_registry_path)
        return True

    def export_to_garden(self, star: dict, domain: str) -> bool:
        if not star or 'embedding' not in star:
            return False
        tree = {
            'id': f"tree_{star.get('id')}",
            'domain': domain,
            'embedding': star['embedding'],
            'media_types': star.get('media_types', []),
            'shape_type': star.get('shape_type', 'icosahedron'),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'source_star_id': star.get('id'),
            'complexity': float(len(star['embedding'])) * 0.1,
            'is_chiral': bool(star.get('is_chiral', False)),
        }
        reg = self._load_registry(self.garden_registry_path, kind='garden')
        reg['trees'].append(tree)
        self._save_registry(reg, self.garden_registry_path)
        return True

    def export_to_bathtub(self, star: dict) -> bool:
        if not star or 'embedding' not in star:
            return False
        cons = {
            'id': f"consolidation_{star.get('id')}",
            'embedding': star['embedding'],
            'media_types': star.get('media_types', []),
            'shape_type': star.get('shape_type', 'sphere'),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'source_star_id': star.get('id'),
            'status': 'pending',
            'consolidation_params': {
                'compression_factor': 0.8,
                'honesty_threshold': 0.7,
                'fractal_depth': 3,
            },
        }
        reg = self._load_registry(self.bathtub_registry_path, kind='bathtub')
        reg['pending_consolidations'].append(cons)
        self._save_registry(reg, self.bathtub_registry_path)
        return True

    def _load_registry(self, path: str, kind: str) -> dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if kind == 'library':
            return {'books': []}
        if kind == 'garden':
            return {'trees': [], 'center_position': [0.0, 0.0, 0.0], 'current_radius': 10.0}
        if kind == 'bathtub':
            return {'pending_consolidations': [], 'completed_consolidations': []}
        return {}

    def _save_registry(self, reg: dict, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(reg, f, indent=2)

