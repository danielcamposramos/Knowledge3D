from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

from pygltflib import GLTF2  # type: ignore


class SelfCurriculumEngine:
    def __init__(self, galaxy_path: str):
        self.galaxy_path = str(galaxy_path)
        self._templates = [
            "What shape should represent low-honesty concept: {}?",
            "Why is honesty score low for star: {}?",
            "How to increase honesty of modality fusion in: {}?",
            "What ray thickness should encode low-resolution embedding: {}?",
            "Which zone should store uncertain knowledge: {}?",
        ]

    def _load_stars(self) -> List[Dict[str, Any]]:
        try:
            gltf = GLTF2().load(self.galaxy_path)
        except Exception:
            return []
        stars: List[Dict[str, Any]] = []
        for node in (gltf.nodes or []):
            extras = getattr(node, 'extras', None)
            if hasattr(extras, 'to_dict'):
                try:
                    extras = extras.to_dict()
                except Exception:
                    extras = dict(extras)
            if isinstance(extras, dict):
                k3d = extras.get('k3d') if isinstance(extras.get('k3d'), dict) else None
                if isinstance(k3d, dict) and str(k3d.get('type') or '').lower() == 'star':
                    stars.append({
                        'id': k3d.get('id') or node.name,
                        'honesty_score': float(k3d.get('honesty_score', 0.5)),
                        'embedding_entropy': float(k3d.get('embedding_entropy', 0.0)),
                    })
        return stars

    def identify_knowledge_gaps(self, honesty_cut: float = 0.5) -> List[Dict[str, Any]]:
        stars = self._load_stars()
        return [s for s in stars if float(s.get('honesty_score', 0.5)) < honesty_cut]

    def generate_training_queries(self, gaps: List[Dict[str, Any]], num_queries: int = 5) -> List[str]:
        qs: List[str] = []
        for g in gaps[:num_queries]:
            tmpl = self._templates[len(qs) % len(self._templates)]
            sid = g.get('id') or '(unknown)'
            qs.append(tmpl.format(sid))
        return qs

    def run_self_curriculum(self) -> List[str]:
        gaps = self.identify_knowledge_gaps()
        if not gaps:
            return []
        return self.generate_training_queries(gaps)

