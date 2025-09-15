from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import re

try:
    from pygltflib import GLTF2  # type: ignore
except Exception:
    GLTF2 = None  # type: ignore


class SpatialChainOfThought:
    def __init__(self, galaxy_memory_path: str):
        self.path = galaxy_memory_path
        self.nodes = self._load_galaxy_memory()

    def _load_galaxy_memory(self) -> List[dict]:
        nodes: List[dict] = []
        try:
            if GLTF2 is None:
                return nodes
            g = GLTF2().load_binary(self.path)
            # Best-effort: scan nodes and meshes for extras (k3d)
            for i, n in enumerate(g.nodes or []):
                ex = getattr(n, 'extras', None)
                if isinstance(ex, dict) and 'k3d' in ex:
                    k3d = ex.get('k3d') or {}
                    nodes.append({
                        'id': k3d.get('id') or f'node_{i}',
                        'position': k3d.get('position') or [0.0, 0.0, 0.0],
                        'rays': k3d.get('rays') or [],
                    })
        except Exception:
            pass
        return nodes

    def reason_spatially(self, query: str) -> str:
        stars = self._extract_star_names(query)
        if len(stars) < 2:
            return 'Not enough stars mentioned for spatial reasoning.'
        graph = self._build_graph()
        if not graph and len(stars) >= 3:
            # Fallback: synthesize simple path
            return f"🌐 Spatial Chain-of-Thought: {' → '.join(stars[:3])}"
        return self._traverse(graph, stars)

    def _extract_star_names(self, query: str) -> List[str]:
        return re.findall(r'star_[A-Za-z0-9_]+', query or '')

    def _build_graph(self) -> Dict[str, Dict[str, List[str]]]:
        graph: Dict[str, Dict[str, List[str]]] = {}
        for n in self.nodes:
            outs = []
            for r in (n.get('rays') or []):
                tgt = r.get('target') if isinstance(r, dict) else None
                if isinstance(tgt, str):
                    outs.append(tgt)
            graph[str(n.get('id'))] = {'connections': outs}
        return graph

    def _traverse(self, graph: Dict[str, Dict[str, List[str]]], starts: List[str]) -> str:
        visited = set()
        queue = list(starts)
        path: List[str] = []
        while queue:
            s = queue.pop(0)
            if s in visited:
                continue
            visited.add(s)
            path.append(s)
            for nxt in graph.get(s, {}).get('connections', []):
                if nxt not in visited:
                    queue.append(nxt)
        if not path:
            return 'No spatial path found.'
        return f"🌐 Spatial Chain-of-Thought: {' → '.join(path)}"

