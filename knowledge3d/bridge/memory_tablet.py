"""
Minimal MemoryTablet wiring to GalaxyUniverseLoader with optional GLB load via ctypes path.
This demonstrates load policy (phrase/user phrase, sublexicals) and can ingest .glb files
containing extras.k3d nodes.
"""

from __future__ import annotations

from typing import List

from .galaxy_loader import GalaxyUniverseLoader
from .glb_ctypes_loader import load_stars_from_glb


class MemoryTablet:
    def __init__(self, universe_capacity_mb: int = 200, enable_sublex: bool = False) -> None:
        self.loader = GalaxyUniverseLoader(universe_capacity_mb=universe_capacity_mb, enable_sublex=enable_sublex)
        self.cache = {}

    def ensure_galaxies(self, user_lang: str, document_langs: List[str]) -> None:
        self.loader.load_for_context(user_lang, document_langs)

    def list_loaded(self) -> List[str]:
        return sorted(self.loader.loaded)

    def load_glb(self, galaxy_id: str, path: str) -> None:
        """
        Load a GLB with extras.k3d stars and mark galaxy as loaded.
        """
        stars = load_stars_from_glb(Path(path))
        self.cache[galaxy_id] = stars
        self.loader.loaded.add(galaxy_id)
