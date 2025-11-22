"""
GalaxyUniverseLoader (stub) for default + on-demand galaxy loading policy.

Policy (meaning-first):
- Default-load: base galaxies (text/visual/audio/reasoning), word-meaning galaxy,
  math-symbol galaxy, punctuation galaxy.
- On-demand: letter-meaning galaxies per detected script (user hint + document).
- Separation: math symbols stay in math galaxy even if glyphs resemble letters; letters
  stay in script-specific letter galaxies.

This loader does not perform actual GLB/ProceduralGalaxy IO; wire `load_glb`/`unload_glb`
to project-specific mechanisms when available.
"""

from __future__ import annotations

from typing import Dict, List, Set


class GalaxyUniverseLoader:
    def __init__(self, universe_capacity_mb: int = 200, enable_sublex: bool = False) -> None:
        self.capacity_mb = universe_capacity_mb
        self.loaded: Set[str] = set()
        self.registry: Dict[str, str] = {}
        self.enable_sublex = enable_sublex
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        defaults = [
            "base_text",
            "base_visual",
            "base_audio",
            "base_reasoning",
            "word_meaning",
            "phrase_meaning",
            "math_symbol",
            "punctuation",
            "user_phrase",  # empty user-defined galaxy (runtime writable)
        ]
        if self.enable_sublex:
            defaults.extend(["syllables_latin", "morphemes_latin"])
        for g in defaults:
            self._load_glb(g)

    def _load_glb(self, galaxy_id: str) -> None:
        # TODO: wire to actual GLB/ProceduralGalaxy loading via ctypes loader
        self.loaded.add(galaxy_id)

    def _unload_glb(self, galaxy_id: str) -> None:
        # TODO: wire to actual unload + optional consolidation.
        self.loaded.discard(galaxy_id)

    def load_for_context(self, user_lang: str, document_langs: List[str]) -> Set[str]:
        scripts = set()
        for lang in [user_lang, *document_langs]:
            if lang in {"en", "pt", "es", "fr", "de", "it", "nl", "sv"}:
                scripts.add("latin_letter")
            elif lang in {"ru", "uk", "bg", "sr", "mk", "be"}:
                scripts.add("cyrillic_letter")
            elif lang in {"ar", "fa", "ur", "ps"}:
                scripts.add("arabic_letter")
            elif lang in {"zh", "ja", "ko"}:
                scripts.add("cjk_letter")
        for script in scripts:
            if script not in self.loaded:
                self._load_glb(script)
        return set(self.loaded)
