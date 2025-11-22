"""
Upsert bridge for procedural-first, meaning-first stars using ProceduralGalaxy +
minimal glTF export (extras.k3d only). Embeddings, when present under embeddings['raw'],
are compressed via ProceduralCompiler (PD codec). This is a stopgap until a full
ctypes GLB pipeline is wired; the call surface stays stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy
from knowledge3d.cranium.procedural_compiler import ProceduralCompiler
from knowledge3d.bridge.glb_ctypes_loader import save_stars_to_glb, GLTF2  # type: ignore


class GalaxyUpsertBridge:
    """
    Minimal JSONL persistence; swap out methods to target ProceduralGalaxy/House.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        # Use a dedicated ProceduralGalaxy root as a drop-in until full GLB/ctypes wiring exists.
        self.output_dir = Path(output_dir) if output_dir else Path(
            "/K3D/Knowledge3D.local/procedural_galaxy_pending"
        )
        self.pg = ProceduralGalaxy(root=self.output_dir)
        self.compiler = ProceduralCompiler()

    def _write_jsonl(self, name: str, rows: List[Dict]) -> Path:
        path = self.output_dir / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def export_gltf(self, name: str, stars: List[Dict]) -> Path:
        """
        Export glTF/GLB with extras.k3d using pygltflib if available.
        """
        out = self.output_dir / f"{name}.glb"
        if GLTF2 is None:
            # Fallback to JSON glTF
            gltf_json = self.output_dir / f"{name}.gltf"
            gltf_json.parent.mkdir(parents=True, exist_ok=True)
            gltf = {
                "asset": {"version": "2.0", "generator": "k3d-ingestion-upsert"},
                "scene": 0,
                "scenes": [{"nodes": list(range(len(stars)))}],
                "nodes": [],
            }
            for idx, star in enumerate(stars):
                gltf["nodes"].append(
                    {
                        "name": star.get("letter_concept")
                        or star.get("meaning_id")
                        or star.get("symbol_concept")
                        or star.get("phrase_id")
                        or f"star_{idx}",
                        "extras": {"k3d": star},
                    }
                )
            gltf_json.write_text(json.dumps(gltf, ensure_ascii=False, indent=2), encoding="utf-8")
            return gltf_json
        return save_stars_to_glb(stars, out)

    def _store_star(self, key: str, star: Dict) -> None:
        """
        Store a star as procedural bytes (JSON-serialized) plus metadata.
        If an embedding vector is present under embeddings['raw'], compress via ProceduralCompiler.
        """
        embeddings = star.get("embeddings", {})
        raw_vec = embeddings.get("raw")
        if raw_vec:
            # Expect raw_vec as list of floats; compile to procedural program
            import numpy as np

            vec = np.array(raw_vec, dtype=np.float32)
            program = self.compiler.compile_embedding(vec)
            program_bytes = program.to_bytes()
            compression = float(vec.nbytes) / max(1, len(program_bytes))
            meta = star.copy()
            meta["embeddings"]["matryoshka"] = None
            self.pg.store_program(key, program_bytes, compression_ratio=compression, metadata=meta)
        else:
            payload = json.dumps(star, ensure_ascii=False).encode("utf-8")
            compression = 1.0
            self.pg.store_program(key, payload, compression_ratio=compression, metadata=star)

    def upsert_letter_meanings(self, stars: List[Dict]) -> Path:
        """Persist letter-meaning stars (one per meaning, many glyph variants)."""
        for star in stars:
            key = star.get("letter_concept", "unknown_letter")
            self._store_star(key, star)
        return self._write_jsonl("letter_meanings", stars)

    def upsert_word_meanings(self, stars: List[Dict]) -> Path:
        """Persist word-meaning stars (sense-disambiguated, compositional)."""
        for star in stars:
            key = star.get("meaning_id", "unknown_word")
            self._store_star(key, star)
        return self._write_jsonl("word_meanings", stars)

    def upsert_math_symbols(self, stars: List[Dict]) -> Path:
        """Persist math symbol stars (operators/constants with math_rpn)."""
        for star in stars:
            key = star.get("symbol_concept", star.get("symbol", "unknown_math"))
            self._store_star(key, star)
        return self._write_jsonl("math_symbols", stars)

    def upsert_audio_to_letters(self, audio_entries: List[Dict]) -> Path:
        """Persist audio attachment payloads for letter stars."""
        return self._write_jsonl("audio_letter_attach", audio_entries)

    def upsert_phrase_meanings(self, stars: List[Dict]) -> Path:
        """Persist phrase-meaning stars (idioms/multiword expressions)."""
        for star in stars:
            key = star.get("phrase_id", "unknown_phrase")
            self._store_star(key, star)
        return self._write_jsonl("phrase_meanings", stars)
