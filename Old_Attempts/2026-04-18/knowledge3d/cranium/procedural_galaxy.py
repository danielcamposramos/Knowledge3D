"""
Procedural Galaxy storage for compact opcode programs.

Programs are stored under /K3D/Knowledge3D.local/procedural_galaxy by default
to keep large binary data outside Git. Each program corresponds to a key
(e.g., character glyph) and can be executed to regenerate embeddings on demand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .procedural_compiler import ProceduralCompiler, ProceduralProgram


DEFAULT_GALAXY_ROOT = Path("/K3D/Knowledge3D.local/procedural_galaxy")


class ProceduralGalaxy:
    """Disk-backed store for procedural knowledge programs."""

    def __init__(self, root: Optional[Path] = None, compiler: Optional[ProceduralCompiler] = None) -> None:
        self.root = Path(root) if root else DEFAULT_GALAXY_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.compiler = compiler or ProceduralCompiler()

    # ------------------------------------------------------------------ #
    # Paths / metadata helpers
    # ------------------------------------------------------------------ #
    def _sanitize_key(self, key: str) -> str:
        safe = "".join(c for c in key if c.isalnum() or c in ("-", "_"))
        return safe or "unknown"

    def _program_path(self, key: str) -> Path:
        safe = self._sanitize_key(key)
        return self.root / f"{safe}.ppr"

    def _program_metadata_path(self, key: str) -> Path:
        """Path to per-program metadata (multilingual, RPN programs, etc.)."""
        safe = self._sanitize_key(key)
        return self.root / f"{safe}.meta.json"

    def _meta_path(self) -> Path:
        return self.root / "galaxy_meta.json"

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def store_program(
        self,
        key: str,
        program_bytes: bytes,
        compression_ratio: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store procedural program with optional multilingual metadata.

        Args:
            key: Character/symbol key (e.g., 'a', 'ç', '+')
            program_bytes: Compiled procedural program bytes
            compression_ratio: Achieved compression ratio
            metadata: Optional metadata dict with:
                - math_rpn: What it does (execution bytecode or "")
                - languages: ISO 639-1 codes ['en', 'pt', 'es', ...]
                - glyph_count: Number of glyph variants stored
                - glyphs: List of glyph metadata (visual_rpn, font metadata)
                - timestamp: UTC timestamp
        """
        # Store program bytes
        path = self._program_path(key)
        path.write_bytes(program_bytes)
        logging.info(
            "ProceduralGalaxy: stored %s (%d bytes, %.1f:1 compression)",
            key, len(program_bytes), compression_ratio
        )

        # Store per-program metadata (multilingual, RPN programs)
        if metadata:
            meta_path = self._program_metadata_path(key)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        # Update global galaxy metadata (compression stats)
        self._update_meta(key, len(program_bytes), compression_ratio)

    def load_program(self, key: str) -> ProceduralProgram:
        path = self._program_path(key)
        if not path.exists():
            raise FileNotFoundError(f"No procedural program stored for key '{key}'")
        payload = path.read_bytes()
        return ProceduralProgram.from_bytes(payload)

    def load_metadata(self, key: str) -> Optional[Dict]:
        """
        Load multilingual metadata for a stored program.

        Returns:
            Dictionary with visual_rpn, math_rpn, languages, timestamp
            None if no metadata exists
        """
        meta_path = self._program_metadata_path(key)
        if not meta_path.exists():
            return None

        return json.loads(meta_path.read_text(encoding='utf-8'))

    def execute_program(self, key: str) -> np.ndarray:
        program_path = self._program_path(key)
        payload = program_path.read_bytes()
        return self.compiler.decompile_program(payload)

    def get_compression_stats(self) -> Dict[str, float]:
        meta_path = self._meta_path()
        if not meta_path.exists():
            return {}
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # ------------------------------------------------------------------ #
    def _update_meta(self, key: str, byte_size: int, compression_ratio: float) -> None:
        meta_path = self._meta_path()
        data: Dict[str, Dict[str, float]] = {}
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logging.warning("ProceduralGalaxy: corrupt metadata, resetting.")
        data[key] = {"bytes": float(byte_size), "compression": float(compression_ratio)}
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
