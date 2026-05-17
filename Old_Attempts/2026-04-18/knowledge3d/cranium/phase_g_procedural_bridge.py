"""
Phase G bridge that converts freshly trained character embeddings into
procedural programs and stores them inside the Procedural Galaxy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .procedural_compiler import ProceduralCompiler
from .procedural_galaxy import ProceduralGalaxy


class PhaseGProceduralBridge:
    """Bridge Phase G training events to procedural storage."""

    def __init__(
        self,
        compiler: Optional[ProceduralCompiler] = None,
        galaxy: Optional[ProceduralGalaxy] = None,
    ) -> None:
        self.compiler = compiler or ProceduralCompiler()
        self.galaxy = galaxy or ProceduralGalaxy(compiler=self.compiler)

    def on_character_trained(self, char: str, embedding: np.ndarray) -> None:
        """Compile trained embedding into procedural program and persist it."""
        program = self.compiler.compile_embedding(embedding)
        program_bytes = program.to_bytes()
        compression = float(embedding.nbytes) / max(1, len(program_bytes))
        self.galaxy.store_program(char, program_bytes, compression_ratio=compression)
        logging.info(
            "PhaseGProceduralBridge: char=%s dims=%d -> program=%d bytes (%.1f:1)",
            char,
            embedding.size,
            len(program_bytes),
            compression,
        )

    def materialize_character(self, char: str) -> np.ndarray:
        """Reconstruct a character embedding from its procedural representation."""
        return self.galaxy.execute_program(char)

